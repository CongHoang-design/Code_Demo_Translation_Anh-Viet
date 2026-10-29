# Script huấn luyện mô hình Transformer dịch máy trên bộ dữ liệu IWSLT (định dạng parquet).
# Luồng chính: train_model -> get_ds (tokenizer, lọc độ dài, DataLoader) -> get_model -> vòng lặp epoch.
import html
import warnings

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # Sinh câu dịch cho MỘT câu nguồn bằng greedy search: mỗi bước chọn token có xác suất cao nhất.
    # Vào: source (1, S), source_mask (1, 1, S). Ra: tensor 1 chiều các id, bắt đầu bằng [SOS].
    # Greedy đơn giản và nhanh nhưng có thể bỏ lỡ câu dịch tốt hơn so với beam search.
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Encode câu nguồn đúng một lần; mọi bước decode dùng lại kết quả này.
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        # Dừng khi đạt độ dài tối đa để tránh lặp vô hạn nếu model không sinh [EOS].
        if decoder_input.size(1) == max_len:
            break

        # Mask nhân quả được dựng lại theo độ dài hiện tại của chuỗi đang sinh.
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Chỉ vị trí cuối của chuỗi mới dự đoán token kế tiếp.
        prob = model.linear(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    # Dịch thử vài câu trong tập validation và in ra để kiểm tra bằng mắt chất lượng model.
    # NOTE: global_state và writer hiện chưa được dùng (chưa ghi metric nào vào TensorBoard).
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # greedy_decode chỉ xử lý một câu mỗi lần, nên DataLoader validation phải có batch_size=1.
            assert encoder_input.size(0) == 1, "Batch phai la 1 cho validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # src_text/tgt_text được DataLoader gom thành list, nên lấy phần tử [0].
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Dùng print_msg (tqdm.write) thay cho print để không làm vỡ thanh tiến trình.
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

def get_all_sentences(ds, lang):
    # Generator để tokenizer học từ vựng mà không phải nạp cả tập dữ liệu vào bộ nhớ.
    for item in ds:
        yield html.unescape(item['translation'][lang])

def get_or_build_tokenizer(config, ds, lang):
    # Cache tokenizer ra file để các lần chạy sau dùng lại.
    # WARNING: nếu file đã tồn tại thì không train lại, dù dữ liệu hay min_frequency đã đổi.
    # Muốn xây lại từ vựng thì phải xóa file tokenizer cũ.
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        # WordLevel: mỗi từ là một token; từ không có trong từ vựng thành [UNK].
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        # min_frequency=2 loại các từ chỉ xuất hiện một lần để từ vựng gọn hơn (đổi lại sẽ có nhiều [UNK]).
        # Token đặc biệt phải khớp với tên mà BilingualDataset tra cứu.
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def load_iwslt_splits(config):
    # Đọc dữ liệu cục bộ từ các file parquet đã tải sẵn, chia theo train/validation/test.
    base = config['parquet_base']
    return load_dataset("parquet", data_files={
        "train": f"{base}/train/0000.parquet",
        "validation": f"{base}/validation/0000.parquet",
        "test": f"{base}/test/0000.parquet",
    })

def filter_by_length(ds, tokenizer_src, tokenizer_tgt, config):
    # BilingualDataset ném lỗi với câu quá dài, nên lọc trước để không dừng giữa chừng.
    # Giữ các câu vừa với seq_len (src cần chỗ cho SOS+EOS, tgt cần chỗ cho SOS hoặc EOS).
    # Điều kiện >= 1 loại luôn các câu rỗng.
    max_src = config['seq_len'] - 2
    max_tgt = config['seq_len'] - 1
    keep = []
    for i, ex in enumerate(ds):
        t = ex['translation']
        n_src = len(tokenizer_src.encode(html.unescape(t[config['lang_src']])).ids)
        n_tgt = len(tokenizer_tgt.encode(html.unescape(t[config['lang_tgt']])).ids)
        if 1 <= n_src <= max_src and 1 <= n_tgt <= max_tgt:
            keep.append(i)
    print(f'Giu {len(keep)}/{len(ds)} cau (seq_len={config["seq_len"]})')
    return ds.select(keep)

def get_ds(config):
    # Chuẩn bị toàn bộ dữ liệu: tokenizer, lọc độ dài, Dataset và DataLoader.
    splits = load_iwslt_splits(config)
    train_ds_raw = splits["train"]
    val_ds_raw = splits["validation"]

    # Từ vựng chỉ học từ tập train (trước khi lọc) để không rò rỉ thông tin từ validation.
    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw, config['lang_tgt'])

    train_ds_raw = filter_by_length(train_ds_raw, tokenizer_src, tokenizer_tgt, config)
    val_ds_raw = filter_by_length(val_ds_raw, tokenizer_src, tokenizer_tgt, config)

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # Validation dùng batch_size=1 vì greedy_decode xử lý từng câu; shuffle để mỗi epoch xem câu khác nhau.
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    # NOTE: số tầng lấy từ config['num_layers'] (mặc định 3, dùng cho thí nghiệm ablation);
    # số head (h), dropout và d_ff dùng mặc định của build_transformer.
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'],
                              n=config.get('num_layers', 3))
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    # eps=1e-9 theo cấu hình huấn luyện Transformer trong paper.
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Khôi phục checkpoint để tiếp tục huấn luyện: cần cả trạng thái optimizer (momentum của Adam)
    # lẫn epoch và global_step để đường cong TensorBoard liền mạch.
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # ignore_index: vị trí [PAD] không tính vào loss.
    # label_smoothing (mặc định 0.1, lấy từ config['label_smoothing']) giảm độ tự tin quá mức của model,
    # giúp tổng quát hóa tốt hơn (theo paper); đặt 0 để tắt khi làm ablation.
    # NOTE: PAD id lấy từ tokenizer_src; chỉ đúng nếu hai tokenizer gán cùng id cho [PAD].
    # NOTE: model.linear đã trả log-softmax, còn CrossEntropyLoss áp dụng log-softmax lần nữa.
    # Log-softmax là phép lũy đẳng nên kết quả loss không đổi; chỉ thừa tính toán.
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=config.get('label_smoothing', 0.1)).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f'Xu ly epoch {epoch:02d}')
        for batch in batch_iterator:
            # Đặt lại chế độ train ở mỗi batch vì run_validation đã chuyển model sang eval().
            model.train()

            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)    # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)    # (B, 1, seq_len, seq_len)

            # Forward pass với bfloat16 tự động trên GPU để tiết kiệm bộ nhớ và tăng tốc; trên CPU thì tắt.
            # Mask nguồn dùng chung cho self-attention của encoder và cross-attention của decoder.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                linear_output = model.linear(decoder_output)  # (B, seq_len, tgt_vocab)

            label = batch['label'].to(device)  # (B, seq_len): id của token đúng ở mỗi vị trí

            # Trải phẳng batch và thời gian để CrossEntropyLoss nhận (B*seq_len, vocab) và (B*seq_len).
            # NOTE: loss tính ngoài khối autocast trên đầu ra bfloat16 của model.
            loss = loss_fn(linear_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})

            # NOTE: flush ở mỗi bước ghi đĩa liên tục, có thể làm chậm khi số bước lớn.
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        # Cuối mỗi epoch: dịch thử vài câu để quan sát chất lượng.
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer)

        # Lưu checkpoint đầy đủ mỗi epoch (model, optimizer, epoch, step) để có thể resume bằng config['preload'].
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
