# Đánh giá checkpoint đã huấn luyện bằng SacreBLEU, chrF và TER.
# Chạy: python evaluate.py --checkpoint weights/tmodel_04.pt --split test
# Kết quả: in ra màn hình và lưu eval_results/<checkpoint>_<split>.json (+ file bản dịch .hyp.txt).
import argparse
import html
import json
import sys
import time
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU, CHRF, TER

from config import get_config
from dataset import causal_mask
from train import get_model, get_or_build_tokenizer, load_iwslt_splits

sys.stdout.reconfigure(encoding='utf-8')


def encode_source(ids, seq_len, sos, eos, pad):
    # Giống BilingualDataset nhưng cắt bớt câu nguồn quá dài thay vì ném lỗi,
    # để có thể đánh giá cả các câu mà quá trình train đã lọc bỏ.
    ids = ids[:seq_len - 2]
    return [sos] + ids + [eos] + [pad] * (seq_len - len(ids) - 2)


@torch.no_grad()
def greedy_decode_batch(model, src, src_mask, sos, eos, max_len, device):
    # Greedy theo batch: mỗi bước chọn token xác suất cao nhất cho mọi câu cùng lúc.
    # Câu nào đã sinh [EOS] thì các bước sau chỉ điền [EOS] (bị cắt khi giải mã văn bản).
    encoder_output = model.encode(src, src_mask)
    batch = src.size(0)
    decoder_input = torch.full((batch, 1), sos, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    while decoder_input.size(1) < max_len:
        tgt_mask = causal_mask(decoder_input.size(1)).to(device)
        out = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
        next_word = model.linear(out[:, -1]).argmax(dim=-1)
        next_word = torch.where(finished, torch.full_like(next_word, eos), next_word)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)
        finished |= next_word == eos
        if finished.all():
            break
    return decoder_input


def score(hyps, refs):
    # Trả về điểm và chữ ký của từng độ đo để báo cáo kèm theo (đúng yêu cầu M2).
    # force=True: dữ liệu IWSLT'15 vốn đã được tách từ sẵn (dấu câu cách rời), nên tắt cảnh báo
    # "forgot to detokenize" của SacreBLEU. Hệ quả: điểm không so trực tiếp với BLEU của bản dịch đã detokenize.
    metrics = {'BLEU': BLEU(force=True), 'chrF': CHRF(), 'TER': TER()}
    result = {}
    for name, metric in metrics.items():
        s = metric.corpus_score(hyps, [refs])
        result[name] = {'score': round(s.score, 2), 'signature': str(metric.get_signature())}
    return result


def main():
    config = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='mặc định: checkpoint cuối trong thư mục weights')
    parser.add_argument('--split', default='test', choices=['validation', 'test'])
    parser.add_argument('--layers', type=int, default=None, help='số tầng encoder/decoder của checkpoint (mặc định 3)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None, help='chỉ đánh giá N câu đầu (để chạy thử)')
    args = parser.parse_args()

    if args.layers:
        config['num_layers'] = args.layers
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = str(sorted(Path(config['model_folder']).glob('*.pt'))[-1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device} | checkpoint: {checkpoint} | split: {args.split}')

    splits = load_iwslt_splits(config)
    # NOTE: tokenizer được nạp từ file đã lưu lúc train (không train lại, tránh rò rỉ dữ liệu).
    tokenizer_src = get_or_build_tokenizer(config, splits['train'], config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, splits['train'], config['lang_tgt'])
    sos, eos, pad = (tokenizer_tgt.token_to_id(t) for t in ('[SOS]', '[EOS]', '[PAD]'))
    unk = tokenizer_tgt.token_to_id('[UNK]')

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    data = splits[args.split]
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    seq_len = config['seq_len']
    lang_src, lang_tgt = config['lang_src'], config['lang_tgt']

    src_texts = [html.unescape(ex['translation'][lang_src]) for ex in data]
    refs = [html.unescape(ex['translation'][lang_tgt]) for ex in data]
    src_ids = [tokenizer_src.encode(t).ids for t in src_texts]
    tgt_lens = [len(tokenizer_tgt.encode(t).ids) for t in refs]
    # Câu "vừa" theo cùng tiêu chí filter_by_length lúc train (xem train.py).
    fits = [1 <= len(s) <= seq_len - 2 and 1 <= t <= seq_len - 1 for s, t in zip(src_ids, tgt_lens)]

    hyps = []
    start = time.time()
    for i in range(0, len(data), args.batch_size):
        batch_ids = src_ids[i:i + args.batch_size]
        src = torch.tensor([encode_source(s, seq_len, sos, eos, pad) for s in batch_ids], device=device)
        src_mask = (src != pad).unsqueeze(1).unsqueeze(1).int()
        out = greedy_decode_batch(model, src, src_mask, sos, eos, seq_len, device)
        for row in out.tolist():
            row = row[1:]  # bỏ [SOS]
            if eos in row:
                row = row[:row.index(eos)]
            hyps.append(tokenizer_tgt.decode(row))
    elapsed = time.time() - start

    n_unk_ref = sum(tokenizer_tgt.encode(t).ids.count(unk) for t in refs)
    n_tok_ref = sum(tgt_lens)
    subset = [i for i, f in enumerate(fits) if f]
    results = {
        'checkpoint': checkpoint,
        'split': args.split,
        'greedy_decoding': True,
        'num_parameters': n_params,
        'num_sentences': len(data),
        'num_sentences_fit_seq_len': len(subset),
        'seq_len': seq_len,
        'decode_seconds': round(elapsed, 1),
        'ref_unk_rate': round(n_unk_ref / max(n_tok_ref, 1), 4),
        'all_sentences': score(hyps, refs),
        'fit_seq_len_only': score([hyps[i] for i in subset], [refs[i] for i in subset]),
    }

    out_dir = Path('eval_results')
    out_dir.mkdir(exist_ok=True)
    stem = f'{Path(checkpoint).stem}_{args.split}'
    if Path(checkpoint).parent.name != config['model_folder']:
        stem = f'{Path(checkpoint).parent.name}_{stem}'  # checkpoint của ablation nằm trong thư mục riêng
    (out_dir / f'{stem}.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / f'{stem}.hyp.txt').write_text('\n'.join(hyps) + '\n', encoding='utf-8')

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print('\nVí dụ:')
    for i in range(min(3, len(hyps))):
        print(f'SRC: {src_texts[i]}\nREF: {refs[i]}\nHYP: {hyps[i]}\n')


if __name__ == '__main__':
    main()
