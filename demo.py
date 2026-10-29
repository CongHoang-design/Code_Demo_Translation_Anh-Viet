# Demo dịch Anh -> Việt bằng Gradio và đo độ trễ suy luận (mục tiêu M4: < 3 giây cho câu <= 50 token trên GPU).
# Chạy giao diện: python demo.py [--checkpoint weights/tmodel_04.pt] [--port 7860]
# Đo độ trễ:      python demo.py --benchmark [--device cuda|cpu] [--n 200]
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import get_config
from evaluate import encode_source, greedy_decode_batch
from model import build_transformer

MAX_SOURCE_TOKENS = 50  # giới hạn của chỉ tiêu độ trễ; mô hình tự cắt nếu câu dài hơn seq_len - 2

EXAMPLES = [
    'I love deep learning .',
    'She gave me a book yesterday .',
    'In school , we spent a lot of time studying history .',
    'The weather is very nice today .',
]


def clean_text(text):
    # Tokenizer mức từ xuất ra dấu câu cách rời ("bạn ."). Chỉ làm đẹp khi hiển thị, không dùng khi chấm điểm.
    text = re.sub(r'\s+([,.;:!?%)\]])', r'\1', text)
    text = re.sub(r'([(\[])\s+', r'\1', text)
    return text.strip()


class Translator:
    # Nạp một lần (tokenizer, mô hình, checkpoint) rồi dịch từng câu bằng greedy decoding.

    def __init__(self, checkpoint=None, device=None):
        self.config = get_config()
        if checkpoint is None:
            checkpoint = str(sorted(Path(self.config['model_folder']).glob('*.pt'))[-1])
        self.checkpoint = checkpoint
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tok_src = Tokenizer.from_file(self.config['tokenizer_file'].format(self.config['lang_src']))
        self.tok_tgt = Tokenizer.from_file(self.config['tokenizer_file'].format(self.config['lang_tgt']))
        self.sos, self.eos, self.pad = (self.tok_tgt.token_to_id(t) for t in ('[SOS]', '[EOS]', '[PAD]'))
        self.seq_len = self.config['seq_len']

        self.model = build_transformer(
            self.tok_src.get_vocab_size(), self.tok_tgt.get_vocab_size(),
            self.seq_len, self.seq_len, self.config['d_model'],
        ).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

    def translate(self, text):
        # Trả về (bản dịch đã làm đẹp, thời gian suy luận tính bằng giây, số token nguồn).
        ids = self.tok_src.encode(text).ids
        if not ids:
            raise ValueError('Câu nguồn rỗng')
        src = torch.tensor([encode_source(ids, self.seq_len, self.sos, self.eos, self.pad)], device=self.device)
        src_mask = (src != self.pad).unsqueeze(1).unsqueeze(1).int()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = greedy_decode_batch(self.model, src, src_mask, self.sos, self.eos, self.seq_len, self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        row = out[0].tolist()[1:]  # bỏ [SOS]
        if self.eos in row:
            row = row[:row.index(self.eos)]
        return clean_text(self.tok_tgt.decode(row)), elapsed, len(ids)


def launch_ui(translator, port):
    import gradio as gr

    def run(text):
        text = (text or '').strip()
        if not text:
            return '', 'Hãy nhập một câu tiếng Anh.'
        translation, seconds, n_tokens = translator.translate(text)
        note = f'{seconds * 1000:.0f} ms | {n_tokens} token nguồn | greedy | {translator.device}'
        if n_tokens > translator.seq_len - 2:
            note += f' | câu bị cắt còn {translator.seq_len - 2} token'
        elif n_tokens > MAX_SOURCE_TOKENS:
            note += f' | vượt chỉ tiêu {MAX_SOURCE_TOKENS} token'
        return translation, note

    with gr.Blocks(title='Dịch máy Anh–Việt (Transformer)') as demo:
        gr.Markdown('## Dịch máy Anh → Việt (Transformer tự cài đặt bằng PyTorch)\n'
                    'Mô hình huấn luyện từ đầu trên IWSLT\'15, tokenizer mức từ: nên nhập câu có dấu câu '
                    'cách rời như dữ liệu huấn luyện (ví dụ `Hello , world .`).')
        inp = gr.Textbox(label='Tiếng Anh', lines=3)
        btn = gr.Button('Dịch', variant='primary')
        out = gr.Textbox(label='Tiếng Việt', lines=3)
        info = gr.Textbox(label='Thông tin suy luận')
        gr.Examples(EXAMPLES, inputs=inp)
        btn.click(run, inp, [out, info])
        inp.submit(run, inp, [out, info])
    demo.launch(server_port=port)


def benchmark(translator, n, out_dir='eval_results'):
    # Đo độ trễ từng câu (batch = 1) trên các câu của tập test có 1..50 token nguồn.
    from train import load_iwslt_splits
    import html

    data = load_iwslt_splits(translator.config)['test']
    sentences = []
    for ex in data:
        text = html.unescape(ex['translation'][translator.config['lang_src']])
        if 1 <= len(translator.tok_src.encode(text).ids) <= MAX_SOURCE_TOKENS:
            sentences.append(text)
    step = max(len(sentences) // n, 1)
    sentences = sentences[::step][:n]

    for s in sentences[:5]:  # khởi động (cuDNN, cấp phát bộ nhớ) không tính vào kết quả
        translator.translate(s)

    rows = []
    for s in sentences:
        _, seconds, n_tokens = translator.translate(s)
        rows.append((n_tokens, seconds))

    def summary(sel):
        t = sorted(x[1] for x in sel)
        if not t:
            return None
        return {'n': len(t), 'mean_s': round(statistics.mean(t), 3), 'median_s': round(statistics.median(t), 3),
                'p95_s': round(t[min(int(0.95 * len(t)), len(t) - 1)], 3), 'max_s': round(t[-1], 3)}

    result = {
        'device': str(translator.device),
        'gpu': torch.cuda.get_device_name(0) if translator.device.type == 'cuda' else None,
        'torch': torch.__version__,
        'checkpoint': translator.checkpoint,
        'decoding': 'greedy (không dùng KV cache), batch = 1, mô hình đã nạp sẵn',
        'num_sentences': len(rows),
        'all': summary(rows),
        'tokens_1_10': summary([r for r in rows if r[0] <= 10]),
        'tokens_11_30': summary([r for r in rows if 11 <= r[0] <= 30]),
        'tokens_31_50': summary([r for r in rows if 31 <= r[0] <= 50]),
        'share_under_3s': round(sum(r[1] < 3 for r in rows) / len(rows), 4),
    }
    Path(out_dir).mkdir(exist_ok=True)
    path = Path(out_dir) / f'latency_{translator.device.type}.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Đã lưu {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--n', type=int, default=200, help='số câu dùng khi đo độ trễ')
    args = parser.parse_args()

    translator = Translator(args.checkpoint, args.device)
    if args.benchmark:
        benchmark(translator, args.n)
    else:
        launch_ui(translator, args.port)
