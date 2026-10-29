# Chạy lần lượt các thí nghiệm ablation (số lớp, label smoothing), mỗi cấu hình huấn luyện 5 epoch rồi đánh giá trên tập test.
# Chạy: python run_ablation.py            (cấu hình đã có checkpoint cuối sẽ được bỏ qua, nên chạy lại được sau khi bị ngắt)
# Kết quả: weights_ablation/<tên>/, runs/<tên>/ (TensorBoard), eval_results/*.json, ablation_results/<tên>_train.json
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from config import get_config
from train import train_model

SEED = 42

# Mỗi cấu hình chỉ khác cấu hình nền (n = 3, label smoothing = 0,1) đúng một yếu tố.
RUNS = {
    'abl_base': {'num_layers': 3, 'label_smoothing': 0.1},
    'abl_layers6': {'num_layers': 6, 'label_smoothing': 0.1},
    'abl_ls0': {'num_layers': 3, 'label_smoothing': 0.0},
}


def main():
    out_dir = Path('ablation_results')
    out_dir.mkdir(exist_ok=True)
    for name, overrides in RUNS.items():
        config = get_config()
        config.update(overrides)
        config['model_folder'] = f'weights_ablation/{name}'
        config['experiment_name'] = f'runs/{name}'
        last = Path(config['model_folder']) / f"{config['model_basename']}{config['num_epochs'] - 1:02d}.pt"

        if last.exists():
            print(f'[{name}] đã có {last}, bỏ qua huấn luyện')
        else:
            print(f'[{name}] bắt đầu huấn luyện: {overrides}')
            torch.manual_seed(SEED)
            start = time.time()
            train_model(config)
            seconds = time.time() - start
            (out_dir / f'{name}_train.json').write_text(json.dumps({
                'name': name, **overrides, 'seed': SEED, 'epochs': config['num_epochs'],
                'wall_seconds_including_data_prep': round(seconds),
            }, indent=2), encoding='utf-8')
            print(f'[{name}] huấn luyện xong sau {seconds / 60:.1f} phút')

        subprocess.run([sys.executable, 'evaluate.py', '--checkpoint', str(last), '--split', 'test',
                        '--layers', str(overrides['num_layers'])], check=True)


if __name__ == '__main__':
    main()
