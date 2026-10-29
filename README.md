# Transformer dịch máy Anh–Việt bằng PyTorch

Cài đặt từ đầu mô hình Transformer encoder–decoder cho bài toán dịch máy Anh→Việt, huấn luyện trên bộ dữ liệu IWSLT'15, kèm đánh giá (SacreBLEU, chrF, TER), thí nghiệm ablation, kiểm thử đơn vị và ứng dụng demo bằng Gradio.

## Kết quả

Mô hình có 20 222 872 tham số, huấn luyện 5 epoch (khoảng 20 phút trên GPU RTX 3050 Laptop 6 GB), giải mã greedy.

| Chỉ số (tập test, 1 269 câu) | Giá trị |
|---|---|
| SacreBLEU | 20,52 |
| chrF | 39,16 |
| TER (thấp hơn là tốt hơn) | 61,12 |

Ablation (cùng dữ liệu, cùng seed 42, 5 epoch):

| Cấu hình | SacreBLEU | chrF | TER | Số tham số | Thời gian huấn luyện |
|---|---|---|---|---|---|
| Nền: 3 lớp, label smoothing 0,1 | 20,54 | 39,34 | 60,67 | 20,2 triệu | 19,4 phút |
| 6 lớp, label smoothing 0,1 | 20,37 | 38,74 | 60,14 | 25,7 triệu | 28,2 phút |
| 3 lớp, không label smoothing | 20,59 | 39,49 | 60,52 | 20,2 triệu | 17,6 phút |

Độ trễ suy luận (một câu 1–50 token, mô hình đã nạp sẵn, chỉ gồm bước mã hóa và giải mã): trung bình 0,154 s và lớn nhất 0,436 s trên GPU; lớn nhất 1,093 s trên CPU (60 câu).

Lưu ý khi đọc kết quả:

- Hai tập `validation` và `test` của bản parquet trên Hugging Face giống hệt nhau, nên không có tập kiểm định độc lập; mọi điểm số báo cáo trên tập `test` và dùng checkpoint của epoch cuối.
- Dữ liệu đã được tách từ sẵn nên SacreBLEU chạy với `force=True`; điểm này không so sánh trực tiếp được với BLEU của các công bố khác.
- Mô hình chưa hội tụ sau 5 epoch, và các chênh lệch giữa ba cấu hình ablation (dưới 0,2 BLEU, một lần chạy mỗi cấu hình) chưa đủ để kết luận.

## Cấu trúc thư mục

```
config.py          Siêu tham số và đường dẫn
dataset.py         BilingualDataset, mặt nạ padding và mặt nạ nhân quả
model.py           Kiến trúc Transformer (Pre-LN, mã hóa vị trí sin–cos)
train.py           Huấn luyện (bfloat16, Adam, label smoothing)
evaluate.py        Giải mã greedy theo batch và chấm SacreBLEU, chrF, TER
run_ablation.py    Chạy ablation số lớp và label smoothing
demo.py            Giao diện Gradio và đo độ trễ
test_model.py      35 kiểm thử đơn vị
tokenizer_en.json  Bộ tách token mức từ tiếng Anh (30 000 token)
tokenizer_vi.json  Bộ tách token mức từ tiếng Việt (13 686 token)
eval_results/      Kết quả đánh giá (.json) và bản dịch (.hyp.txt)
ablation_results/  Nhật ký huấn luyện của các cấu hình ablation
docs/              Ảnh minh họa
```

## Cài đặt

Yêu cầu Python 3.10 trở lên. Mã nguồn được phát triển và kiểm thử trên Windows 11 với Python 3.10.11, PyTorch 2.14.1 (CUDA), SacreBLEU 2.6.0 và Gradio 6.31.0.

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Nếu có GPU NVIDIA, hãy cài bản PyTorch có CUDA theo hướng dẫn tại [pytorch.org](https://pytorch.org/get-started/locally/) thay cho bản mặc định.

## Dữ liệu

Dữ liệu IWSLT'15 En–Vi (133 318 cặp câu huấn luyện) được `train.py` tự đọc từ bản parquet `IWSLT/mt_eng_vietnamese` trên Hugging Face, nên cần có kết nối mạng ở lần chạy đầu. Hai tệp `tokenizer_en.json` và `tokenizer_vi.json` đã có sẵn trong repo; nếu xóa, chúng sẽ được huấn luyện lại từ tập train.

Tiền xử lý: giải mã thực thể HTML, giữ nguyên cách tách từ có sẵn, lọc cặp câu quá dài (giữ 1–62 token nguồn và 1–63 token đích, còn 127 826 cặp), đệm về độ dài cố định 64.

## Sử dụng

Huấn luyện (checkpoint lưu vào `weights/tmodel_00.pt` đến `tmodel_04.pt`):

```bash
python train.py
```

Đánh giá một checkpoint trên tập test:

```bash
python evaluate.py --checkpoint weights/tmodel_04.pt --split test
```

Kết quả được in ra màn hình và lưu vào `eval_results/`. Thêm `--limit 100` để chạy thử trên 100 câu đầu.

Chạy ablation (3 cấu hình, mỗi cấu hình 5 epoch; cấu hình đã có checkpoint cuối sẽ được bỏ qua):

```bash
python run_ablation.py
```

Chạy giao diện dịch tại http://127.0.0.1:7860:

```bash
python demo.py
```

Vì bộ tách token mức từ được huấn luyện trên dữ liệu đã tách từ, câu nhập nên có dấu câu cách rời, ví dụ `Hello , world .`.

Đo độ trễ:

```bash
python demo.py --benchmark --device cuda
python demo.py --benchmark --device cpu --n 60
```

Chạy kiểm thử đơn vị:

```bash
python test_model.py
```

Bài kiểm thử đầu–cuối (dịch bằng checkpoint thật) tự bỏ qua nếu chưa có `weights/*.pt`.

## Checkpoint

Các file checkpoint (khoảng 240 MB mỗi file, gồm cả trạng thái optimizer) không nằm trong repo vì vượt giới hạn dung lượng của GitHub. Để có checkpoint, chạy `python train.py` hoặc tải lên GitHub Releases / Hugging Face và thêm đường dẫn tải vào mục này.

## Cấu hình chính

| Tham số | Giá trị |
|---|---|
| Số lớp encoder và decoder | 3 |
| d_model / d_ff / số đầu attention | 256 / 1 024 / 4 |
| Dropout | 0,1 |
| Label smoothing | 0,1 |
| Optimizer | Adam, learning rate cố định 1e-4, ε = 1e-9 |
| Batch size / độ dài chuỗi | 64 / 64 |
| Độ chính xác | bfloat16 (autocast) |
| Tách token | WordLevel, hai từ vựng riêng, `min_frequency=2` |

Chỉnh các giá trị này trong `config.py`.

## Hạn chế

- Từ vựng mức từ nên từ hiếm và tên riêng bị thay bằng `[UNK]`, là nguồn lỗi dịch chính.
- Chỉ giải mã greedy, chưa có beam search hay KV cache.
- Huấn luyện ngắn (5 epoch), learning rate cố định, không có warm-up.
- Chỉ dịch từng câu, một chiều Anh→Việt, câu nguồn dài hơn 62 token bị cắt.

## Giấy phép

Mã nguồn được phát hành theo giấy phép [MIT](LICENSE). Giấy phép này chỉ áp dụng cho mã nguồn trong repo; dữ liệu IWSLT'15 và các thư viện bên thứ ba (PyTorch, Hugging Face Datasets, SacreBLEU, Gradio…) có giấy phép riêng.
