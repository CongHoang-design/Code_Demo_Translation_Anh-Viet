import html
from typing import Any

import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    # Biến một cặp câu song ngữ thành tensor có độ dài cố định seq_len, sẵn sàng đưa vào Transformer.
    # Mỗi item trả về: input cho encoder/decoder, mask tương ứng và label để tính loss.

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds  # dataset thô, mỗi phần tử có dạng {'translation': {lang: text}}
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len  # độ dài cố định sau padding, phải khớp seq_len khi build model

        # Id của token đặc biệt, tạo sẵn một lần để không tra cứu lại ở mỗi __getitem__.
        # WARNING: cả ba id đều lấy từ tokenizer_tgt nhưng cũng được dùng cho phía nguồn.
        # Chỉ đúng nếu hai tokenizer gán cùng id cho [SOS]/[EOS]/[PAD].
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        # Dữ liệu thô có thể chứa HTML entity (&amp;, &quot;...) nên cần giải mã trước khi tokenize.
        src_text = html.unescape(src_target_pair['translation'][self.src_lang])
        tgt_text = html.unescape(src_target_pair['translation'][self.tgt_lang])

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Số PAD cần thêm để đủ seq_len.
        # Encoder trừ 2 chỗ cho [SOS] và [EOS]; decoder chỉ trừ 1 vì decoder_input chỉ có
        # [SOS] còn label chỉ có [EOS] (hai chuỗi này lệch nhau đúng một vị trí).
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Câu quá dài bị từ chối thay vì cắt ngầm. Số âm nghĩa là không đủ chỗ cho token đặc biệt.
        # NOTE: ngoại lệ này làm hỏng cả quá trình train nếu có một mẫu dài; nên lọc dữ liệu trước.
        if enc_num_padding_tokens < 0 or  dec_num_padding_tokens < 0:
            raise ValueError('Doan van qua dai')

        # Encoder thấy toàn bộ câu nguồn: [SOS] tokens [EOS] [PAD]...
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64),
            ]
        )

        # Decoder input: [SOS] tokens [PAD]... (không có [EOS]).
        # Teacher forcing: tại vị trí t, decoder nhận token đích thứ t và phải dự đoán token t+1.
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64),
            ]
        )

        # Label là decoder_input dịch sang trái một vị trí: tokens [EOS] [PAD]...
        # (không có [SOS]); đây là đáp án mà decoder cần dự đoán ở mỗi bước.
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64),
            ]
        )

        # Kiểm tra bất biến: mọi chuỗi phải đúng seq_len để gom thành batch được.
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # Mask quy ước 1 = được nhìn, 0 = bị cấm. Thêm 2 chiều để broadcast với
            # attention score (B, h, S, S): (seq_len) -> (1, 1, seq_len).
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # Kết hợp mask padding (1, 1, S) với mask nhân quả (1, S, S): vị trí chỉ được nhìn
            # token không phải PAD và nằm ở hiện tại hoặc quá khứ.
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            # Giữ văn bản gốc để in ra khi đánh giá/debug, không đưa vào model.
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size: int):
    # Mask tam giác dưới (size x size): hàng i chỉ được nhìn các cột j <= i.
    # triu(diagonal=1) lấy phần tam giác trên (tương lai) bằng 1; so sánh == 0 đảo lại
    # để vùng được phép nhìn là True.
    # Ra: (1, size, size), kiểu bool.
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
