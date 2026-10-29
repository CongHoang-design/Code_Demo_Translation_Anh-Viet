# Cài đặt kiến trúc Transformer encoder-decoder (Vaswani et al., 2017) cho bài toán dịch máy.
# Quy ước shape: B = batch, S = độ dài câu nguồn, T = độ dài câu đích, d = d_model.
import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    # Đổi id token (B, S) thành vector (B, S, d_model).

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Nhân sqrt(d_model) theo paper: weight embedding khởi tạo nhỏ nên cần phóng to
        # để không bị át bởi positional encoding khi hai tensor được cộng.
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # Attention không biết thứ tự token, nên cộng thêm vector vị trí cố định (không học):
    #   pe[pos, 2i]   = sin(pos / 10000^(2i/d))
    #   pe[pos, 2i+1] = cos(pos / 10000^(2i/d))

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len  # độ dài tối đa hỗ trợ; câu dài hơn sẽ không khớp shape
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)

        # 1 / 10000^(2i/d) viết dưới dạng exp(log(...)) để ổn định số học.
        # Tần số giảm dần từ 1 xuống ~1/10000 theo chỉ số chiều.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Chiều chẵn dùng sin, chiều lẻ dùng cos.
        # WARNING: d_model lẻ sẽ lỗi shape ở dòng cos vì số cột lẻ ít hơn số cột chẵn.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d) để broadcast theo batch

        # Buffer: không phải tham số học, nhưng được lưu trong state_dict và tự chuyển
        # sang GPU cùng model (tránh lỗi "tensors on different devices").
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Cắt theo độ dài thực tế của batch để câu ngắn hơn seq_len vẫn chạy được.
        # NOTE: requires_grad_(False) thực ra thừa vì buffer vốn không có gradient.
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    # Chuẩn hóa từng token theo chiều đặc trưng cuối để ổn định huấn luyện.

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps  # chống chia cho 0 khi std ~ 0
        self.alpha = nn.Parameter(torch.ones(1))   # scale học được (gamma)
        self.bias = nn.Parameter(torch.zeros(1))   # shift học được (beta)
        # NOTE: alpha/bias là scalar, khác bản chuẩn (nn.LayerNorm) dùng một giá trị cho mỗi chiều.

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)  # NOTE: std mặc định là unbiased (chia n-1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    # FFN(x) = max(0, xW1 + b1)W2 + b2, áp dụng độc lập cho từng vị trí.
    # Mở rộng d_model -> d_ff rồi thu về d_model: tăng khả năng biến đổi phi tuyến
    # sau bước attention (vốn chủ yếu là tổ hợp tuyến tính).

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (B, S, d) -> (B, S, d_ff) -> ReLU -> Dropout -> (B, S, d)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):
    # Dùng chung cho self-attention và cross-attention; chỉ khác ở nguồn của q, k, v.
    # Vào: q, k, v dạng (B, *, d). Ra: (B, S_q, d).

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model khong chia het cho h"

        self.d_k = d_model // h  # số chiều mỗi head, ví dụ 512 / 4 = 128
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # trộn thông tin giữa các head sau khi ghép
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V
        d_k = query.shape[-1]

        # Chia sqrt(d_k) để tích vô hướng không quá lớn, tránh softmax bão hòa
        # (gradient gần 0) khi d_k lớn.
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, S_q, S_k)

        # Quy ước mask: 0 = vị trí bị cấm, 1 = được nhìn.
        # Dùng -1e9 thay vì -inf để tránh NaN khi cả một hàng bị mask.
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Trả thêm ma trận attention để phục vụ visualize/debug.
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Tách d_model thành h head: (B, S, d) -> (B, S, h, d_k) -> (B, h, S, d_k),
        # đưa h lên trước S để mỗi head tính attention độc lập.
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # NOTE: lưu attention_scores vào self làm model mang trạng thái ẩn sau mỗi lần forward.
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Ghép các head: (B, h, S, d_k) -> (B, S, h, d_k) -> (B, S, d).
        # contiguous() bắt buộc vì view() cần bộ nhớ liên tục sau transpose.
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    # Bọc một sublayer bằng norm + dropout + skip connection.

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # Pre-norm: x + Dropout(Sublayer(Norm(x))). Khác paper gốc (post-norm); pre-norm
        # huấn luyện ổn định hơn nhưng vì thế Encoder/Decoder cần thêm một norm cuối.
        # sublayer là callable nên một lớp này dùng chung cho attention và FFN.
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    # Một tầng encoder: self-attention rồi FFN, mỗi bước có residual riêng.

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # q = k = v = x; src_mask che các vị trí padding của câu nguồn.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    # Xếp chồng N EncoderBlock; norm cuối bù cho việc dùng pre-norm trong từng block.

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    # Một tầng decoder: masked self-attention, cross-attention, rồi FFN.

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # tgt_mask gồm mask padding và mask nhân quả (tam giác dưới) để token ở vị trí t
        # không nhìn thấy token t+1.. trong lúc huấn luyện (teacher forcing).
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # Cross-attention: query từ decoder, key/value từ encoder, nên mỗi từ đích
        # có thể "nhìn" vào các từ nguồn liên quan. Mask ở đây là src_mask (padding nguồn).
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    # Xếp chồng N DecoderBlock; mọi block dùng chung một encoder_output.

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class LinearLayer(nn.Module):
    # Chiếu (B, T, d_model) sang (B, T, vocab_size): một điểm số cho mỗi từ trong từ vựng.

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linea = nn.Linear(d_model, vocab_size)  # NOTE: tên "linea" là typo nhưng nằm trong state_dict, đổi tên sẽ hỏng checkpoint cũ

    def forward(self, x):
        # NOTE: trả về log-probability, hợp với NLLLoss. Dùng CrossEntropyLoss vẫn cho cùng
        # kết quả (log_softmax là phép lũy đẳng) nhưng tính thừa một lần log_softmax.
        return torch.log_softmax(self.linea(x), dim=-1)


class Transformer(nn.Module):
    # Chỉ là bộ chứa các thành phần; không có forward().
    # Tách encode/decode/linear để lúc suy luận chỉ encode câu nguồn một lần,
    # sau đó decode lặp lại từng token.
    # Luồng gọi: encode(src, src_mask) -> decode(enc_out, src_mask, tgt, tgt_mask) -> linear(out).

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, linear_layer: LinearLayer) -> None:
       super().__init__()
       self.encoder = encoder
       self.decoder = decoder
       self.src_embed = src_embed
       self.tgt_embed = tgt_embed
       self.src_pos = src_pos
       self.tgt_pos = tgt_pos
       self.linear_layer = linear_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def linear(self, x):
        return self.linear_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, n: int = 3, h: int = 4, dropout: float = 0.1, d_ff = 1024) -> Transformer:
    # Factory: lắp ráp toàn bộ mô hình từ hyperparameter.
    # NOTE: mặc định n=3, h=4, d_ff=1024 nhỏ hơn paper (6, 8, 2048); chủ ý giảm cho model nhẹ.

    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Mỗi block tự tạo attention/FFN riêng: các tầng không chia sẻ trọng số.
    encoder_blocks = []
    for _ in range(n):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder có thêm một attention cho cross-attention với encoder.
    decoder_blocks = []
    for _ in range(n):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    linear_layer = LinearLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, linear_layer)

    # Xavier uniform giữ phương sai activation/gradient ổn định giữa các tầng.
    # Chỉ áp dụng cho tensor >= 2 chiều (ma trận trọng số); bias 1 chiều giữ mặc định.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
