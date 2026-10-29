# Unit test cho model.py, dataset.py và các hàm hỗ trợ (mask, positional encoding, kích thước tensor, causal, overfit).
# Chạy: python test_model.py   (hoặc: python -m unittest test_model -v)
import math
import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import BilingualDataset, causal_mask
from model import (
    FeedForward,
    InputEmbedding,
    LayerNormalization,
    MultiHeadAttention,
    PositionalEncoding,
    build_transformer,
)

# Cấu hình nhỏ để test chạy nhanh trên CPU.
D_MODEL, H, N, D_FF = 32, 4, 2, 64
SRC_VOCAB, TGT_VOCAB, SEQ_LEN = 50, 60, 12
PAD, SOS, EOS = 0, 1, 2


def make_model(dropout=0.0):
    # dropout = 0 và model.eval() để kết quả xác định, so sánh được giữa hai lần chạy.
    torch.manual_seed(0)
    model = build_transformer(SRC_VOCAB, TGT_VOCAB, SEQ_LEN, SEQ_LEN, d_model=D_MODEL, n=N, h=H, dropout=dropout, d_ff=D_FF)
    return model.eval()


class FakeTokenizer:
    # Tokenizer giả: mỗi ký tự là một token (id = ord % 40 + 3, tránh trùng id đặc biệt 0/1/2).
    def token_to_id(self, token):
        return {'[PAD]': PAD, '[SOS]': SOS, '[EOS]': EOS}[token]

    def encode(self, text):
        class Result:
            ids = [ord(c) % 40 + 3 for c in text]
        return Result


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.pe = PositionalEncoding(D_MODEL, SEQ_LEN, dropout=0.0)

    def test_buffer_shape_and_not_trainable(self):
        self.assertEqual(self.pe.pe.shape, (1, SEQ_LEN, D_MODEL))
        self.assertEqual(len(list(self.pe.parameters())), 0)
        self.assertIn('pe', self.pe.state_dict())

    def test_position_zero(self):
        # pos = 0: sin(0) = 0 ở chiều chẵn, cos(0) = 1 ở chiều lẻ.
        row = self.pe.pe[0, 0]
        self.assertTrue(torch.allclose(row[0::2], torch.zeros(D_MODEL // 2)))
        self.assertTrue(torch.allclose(row[1::2], torch.ones(D_MODEL // 2)))

    def test_formula(self):
        pos, i = 5, 3
        angle = pos / (10000 ** (2 * i / D_MODEL))
        self.assertAlmostEqual(self.pe.pe[0, pos, 2 * i].item(), math.sin(angle), places=5)
        self.assertAlmostEqual(self.pe.pe[0, pos, 2 * i + 1].item(), math.cos(angle), places=5)

    def test_shorter_sequence(self):
        out = self.pe(torch.zeros(2, 5, D_MODEL))
        self.assertEqual(out.shape, (2, 5, D_MODEL))
        self.assertTrue(torch.allclose(out[0], self.pe.pe[0, :5]))

    def test_odd_d_model_fails(self):
        with self.assertRaises(RuntimeError):
            PositionalEncoding(31, SEQ_LEN, dropout=0.0)


class TestBasicLayers(unittest.TestCase):
    def test_embedding_scale_and_shape(self):
        emb = InputEmbedding(D_MODEL, SRC_VOCAB)
        x = torch.randint(0, SRC_VOCAB, (3, 7))
        out = emb(x)
        self.assertEqual(out.shape, (3, 7, D_MODEL))
        self.assertTrue(torch.allclose(out, emb.embedding(x) * math.sqrt(D_MODEL)))

    def test_layer_norm_statistics(self):
        x = torch.randn(4, 6, D_MODEL) * 5 + 3
        out = LayerNormalization()(x)
        self.assertTrue(torch.allclose(out.mean(-1), torch.zeros(4, 6), atol=1e-4))
        self.assertTrue(torch.allclose(out.std(-1), torch.ones(4, 6), atol=1e-3))

    def test_feed_forward_shape(self):
        out = FeedForward(D_MODEL, D_FF, 0.0)(torch.randn(2, 5, D_MODEL))
        self.assertEqual(out.shape, (2, 5, D_MODEL))


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mha = MultiHeadAttention(D_MODEL, H, 0.0).eval()

    def test_d_model_must_divide_h(self):
        with self.assertRaises(AssertionError):
            MultiHeadAttention(30, 4, 0.0)

    def test_output_shape_and_weights_sum_to_one(self):
        x = torch.randn(2, 6, D_MODEL)
        out = self.mha(x, x, x, None)
        self.assertEqual(out.shape, (2, 6, D_MODEL))
        self.assertEqual(self.mha.attention_scores.shape, (2, H, 6, 6))
        sums = self.mha.attention_scores.sum(-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_cross_attention_shape(self):
        q = torch.randn(2, 4, D_MODEL)
        kv = torch.randn(2, 9, D_MODEL)
        out = self.mha(q, kv, kv, None)
        self.assertEqual(out.shape, (2, 4, D_MODEL))
        self.assertEqual(self.mha.attention_scores.shape, (2, H, 4, 9))

    def test_padding_mask_zeroes_weights(self):
        x = torch.randn(1, 5, D_MODEL)
        mask = torch.tensor([1, 1, 1, 0, 0]).view(1, 1, 1, 5)
        self.mha(x, x, x, mask)
        w = self.mha.attention_scores
        self.assertTrue(torch.all(w[..., 3:] < 1e-6))
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(1, H, 5), atol=1e-5))

    def test_causal_mask_zeroes_future(self):
        x = torch.randn(1, 5, D_MODEL)
        self.mha(x, x, x, causal_mask(5).unsqueeze(0))
        w = self.mha.attention_scores
        future = torch.triu(torch.ones(5, 5), diagonal=1).bool()
        self.assertTrue(torch.all(w[0, :, future] < 1e-6))

    def test_hand_computed_attention(self):
        # score = [2, 3, 1] / sqrt(d_k = 1), mask [1, 1, 0] -> softmax([2, 3]) = [0.2689, 0.7311, 0].
        q = torch.tensor([[[[1.0]]]])
        k = torch.tensor([[[[2.0], [3.0], [1.0]]]])
        v = torch.tensor([[[[10.0], [20.0], [30.0]]]])
        mask = torch.tensor([1, 1, 0]).view(1, 1, 1, 3)
        out, w = MultiHeadAttention.attention(q, k, v, mask, None)
        self.assertTrue(torch.allclose(w.flatten(), torch.tensor([0.2689, 0.7311, 0.0]), atol=1e-4))
        self.assertAlmostEqual(out.item(), 0.2689 * 10 + 0.7311 * 20, places=2)

    def test_report_example_three_tokens(self):
        # Ví dụ tính tay trong báo cáo (Bảng 2.6–2.7): Q = K, d_k = 2, có và không có causal mask.
        qk = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).view(1, 1, 3, 2)
        v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).view(1, 1, 3, 2)
        out, w = MultiHeadAttention.attention(qk, qk, v, None, None)
        self.assertTrue(torch.allclose(w[0, 0, 0], torch.tensor([0.401, 0.198, 0.401]), atol=1e-3))
        self.assertTrue(torch.allclose(out[0, 0, 0], torch.tensor([3.0, 4.0]), atol=1e-2))
        out_c, w_c = MultiHeadAttention.attention(qk, qk, v, causal_mask(3).unsqueeze(0), None)
        self.assertTrue(torch.allclose(w_c[0, 0, 1], torch.tensor([0.330, 0.670, 0.0]), atol=1e-3))
        self.assertTrue(torch.allclose(out_c[0, 0, 0], torch.tensor([1.0, 2.0]), atol=1e-4))


class TestCausalMask(unittest.TestCase):
    def test_lower_triangular(self):
        m = causal_mask(4)
        self.assertEqual(m.shape, (1, 4, 4))
        self.assertEqual(m.dtype, torch.bool)
        self.assertTrue(torch.equal(m[0], torch.tril(torch.ones(4, 4)).bool()))


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.src = torch.tensor([[SOS, 5, 6, 7, EOS, PAD, PAD, PAD], [SOS, 8, 9, 10, 11, 12, EOS, PAD]])
        self.src_mask = (self.src != PAD).view(2, 1, 1, -1).int()
        self.tgt = torch.tensor([[SOS, 5, 6, 7, PAD, PAD], [SOS, 8, 9, 10, 11, PAD]])
        self.tgt_mask = (self.tgt != PAD).view(2, 1, 1, -1).int() & causal_mask(6)

    def test_forward_shapes(self):
        enc = self.model.encode(self.src, self.src_mask)
        self.assertEqual(enc.shape, (2, 8, D_MODEL))
        dec = self.model.decode(enc, self.src_mask, self.tgt, self.tgt_mask)
        self.assertEqual(dec.shape, (2, 6, D_MODEL))
        self.assertEqual(self.model.linear(dec).shape, (2, 6, TGT_VOCAB))

    def test_output_is_log_probability(self):
        enc = self.model.encode(self.src, self.src_mask)
        out = self.model.linear(self.model.decode(enc, self.src_mask, self.tgt, self.tgt_mask))
        self.assertTrue(torch.all(out <= 0))
        sums = out.exp().sum(-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_decoder_is_causal(self):
        # Đổi token ở tương lai không được làm thay đổi đầu ra ở các vị trí trước đó.
        enc = self.model.encode(self.src, self.src_mask)
        base = self.model.decode(enc, self.src_mask, self.tgt, self.tgt_mask)
        changed = self.tgt.clone()
        changed[:, 3:] = 20
        out = self.model.decode(enc, self.src_mask, changed, self.tgt_mask)
        self.assertTrue(torch.allclose(base[:, :3], out[:, :3], atol=1e-5))
        self.assertFalse(torch.allclose(base[:, 3:], out[:, 3:], atol=1e-5))

    def test_encoder_ignores_padding(self):
        # Nội dung ở vị trí PAD (đã mask) không được ảnh hưởng tới các vị trí thật.
        base = self.model.encode(self.src, self.src_mask)
        changed = self.src.clone()
        changed[0, 5:] = 33
        out = self.model.encode(changed, self.src_mask)
        self.assertTrue(torch.allclose(base[0, :5], out[0, :5], atol=1e-5))

    def test_deterministic_in_eval_mode(self):
        a = self.model.encode(self.src, self.src_mask)
        b = self.model.encode(self.src, self.src_mask)
        self.assertTrue(torch.equal(a, b))

    def test_all_parameters_receive_gradient(self):
        model = make_model().train()
        enc = model.encode(self.src, self.src_mask)
        model.linear(model.decode(enc, self.src_mask, self.tgt, self.tgt_mask)).sum().backward()
        self.assertEqual([n for n, p in model.named_parameters() if p.grad is None], [])

    def test_state_dict_round_trip(self):
        other = build_transformer(SRC_VOCAB, TGT_VOCAB, SEQ_LEN, SEQ_LEN, d_model=D_MODEL, n=N, h=H, dropout=0.0, d_ff=D_FF).eval()
        other.load_state_dict(self.model.state_dict())
        self.assertTrue(torch.allclose(self.model.encode(self.src, self.src_mask), other.encode(self.src, self.src_mask)))

    def test_overfit_tiny_batch(self):
        # Mô hình đúng phải học thuộc 4 câu nhỏ: loss giảm mạnh sau vài chục bước.
        torch.manual_seed(0)
        model = build_transformer(SRC_VOCAB, TGT_VOCAB, SEQ_LEN, SEQ_LEN, d_model=D_MODEL, n=N, h=H, dropout=0.0, d_ff=D_FF)
        model.train()
        src = torch.randint(3, SRC_VOCAB, (4, 6))
        tgt_in = torch.cat([torch.full((4, 1), SOS), torch.randint(3, TGT_VOCAB, (4, 5))], dim=1)
        label = torch.cat([tgt_in[:, 1:], torch.full((4, 1), EOS)], dim=1)
        src_mask = torch.ones(4, 1, 1, 6, dtype=torch.int)
        tgt_mask = causal_mask(6).unsqueeze(0).int()
        loss_fn = nn.NLLLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(150):
            enc = model.encode(src, src_mask)
            out = model.linear(model.decode(enc, src_mask, tgt_in, tgt_mask))
            loss = loss_fn(out.view(-1, TGT_VOCAB), label.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 0.2 * losses[0])


class TestBilingualDataset(unittest.TestCase):
    def setUp(self):
        raw = [{'translation': {'en': 'abc &amp; d', 'vi': 'xyz'}}]
        self.ds = BilingualDataset(raw, FakeTokenizer(), FakeTokenizer(), 'en', 'vi', SEQ_LEN)
        self.item = self.ds[0]

    def test_lengths_and_special_tokens(self):
        enc, dec, label = self.item['encoder_input'], self.item['decoder_input'], self.item['label']
        for t in (enc, dec, label):
            self.assertEqual(t.shape, (SEQ_LEN,))
        self.assertEqual((enc[0].item(), enc[8].item()), (SOS, EOS))  # 'abc & d' = 7 token, EOS ở vị trí 8
        self.assertEqual(dec[0].item(), SOS)
        self.assertNotIn(EOS, dec.tolist())
        self.assertEqual(label[3].item(), EOS)

    def test_label_is_decoder_input_shifted(self):
        dec, label = self.item['decoder_input'], self.item['label']
        self.assertTrue(torch.equal(dec[1:4], label[0:3]))

    def test_html_unescape(self):
        self.assertEqual(self.item['src_text'], 'abc & d')

    def test_masks(self):
        self.assertEqual(self.item['encoder_mask'].shape, (1, 1, SEQ_LEN))
        self.assertEqual(int(self.item['encoder_mask'].sum()), 9)  # SOS + 7 token + EOS
        dm = self.item['decoder_mask']
        self.assertEqual(dm.shape, (1, SEQ_LEN, SEQ_LEN))
        self.assertEqual(int(dm[0, 0].sum()), 1)   # vị trí 0 chỉ nhìn chính nó
        self.assertEqual(int(dm[0, 3].sum()), 4)   # vị trí 3 nhìn 0..3
        self.assertEqual(int(dm[0, 8].sum()), 4)   # vị trí PAD: vẫn chỉ nhìn các token thật

    def test_too_long_sentence_raises(self):
        raw = [{'translation': {'en': 'a' * SEQ_LEN, 'vi': 'x'}}]
        with self.assertRaises(ValueError):
            BilingualDataset(raw, FakeTokenizer(), FakeTokenizer(), 'en', 'vi', SEQ_LEN)[0]

    def test_batching_with_default_collate(self):
        batch = next(iter(torch.utils.data.DataLoader(self.ds, batch_size=1)))
        self.assertEqual(batch['encoder_input'].shape, (1, SEQ_LEN))
        self.assertEqual(batch['decoder_mask'].shape, (1, 1, SEQ_LEN, SEQ_LEN))
        self.assertEqual(batch['src_text'], ['abc & d'])


class TestInference(unittest.TestCase):
    # Bước suy luận cơ bản (M1): greedy decode và các hàm hỗ trợ dùng trong evaluate.py / demo.py.
    def setUp(self):
        self.model = make_model()
        self.src = torch.tensor([[SOS, 5, 6, 7, EOS, PAD, PAD, PAD]])
        self.mask = (self.src != PAD).view(1, 1, 1, -1).int()

    def test_encode_source_pads_and_truncates(self):
        from evaluate import encode_source
        self.assertEqual(encode_source([5, 6], 6, SOS, EOS, PAD), [SOS, 5, 6, EOS, PAD, PAD])
        self.assertEqual(encode_source([5, 6, 7, 8, 9], 6, SOS, EOS, PAD), [SOS, 5, 6, 7, 8, EOS])

    def test_batch_greedy_decode(self):
        from evaluate import greedy_decode_batch
        out = greedy_decode_batch(self.model, self.src, self.mask, SOS, EOS, SEQ_LEN, torch.device('cpu'))
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out[0, 0].item(), SOS)
        self.assertLessEqual(out.shape[1], SEQ_LEN)
        self.assertTrue(((out >= 0) & (out < TGT_VOCAB)).all())

    def test_batch_decode_matches_single_sentence_decode(self):
        # Hai cài đặt greedy (theo batch trong evaluate.py và từng câu trong train.py) phải cho cùng kết quả.
        try:
            from train import greedy_decode
        except ImportError as e:
            self.skipTest(f'không import được train.py: {e}')
        from evaluate import greedy_decode_batch
        cpu = torch.device('cpu')
        batch = greedy_decode_batch(self.model, self.src, self.mask, SOS, EOS, SEQ_LEN, cpu)[0].tolist()
        single = greedy_decode(self.model, self.src, self.mask, FakeTokenizer(), FakeTokenizer(), SEQ_LEN, cpu).tolist()
        n = min(len(single), len(batch))
        self.assertEqual(batch[:n], single[:n])

    def test_clean_text(self):
        from demo import clean_text
        self.assertEqual(clean_text('Tôi yêu bạn .'), 'Tôi yêu bạn.')
        self.assertEqual(clean_text('( xin chào ) , bạn !'), '(xin chào), bạn!')


class TestTranslatorWithRealCheckpoint(unittest.TestCase):
    CHECKPOINTS = sorted(Path('weights').glob('*.pt')) if Path('weights').exists() else []

    @unittest.skipUnless(CHECKPOINTS and Path('tokenizer_en.json').exists(), 'chưa có checkpoint hoặc tokenizer')
    def test_translate_end_to_end(self):
        from demo import Translator
        translator = Translator(str(self.CHECKPOINTS[-1]), device='cpu')
        text, seconds, n_tokens = translator.translate('I love you .')
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertGreater(seconds, 0)
        self.assertEqual(n_tokens, 4)
        with self.assertRaises(ValueError):
            translator.translate('   ')


if __name__ == '__main__':
    unittest.main(verbosity=2)
