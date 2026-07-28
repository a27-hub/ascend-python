import unittest
import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestDenseLightningIndexerSoftmaxLse(TestCase):
    """
    npu_dense_lightning_indexer_softmax_lse computes the softmax max / sum inputs
    for npu_dense_lightning_indexer_grad_kl_loss:
        res = AttentionMask(ReduceSum_N1(ReLU(Q_index @ K_index^T) * W))   # rightDownCausal mask
        softmax_max_index = max_over_S2(res)
        softmax_sum_index = sum_over_S2(exp(res - softmax_max_index))
    """

    def _deal_sparse_mask(self, res):
        # res: (B, S1, S2); rightDownCausal (sparse_mode=3): lower-triangular w.r.t. right vertex
        b, s1, s2 = res.shape
        assert s1 <= s2
        for s1_idx in range(s1):
            res[:, s1_idx, (s2 - s1 + s1_idx + 1):] = float("-inf")
        return res

    def _softmax_lse_ref(self, query_index, key_index, weight, layout):
        # Normalize TND (3D) -> BSND (4D) with B=1 for a single reference path.
        is_tnd = (layout == "TND")
        if is_tnd:
            query_index = query_index.unsqueeze(0)
            key_index = key_index.unsqueeze(0)
            weight = weight.unsqueeze(0)
        # query_index: (B, S1, N1, D), key_index: (B, S2, N2, D), weight: (B, S1, N1)
        _, _, n1, d = query_index.shape
        _, _, n2, _ = key_index.shape
        group = n1 // n2  # N2 is 1, so group == N1
        q = query_index.permute(0, 2, 1, 3).contiguous()                       # (B, N1, S1, D)
        k = key_index.permute(0, 2, 3, 1).contiguous().repeat(1, group, 1, 1)  # (B, N1, D, S2)
        s = torch.matmul(q.to(torch.float32), k.to(torch.float32))            # (B, N1, S1, S2)
        s = torch.relu(s)
        w = weight.permute(0, 2, 1).contiguous().unsqueeze(-1).to(torch.float32)  # (B, N1, S1, 1)
        s = s * w
        res = s.sum(dim=1, keepdim=True).squeeze(1)                            # (B, S1, S2) ReduceSum over N1
        res = self._deal_sparse_mask(res)
        softmax_max = res.max(dim=-1).values                                   # (B, S1)
        softmax_sum = torch.exp(res - softmax_max.unsqueeze(-1)).sum(dim=-1)   # (B, S1)
        if is_tnd:
            # TND output shape = (Nidx2, T1) = (1, S1)
            return softmax_max.squeeze(0).unsqueeze(0), softmax_sum.squeeze(0).unsqueeze(0)
        # BSND output shape = (B, Nidx2, S1) = (B, 1, S1)
        return softmax_max.unsqueeze(1), softmax_sum.unsqueeze(1)

    def _gen_inputs(self, n1, dtype, weight_dtype, layout):
        n2, d = 1, 128
        if layout == "TND":
            t1, t2 = 128, 256
            q = torch.randn(t1, n1, d, dtype=dtype)
            k = torch.randn(t2, n2, d, dtype=dtype)
            w = torch.randn(t1, n1, dtype=weight_dtype)
            return q, k, w, [t1], [t2]
        # BSND
        b, s1, s2 = 1, 128, 256
        q = torch.randn(b, s1, n1, d, dtype=dtype)
        k = torch.randn(b, s2, n2, d, dtype=dtype)
        w = torch.randn(b, s1, n1, dtype=weight_dtype)
        return q, k, w, None, None

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_tnd_fp16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        q, k, w, qlen, klen = self._gen_inputs(8, torch.float16, torch.float16, "TND")
        cpu_max, cpu_sum = self._softmax_lse_ref(q, k, w, "TND")
        npu_max, npu_sum = torch_npu.npu_dense_lightning_indexer_softmax_lse(
            q.npu(), k.npu(), w.npu(), actual_seq_qlen=qlen, actual_seq_klen=klen,
            layout="TND", sparse_mode=3)
        self.assertRtolEqual(cpu_max, npu_max.cpu(), prec=1e-3)
        self.assertRtolEqual(cpu_sum, npu_sum.cpu(), prec=1e-3)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_bsnd_fp16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        q, k, w, _, _ = self._gen_inputs(8, torch.float16, torch.float16, "BSND")
        cpu_max, cpu_sum = self._softmax_lse_ref(q, k, w, "BSND")
        npu_max, npu_sum = torch_npu.npu_dense_lightning_indexer_softmax_lse(
            q.npu(), k.npu(), w.npu(), layout="BSND", sparse_mode=3)
        self.assertRtolEqual(cpu_max, npu_max.cpu(), prec=1e-3)
        self.assertRtolEqual(cpu_sum, npu_sum.cpu(), prec=1e-3)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_tnd_bf16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        q, k, w, qlen, klen = self._gen_inputs(16, torch.bfloat16, torch.bfloat16, "TND")
        cpu_max, cpu_sum = self._softmax_lse_ref(q, k, w, "TND")
        npu_max, npu_sum = torch_npu.npu_dense_lightning_indexer_softmax_lse(
            q.npu(), k.npu(), w.npu(), actual_seq_qlen=qlen, actual_seq_klen=klen,
            layout="TND", sparse_mode=3)
        self.assertRtolEqual(cpu_max.to(torch.float32), npu_max.cpu(), prec=4e-3)
        self.assertRtolEqual(cpu_sum.to(torch.float32), npu_sum.cpu(), prec=4e-3)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_weight_fp32(self):
        # weight float32 is allowed to differ from query/key (fp16) dtype
        torch.manual_seed(0)
        np.random.seed(0)
        q, k, w, qlen, klen = self._gen_inputs(8, torch.float16, torch.float32, "TND")
        cpu_max, cpu_sum = self._softmax_lse_ref(q, k, w, "TND")
        npu_max, npu_sum = torch_npu.npu_dense_lightning_indexer_softmax_lse(
            q.npu(), k.npu(), w.npu(), actual_seq_qlen=qlen, actual_seq_klen=klen,
            layout="TND", sparse_mode=3)
        self.assertRtolEqual(cpu_max, npu_max.cpu(), prec=1e-3)
        self.assertRtolEqual(cpu_sum, npu_sum.cpu(), prec=1e-3)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_nidx1_64(self):
        torch.manual_seed(0)
        np.random.seed(0)
        q, k, w, qlen, klen = self._gen_inputs(64, torch.float16, torch.float16, "TND")
        cpu_max, cpu_sum = self._softmax_lse_ref(q, k, w, "TND")
        npu_max, npu_sum = torch_npu.npu_dense_lightning_indexer_softmax_lse(
            q.npu(), k.npu(), w.npu(), actual_seq_qlen=qlen, actual_seq_klen=klen,
            layout="TND", sparse_mode=3)
        self.assertRtolEqual(cpu_max, npu_max.cpu(), prec=1e-3)
        self.assertRtolEqual(cpu_sum, npu_sum.cpu(), prec=1e-3)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_graph_mode(self):
        torch.manual_seed(0)
        np.random.seed(0)
        q, k, w, _, _ = self._gen_inputs(8, torch.float16, torch.float16, "BSND")
        cpu_max, cpu_sum = self._softmax_lse_ref(q, k, w, "BSND")

        class Model(torch.nn.Module):
            def forward(self, q, k, w):
                return torch_npu.npu_dense_lightning_indexer_softmax_lse(
                    q, k, w, layout="BSND", sparse_mode=3)

        model = Model().npu()
        compiled = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
        npu_max, npu_sum = compiled(q.npu(), k.npu(), w.npu())
        self.assertRtolEqual(cpu_max, npu_max.cpu(), prec=1e-3)
        self.assertRtolEqual(cpu_sum, npu_sum.cpu(), prec=1e-3)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_output_shape_dtype(self):
        # BSND -> (B, Nidx2, S1); TND -> (Nidx2, T1); dtype always float32
        torch.manual_seed(0)
        np.random.seed(0)
        q, k, w, qlen, klen = self._gen_inputs(8, torch.float16, torch.float16, "BSND")
        bsnd_max, _ = torch_npu.npu_dense_lightning_indexer_softmax_lse(
            q.npu(), k.npu(), w.npu(), layout="BSND", sparse_mode=3)
        self.assertEqual(bsnd_max.shape, torch.Size([1, 1, 128]))
        self.assertEqual(bsnd_max.dtype, torch.float32)
        q2, k2, w2, qlen2, klen2 = self._gen_inputs(8, torch.float16, torch.float16, "TND")
        tnd_max, tnd_sum = torch_npu.npu_dense_lightning_indexer_softmax_lse(
            q2.npu(), k2.npu(), w2.npu(), actual_seq_qlen=qlen2, actual_seq_klen=klen2,
            layout="TND", sparse_mode=3)
        self.assertEqual(tnd_max.shape, torch.Size([1, 128]))
        self.assertEqual(tnd_sum.shape, torch.Size([1, 128]))
        self.assertEqual(tnd_max.dtype, torch.float32)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_neg_dtype_mismatch_query_key(self):
        q, _, w, qlen, klen = self._gen_inputs(8, torch.float16, torch.float16, "TND")
        n2, d = 1, 128
        k = torch.randn(256, n2, d, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_dense_lightning_indexer_softmax_lse(
                q.npu(), k.npu(), w.npu(), actual_seq_qlen=qlen, actual_seq_klen=klen,
                layout="TND", sparse_mode=3)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_neg_weight_dtype_mismatch(self):
        q, k, _, qlen, klen = self._gen_inputs(8, torch.float16, torch.float16, "TND")
        w = torch.randn(128, 8, dtype=torch.bfloat16)  # non-fp32 and != query fp16
        with self.assertRaises(RuntimeError):
            torch_npu.npu_dense_lightning_indexer_softmax_lse(
                q.npu(), k.npu(), w.npu(), actual_seq_qlen=qlen, actual_seq_klen=klen,
                layout="TND", sparse_mode=3)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_neg_sparse_mode(self):
        q, k, w, qlen, klen = self._gen_inputs(8, torch.float16, torch.float16, "TND")
        with self.assertRaises(RuntimeError):
            torch_npu.npu_dense_lightning_indexer_softmax_lse(
                q.npu(), k.npu(), w.npu(), actual_seq_qlen=qlen, actual_seq_klen=klen,
                layout="TND", sparse_mode=0)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is required")
    def test_neg_invalid_layout(self):
        q, k, w, qlen, klen = self._gen_inputs(8, torch.float16, torch.float16, "TND")
        with self.assertRaises(RuntimeError):
            torch_npu.npu_dense_lightning_indexer_softmax_lse(
                q.npu(), k.npu(), w.npu(), actual_seq_qlen=qlen, actual_seq_klen=klen,
                layout="BSH", sparse_mode=3)


if __name__ == "__main__":
    run_tests()
