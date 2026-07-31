# Copyright (c) 2025 Huawei Technologies Co., Ltd
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuFusedInferAttentionScore(TestCase):
    """Test for npu_fused_infer_attention_score — FusedInferAttentionScore V5.

    Reference: manual self-attention (Q@K^T / sqrt(d) → softmax → @V) on CPU.
    Layouts tested: BNSD, BSH.
    """

    def _supported_attention_bnsd(self, query, key, value, scale, atten_mask=None):
        """CPU reference: manual scaled dot-product attention for BNSD layout.
        BNSD: (Batch, NumHeads, SeqLen, HeadDim)
        Returns attention_out as float32 numpy.
        """
        q = query.float()
        k = key.float()
        v = value.float()
        d = q.size(-1)
        s = scale if scale > 0 else 1.0 / np.sqrt(d)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * s
        if atten_mask is not None:
            mask = atten_mask.float()
            while mask.dim() < attn_scores.dim():
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores + (1.0 - mask) * (-1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        return out.float().numpy()

    def _supported_attention_bsh(self, query, key, value, num_heads, scale):
        """CPU reference: manual scaled dot-product attention for BSH layout.
        BSH: (Batch, SeqLen, HiddenSize) where HiddenSize = NumHeads * HeadDim
        """
        B, S, H = query.shape
        D = H // num_heads
        q = query.float().view(B, S, num_heads, D).transpose(1, 2)
        k = key.float().view(B, -1, num_heads, D).transpose(1, 2)
        v = value.float().view(B, -1, num_heads, D).transpose(1, 2)
        return self._supported_attention_bnsd(q, k, v, scale)

    def _make_tensors(self, dtype, B=1, N=2, S=16, D=64, device="npu"):
        """Create valid query/key/value tensors for BNSD layout."""
        shape = (B, N, S, D)
        q = torch.randn(shape, dtype=dtype, device=device)
        k = torch.randn(shape, dtype=dtype, device=device)
        v = torch.randn(shape, dtype=dtype, device=device)
        return q, k, v

    def supported_op_exec(self, query, key, value, num_heads, input_layout="BNSD",
                          scale=1.0, atten_mask=None):
        """CPU reference implementation, returns (attention_out, softmax_lse) as numpy."""
        if input_layout == "BSH":
            ref_out = self._supported_attention_bsh(query, key, value, num_heads, scale)
        else:
            ref_out = self._supported_attention_bnsd(query, key, value, scale, atten_mask)
        return ref_out

    def custom_op_exec(self, query, key, value, num_heads, input_layout="BNSD",
                       scale=1.0, atten_mask=None, softmax_lse_flag=False,
                       num_key_value_heads=0):
        """NPU operator execution, returns (attention_out, softmax_lse) as CPU numpy."""
        q_npu = query.npu().clone() if not query.is_npu else query.clone()
        k_npu = key.npu().clone() if not key.is_npu else key.clone()
        v_npu = value.npu().clone() if not value.is_npu else value.clone()
        if atten_mask is not None:
            atten_mask = atten_mask.npu().clone() if not atten_mask.is_npu else atten_mask.clone()

        attn_out, lse = torch_npu.npu_fused_infer_attention_score(
            q_npu, k_npu, v_npu,
            num_heads=num_heads, input_layout=input_layout,
            scale=scale, atten_mask=atten_mask,
            softmax_lse_flag=softmax_lse_flag,
            num_key_value_heads=num_key_value_heads)
        return attn_out.cpu().float().numpy(), lse.cpu().float().numpy()

    # ── Positive Tests ──────────────────────────────────────────────

    @SupportedDevices(['Ascend910B'])
    def test_basic_bnsd_fp16(self, device="npu"):
        """BNSD layout, float16, basic GQA self-attention."""
        torch.manual_seed(0)
        np.random.seed(0)
        B, N, S, D = 1, 2, 16, 64
        q, k, v = self._make_tensors(torch.float16, B, N, S, D)
        scale = 1.0 / np.sqrt(D)

        ref = self.supported_op_exec(q.cpu(), k.cpu(), v.cpu(), N, "BNSD", scale)
        npu_out, npu_lse = self.custom_op_exec(q, k, v, N, "BNSD", scale)
        self.assertRtolEqual(ref, npu_out, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_basic_bnsd_bf16(self, device="npu"):
        """BNSD layout, bfloat16, basic GQA self-attention."""
        torch.manual_seed(0)
        np.random.seed(0)
        B, N, S, D = 1, 2, 16, 64
        q, k, v = self._make_tensors(torch.bfloat16, B, N, S, D)
        scale = 1.0 / np.sqrt(D)

        ref = self.supported_op_exec(q.cpu(), k.cpu(), v.cpu(), N, "BNSD", scale)
        npu_out, npu_lse = self.custom_op_exec(q, k, v, N, "BNSD", scale)
        self.assertRtolEqual(ref, npu_out, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_basic_bsh_fp16(self, device="npu"):
        """BSH layout, float16 — verify output shape and dtype."""
        torch.manual_seed(0)
        np.random.seed(0)
        N, D = 2, 64
        B, S, H = 1, 16, N * D
        q = torch.randn(B, S, H, dtype=torch.float16).npu()
        k = torch.randn(B, S, H, dtype=torch.float16).npu()
        v = torch.randn(B, S, H, dtype=torch.float16).npu()
        scale = 1.0 / np.sqrt(D)

        attn_out, lse = torch_npu.npu_fused_infer_attention_score(
            q, k, v, num_heads=N, input_layout="BSH", scale=scale)
        # BSH output matches query shape
        self.assertEqual(attn_out.shape, torch.Size([B, S, H]))
        self.assertEqual(attn_out.dtype, torch.float16)

    @SupportedDevices(['Ascend910B'])
    def test_with_atten_mask(self, device="npu"):
        """BNSD layout with attention mask — verify output shape."""
        torch.manual_seed(0)
        np.random.seed(0)
        B, N, S, D = 1, 2, 16, 64
        q, k, v = self._make_tensors(torch.float16, B, N, S, D)
        scale = 1.0 / np.sqrt(D)
        # Lower-triangular (causal) mask: shape (1, 1, S, S)
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool)).view(1, 1, S, S)

        attn_out, lse = torch_npu.npu_fused_infer_attention_score(
            q.npu(), k.npu(), v.npu(),
            num_heads=N, input_layout="BNSD", scale=scale,
            atten_mask=mask.npu())
        self.assertEqual(attn_out.shape, torch.Size([B, N, S, D]))
        self.assertEqual(attn_out.dtype, torch.float16)

    @SupportedDevices(['Ascend910B'])
    def test_with_gqa(self, device="npu"):
        """BNSD layout, GQA with num_key_value_heads != num_heads."""
        torch.manual_seed(0)
        np.random.seed(0)
        B, N, KV_N, S, D = 1, 4, 2, 16, 64
        q = torch.randn(B, N, S, D, dtype=torch.float16)
        k = torch.randn(B, KV_N, S, D, dtype=torch.float16)
        v = torch.randn(B, KV_N, S, D, dtype=torch.float16)
        scale = 1.0 / np.sqrt(D)

        # GQA reference: repeat KV heads to match Q heads
        k_rep = k.repeat_interleave(N // KV_N, dim=1)
        v_rep = v.repeat_interleave(N // KV_N, dim=1)
        ref = self.supported_op_exec(q, k_rep, v_rep, N, "BNSD", scale)

        npu_out, npu_lse = self.custom_op_exec(q, k, v, N, "BNSD", scale,
                                                num_key_value_heads=KV_N)
        self.assertRtolEqual(ref, npu_out, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_output_shape(self, device="npu"):
        """Verify output shapes for BNSD layout with softmax_lse."""
        torch.manual_seed(0)
        np.random.seed(0)
        B, N, S, D = 2, 4, 32, 64
        q, k, v = self._make_tensors(torch.float16, B, N, S, D)
        scale = 1.0 / np.sqrt(D)

        attn_out, lse = torch_npu.npu_fused_infer_attention_score(
            q.npu(), k.npu(), v.npu(),
            num_heads=N, input_layout="BNSD",
            scale=scale, softmax_lse_flag=True)
        self.assertEqual(attn_out.shape, torch.Size([B, N, S, D]))
        self.assertEqual(lse.shape, torch.Size([B, N, S, 1]))
        self.assertEqual(lse.dtype, torch.float32)

    # ── Negative Tests ──────────────────────────────────────────────

    @SupportedDevices(['Ascend910B'])
    def test_neg_query_invalid_dtype(self, device="npu"):
        """Invalid query dtype (int64 on unsupported path) should raise."""
        torch.manual_seed(0)
        q = torch.randint(0, 10, (1, 2, 16, 64), dtype=torch.int64).npu()
        k = torch.randn(1, 2, 16, 64, dtype=torch.float16).npu()
        v = torch.randn(1, 2, 16, 64, dtype=torch.float16).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=2, input_layout="BNSD")

    @SupportedDevices(['Ascend910B'])
    def test_neg_key_invalid_dtype(self, device="npu"):
        """Invalid key dtype should raise."""
        torch.manual_seed(0)
        q = torch.randn(1, 2, 16, 64, dtype=torch.float16).npu()
        k = torch.randint(0, 10, (1, 2, 16, 64), dtype=torch.int64).npu()
        v = torch.randn(1, 2, 16, 64, dtype=torch.float16).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=2, input_layout="BNSD")


if __name__ == "__main__":
    run_tests()
