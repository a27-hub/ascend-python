import unittest
import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests


def _ref_attention(q, k, v, scale, is_causal=True, window_left=-1):
    q, k, v = (t.float() for t in (q, k, v))
    n_q, n_kv = q.shape[1], k.shape[1]
    if n_q != n_kv:
        k = k.repeat_interleave(n_q // n_kv, dim=1)
        v = v.repeat_interleave(n_q // n_kv, dim=1)
    s = torch.einsum("qnd,knd->nqk", q, k) * scale
    if is_causal:
        q_len, k_len = s.shape[-2:]
        mask = torch.ones(q_len, k_len, dtype=torch.bool).triu(
            1 + k_len - q_len
        )
        s = s.masked_fill(mask, float("-inf"))
    if window_left >= 0:
        q_len, k_len = s.shape[-2:]
        q_positions = torch.arange(q_len)[:, None] + k_len - q_len
        k_positions = torch.arange(k_len)[None, :]
        s = s.masked_fill(
            k_positions < q_positions - window_left, float("-inf")
        )
    lse = torch.logsumexp(s, dim=-1)
    out = torch.einsum("nqk,knd->qnd", torch.softmax(s, dim=-1), v)
    return out, lse


@unittest.skipIf(
    tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 16),
    "native FlashAttention registration requires PyTorch 2.16 or newer",
)
class TestFlashAttention(TestCase):
    def test_native_registration(self):
        for name in ("_flash_attention_forward", "_flash_attention_backward"):
            self.assertTrue(
                torch._C._dispatch_has_kernel_for_dispatch_key(
                    f"aten::{name}", "PrivateUse1"
                )
            )

    def test_dense_causal_forward_backward(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                b, s, n, d = 2, 128, 4, 64
                q, k, v = (
                    torch.randn(b, s, n, d, dtype=dtype, device="npu")
                    for _ in range(3)
                )
                scale = d**-0.5
                out, lse, rng, unused, _ = (
                    torch.ops.aten._flash_attention_forward(
                        q,
                        k,
                        v,
                        None,
                        None,
                        s,
                        s,
                        0.0,
                        True,
                        False,
                        scale=scale,
                    )
                )
                grad_out = torch.randn_like(out)
                grads = torch.ops.aten._flash_attention_backward(
                    grad_out,
                    q,
                    k,
                    v,
                    out,
                    lse,
                    None,
                    None,
                    s,
                    s,
                    0.0,
                    True,
                    rng,
                    unused,
                    scale=scale,
                )
                self.assertEqual(out.shape, (b, s, n, d))
                self.assertEqual(lse.shape, (b, n, s))
                for i in range(b):
                    refs = tuple(
                        x[i].float().cpu().requires_grad_(True)
                        for x in (q, k, v)
                    )
                    ref_out, ref_lse = _ref_attention(*refs, scale)
                    ref_grads = torch.autograd.grad(
                        ref_out, refs, grad_out[i].float().cpu()
                    )
                    for got, ref, tolerance in (
                        (out[i], ref_out, 2e-2),
                        (lse[i], ref_lse, 2e-2),
                        *(
                            (got[i], ref, 5e-2)
                            for got, ref in zip(grads, ref_grads)
                        ),
                    ):
                        self.assertRtolEqual(
                            got.float().cpu().numpy(),
                            ref.detach().numpy(),
                            prec=tolerance,
                        )

    def test_varlen_asymmetric_forward_backward(self):
        for dtype in (torch.bfloat16, torch.float16):
            for is_causal in (False, True):
                with self.subTest(is_causal=is_causal, dtype=dtype):
                    torch.manual_seed(0)
                    q_lengths, k_lengths = [32, 64], [48, 96]
                    n, n_kv, d = 4, 2, 64
                    q = torch.randn(
                        sum(q_lengths), n, d, dtype=dtype, device="npu"
                    )
                    k = torch.randn(
                        sum(k_lengths), n_kv, d, dtype=dtype, device="npu"
                    )
                    v = torch.randn_like(k)
                    cu_q = torch.tensor(
                        [0, 32, 96], dtype=torch.int32, device="npu"
                    )
                    cu_k = torch.tensor(
                        [0, 48, 144], dtype=torch.int32, device="npu"
                    )
                    out, lse, rng, unused, _ = (
                        torch.ops.aten._flash_attention_forward(
                            q,
                            k,
                            v,
                            cu_q,
                            cu_k,
                            max(q_lengths),
                            max(k_lengths),
                            0.0,
                            is_causal,
                            False,
                        )
                    )
                    grad_out = torch.randn_like(out)
                    grads = torch.ops.aten._flash_attention_backward(
                        grad_out,
                        q,
                        k,
                        v,
                        out,
                        lse,
                        cu_q,
                        cu_k,
                        max(q_lengths),
                        max(k_lengths),
                        0.0,
                        is_causal,
                        rng,
                        unused,
                    )

                    qr, kr, vr = (
                        x.float().cpu().requires_grad_(True) for x in (q, k, v)
                    )
                    ref_outs, ref_lses = [], []
                    q_start = k_start = 0
                    for q_len, k_len in zip(q_lengths, k_lengths):
                        q_slice = slice(q_start, q_start + q_len)
                        k_slice = slice(k_start, k_start + k_len)
                        ref_out, ref_lse = _ref_attention(
                            qr[q_slice],
                            kr[k_slice],
                            vr[k_slice],
                            d**-0.5,
                            is_causal,
                        )
                        ref_outs.append(ref_out)
                        ref_lses.append(ref_lse)
                        q_start += q_len
                        k_start += k_len
                    ref_out = torch.cat(ref_outs)
                    ref_lse = torch.cat(ref_lses, dim=1)
                    ref_grads = torch.autograd.grad(
                        ref_out, (qr, kr, vr), grad_out.float().cpu()
                    )
                    self.assertEqual(lse.shape, (n, sum(q_lengths)))
                    for got, ref, tolerance in (
                        (out, ref_out, 2e-2),
                        (lse, ref_lse, 2e-2),
                        *(
                            (got, ref, 5e-2)
                            for got, ref in zip(grads, ref_grads)
                        ),
                    ):
                        self.assertRtolEqual(
                            got.float().cpu().numpy(),
                            ref.detach().numpy(),
                            prec=tolerance,
                        )

    def test_varlen_shared_lengths_left_window(self):
        torch.manual_seed(1)
        lengths = [32, 64]
        cu = torch.tensor([0, 32, 96], dtype=torch.int32, device="npu")
        q, k, v = (
            torch.randn(96, 2, 64, dtype=torch.bfloat16, device="npu")
            for _ in range(3)
        )
        out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(
            q, k, v, cu, cu, 64, 64, 0.0, True, False, window_size_left=8
        )
        grad_out = torch.randn_like(out)
        grads = torch.ops.aten._flash_attention_backward(
            grad_out,
            q,
            k,
            v,
            out,
            lse,
            cu,
            cu,
            64,
            64,
            0.0,
            True,
            rng,
            unused,
            window_size_left=8,
        )
        refs = tuple(x.float().cpu().requires_grad_(True) for x in (q, k, v))
        ref_outs, ref_lses = [], []
        start = 0
        for length in lengths:
            section = slice(start, start + length)
            ref_out, ref_lse = _ref_attention(
                *(x[section] for x in refs), 64**-0.5, window_left=8
            )
            ref_outs.append(ref_out)
            ref_lses.append(ref_lse)
            start += length
        ref_out = torch.cat(ref_outs)
        ref_lse = torch.cat(ref_lses, dim=1)
        ref_grads = torch.autograd.grad(ref_out, refs, grad_out.float().cpu())
        for got, ref in zip(
            (out, lse, *grads), (ref_out, ref_lse, *ref_grads)
        ):
            self.assertRtolEqual(
                got.float().cpu().numpy(), ref.detach().numpy(), prec=5e-2
            )

    def test_unsupported_features_raise(self):
        s, n, d = 32, 2, 64
        q, k, v = (
            torch.randn(1, s, n, d, dtype=torch.bfloat16, device="npu")
            for _ in range(3)
        )
        with self.assertRaises(NotImplementedError):
            torch.ops.aten._flash_attention_forward(
                q, k, v, None, None, s, s, 0.1, True, False
            )


if __name__ == "__main__":
    run_tests()
