import unittest

import torch
import torch_npu
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import TestCase, run_tests


class TestFusionAttentionTensorRngMeta(TestCase):
    """Meta contract of the tensor-RNG wrapper ops (see issue #694 / MR 6179).

    torch.compile relies on the meta outputs: the seed/offset/numels must stay
    tensors (shape [1], dtype int64) so AOTAutograd can thread the runtime RNG
    state into the backward graph instead of constant-folding it to 0.
    """

    def test_forward_meta_contract(self):
        with FakeTensorMode():
            q = torch.empty(2, 2, 8, 16, dtype=torch.float16, device="npu")
            k = torch.empty(2, 2, 8, 16, dtype=torch.float16, device="npu")
            v = torch.empty(2, 2, 8, 16, dtype=torch.float16, device="npu")
            out = torch.ops.npu.npu_fusion_attention_tensor_rng.default(
                q, k, v, 2, "BNSD", None, None, None, 0.25, 0.9,
            )
        self.assertEqual(len(out), 7)
        for i in (0, 1, 2, 3):
            self.assertIsInstance(out[i], torch.Tensor)
        for i in (4, 5, 6):
            self.assertIsInstance(out[i], torch.Tensor)
            self.assertEqual(out[i].dtype, torch.int64)
            self.assertEqual(tuple(out[i].shape), (1,))

    def test_grad_meta_contract(self):
        with FakeTensorMode():
            q = torch.empty(2, 2, 8, 16, dtype=torch.float16, device="npu")
            k = torch.empty(2, 2, 8, 16, dtype=torch.float16, device="npu")
            v = torch.empty(2, 2, 8, 16, dtype=torch.float16, device="npu")
            dy = torch.empty(2, 2, 8, 16, dtype=torch.float16, device="npu")
            rng = torch.empty([1], dtype=torch.int64, device="npu")
            dq, dk, dv, dpse, dsink = torch.ops.npu.npu_fusion_attention_grad_tensor_rng.default(
                q, k, v, dy, 2, "BNSD", seed=rng, offset=rng, numels=rng,
            )
        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)
        self.assertEqual(dpse.numel(), 0)
        self.assertEqual(dsink.numel(), 0)


@unittest.skipIf(
    "950" not in torch.npu.get_device_name(0).upper(),
    "regression only applies to Ascend950 (v1 fallback path)",
)
class TestSdpaDropoutCompileGrad(TestCase):
    """Compiled SDPA + dropout must produce the same gradients as eager on 950."""

    def test_dropout_grad_matches_eager(self):
        torch.manual_seed(558)
        torch.npu.manual_seed(558)
        B, H, S, E, NUM, DROPOUT = 4, 32, 38, 64, 3, 0.1

        def sdpa(q, k, v, dropout_p):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        qs, ks, vs, qc, kc, vc = [], [], [], [], [], []
        for _ in range(NUM):
            q = torch.randn(B, H, S, E, dtype=torch.half, device="npu").requires_grad_(True)
            k = torch.randn(B, H, S, E, dtype=torch.half, device="npu").requires_grad_(True)
            v = torch.randn(B, H, S, E, dtype=torch.half, device="npu").requires_grad_(True)
            qs.append(q)
            ks.append(k)
            vs.append(v)
            qc.append(q.clone().detach().requires_grad_(True))
            kc.append(q.clone().detach().requires_grad_(True))
            vc.append(q.clone().detach().requires_grad_(True))
            qc[-1].data = q.detach().clone()
            kc[-1].data = k.detach().clone()
            vc[-1].data = v.detach().clone()

        state = torch.npu.get_rng_state()
        eager_out, eager_grad = [], []
        for i in range(NUM):
            out = sdpa(qs[i], ks[i], vs[i], DROPOUT)
            out.sum().backward()
            eager_out.append(out.detach().clone())
            eager_grad.append([t.grad.detach().clone() for t in (qs[i], ks[i], vs[i])])

        compiled = torch.compile(sdpa, backend="aot_eager", dynamic=True)
        torch.npu.set_rng_state(state)
        for i in range(NUM):
            out = compiled(qc[i], kc[i], vc[i], DROPOUT)
            out.sum().backward()
            self.assertTrue(
                torch.allclose(eager_out[i], out, rtol=1e-4, atol=1e-4),
                msg=f"iter{i} forward mismatch",
            )
            for name, ref, got in zip(("q", "k", "v"), eager_grad[i], (qc[i].grad, kc[i].grad, vc[i].grad)):
                self.assertTrue(
                    torch.allclose(ref, got, rtol=1e-4, atol=1e-4),
                    msg=f"iter{i} grad_{name} mismatch (RNG state lost in backward)",
                )


if __name__ == "__main__":
    run_tests()
