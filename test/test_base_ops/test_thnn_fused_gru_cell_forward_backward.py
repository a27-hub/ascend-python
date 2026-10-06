import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestThnnFusedGruCellForwardBackward(TestCase):
    """Training-path coverage of the direct _thnn_fused_gru_cell call.

    Autograd is owned by the aten derivative formula: first-order backward
    dispatches to _thnn_fused_gru_cell_backward (fused on Ascend950, decomposed
    fallback on Ascend910B) and double-backward to
    _thnn_differentiable_gru_cell_backward.
    """

    def make_training_inputs(self, B=8, H=32, dtype=torch.float32):
        device = "npu"
        gi = torch.randn(B, 3 * H, dtype=dtype, device=device).clamp(-1.5, 1.5)
        gh = torch.randn(B, 3 * H, dtype=dtype, device=device).clamp(-1.5, 1.5)
        hx = torch.randn(B, H, dtype=dtype, device=device).clamp(-1.5, 1.5)
        b_ih = torch.randn(3 * H, dtype=dtype, device=device)
        b_hh = torch.randn(3 * H, dtype=dtype, device=device)
        return gi, gh, hx, b_ih, b_hh

    def reference_grads(self, gi, gh, hx, b_ih, b_hh, grad_hy):
        tensors = [t.detach().cpu().float().requires_grad_(True) for t in (gi, gh, hx, b_ih, b_hh)]
        r_gi, r_gh, r_hx, r_bi, r_bh = tensors
        ig = (r_gi + r_bi).unsafe_chunk(3, 1)
        hg = (r_gh + r_bh).unsafe_chunk(3, 1)
        r = torch.sigmoid(ig[0] + hg[0])
        z = torch.sigmoid(ig[1] + hg[1])
        n = torch.tanh(ig[2] + r * hg[2])
        hy = (1 - z) * n + z * r_hx
        (hy * grad_hy.cpu().float()).sum().backward()
        return [t.grad for t in tensors]

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_forward_backward(self):
        """Direct training call with biases: hy and all five gradients vs CPU."""
        gi, gh, hx, b_ih, b_hh = self.make_training_inputs()
        tensors = [t.detach().requires_grad_(True) for t in (gi, gh, hx, b_ih, b_hh)]
        t_gi, t_gh, t_hx, t_bi, t_bh = tensors
        grad_hy = torch.randn_like(hx)
        hy, workspace = torch.ops.aten._thnn_fused_gru_cell(t_gi, t_gh, t_hx, t_bi, t_bh)
        # The formula records a node, so a real workspace must have been produced
        # (lazily by the composite on Ascend910B, always by the fused kernel).
        self.assertIsNotNone(workspace)
        (hy * grad_hy).sum().backward()
        ref = self.reference_grads(gi, gh, hx, b_ih, b_hh, grad_hy)
        for npu_g, cpu_g in zip(tensors, ref):
            self.assertRtolEqual(npu_g.grad.cpu().numpy(), cpu_g.numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_double_backward(self):
        """Double backward of the direct call (create_graph=True routes through
        _thnn_differentiable_gru_cell_backward)."""
        gi, gh, hx, b_ih, b_hh = self.make_training_inputs(B=4, H=8)
        tensors = [t.detach().requires_grad_(True) for t in (gi, gh, hx, b_ih, b_hh)]
        hy, _ = torch.ops.aten._thnn_fused_gru_cell(*tensors)
        loss = (hy * hy).sum()
        first_grads = torch.autograd.grad(loss, tensors, create_graph=True)
        grad_norm = sum(g.float().pow(2).sum() for g in first_grads)
        second_grads = torch.autograd.grad(grad_norm, tensors)
        for g in second_grads:
            self.assertFalse(g is None)
            self.assertFalse(torch.all(torch.isnan(g.cpu())))

    @SupportedDevices(["Ascend910B"])
    def test_thnn_fused_gru_cell_lazy_workspace(self):
        """Lazy workspace semantics of the composite fallback (Ascend910B):
        the (B, 5H) workspace is built iff the autograd formula records a node
        (grad mode on and some input requires grad); inference and no_grad
        evaluation loops stay free of the extra allocation."""
        B, H = 8, 32

        gi, gh, hx = self.make_training_inputs(B=B, H=H)[:3]
        hy, ws = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        self.assertIsNone(ws)  # pure inference: no requires_grad input

        rg = [t.detach().requires_grad_(True) for t in (gi, gh, hx)]
        with torch.no_grad():
            hy2, ws2 = torch.ops.aten._thnn_fused_gru_cell(*rg)
        self.assertIsNone(ws2)  # eval loop: requires_grad inputs but grad mode off

        hy3, ws3 = torch.ops.aten._thnn_fused_gru_cell(*rg)
        self.assertIsNotNone(ws3)  # training: node will be recorded
        self.assertEqual(tuple(ws3.shape), (B, 5 * H))


if __name__ == "__main__":
    run_tests()
