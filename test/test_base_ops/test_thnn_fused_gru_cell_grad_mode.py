import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestThnnFusedGruCellGradMode(TestCase):
    """GradMode compatibility of the gru_cell / _thnn_fused_gru_cell paths:
    no_grad, inference_mode, mode switching and nested enable_grad must behave
    identically to eager PyTorch semantics while autograd is owned by the aten
    derivative formula."""

    def make_gru_cell_inputs(self, B=8, H=32, I=64):
        device = "npu"
        return (torch.randn(B, I, dtype=torch.float32, device=device),
                torch.randn(B, H, dtype=torch.float32, device=device),
                torch.randn(3 * H, I, dtype=torch.float32, device=device),
                torch.randn(3 * H, H, dtype=torch.float32, device=device),
                torch.randn(3 * H, dtype=torch.float32, device=device),
                torch.randn(3 * H, dtype=torch.float32, device=device))

    def make_thnn_inputs(self, B=8, H=32):
        device = "npu"
        x = torch.randn(B, H, dtype=torch.float32, device=device)
        hx = torch.randn(B, H, dtype=torch.float32, device=device)
        w_ih = torch.randn(3 * H, H, dtype=torch.float32, device=device)
        w_hh = torch.randn(3 * H, H, dtype=torch.float32, device=device)
        return x, hx, w_ih, w_hh

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_no_grad_consistent(self):
        """no_grad output matches the default-mode output."""
        inp = self.make_gru_cell_inputs()
        hy_d = torch.ops.aten.gru_cell(*inp)
        with torch.no_grad():
            hy_n = torch.ops.aten.gru_cell(*inp)
        self.assertFalse(hy_d.requires_grad or hy_n.requires_grad)
        self.assertRtolEqual(hy_d.cpu().numpy(), hy_n.cpu().numpy(), prec=1.e-5)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_inference_mode(self):
        inp = self.make_gru_cell_inputs()
        with torch.inference_mode():
            hy = torch.ops.aten.gru_cell(*inp)
        self.assertFalse(hy.requires_grad)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_inference_then_training(self):
        """After an inference-mode round, training still records gradients."""
        inp = self.make_gru_cell_inputs()
        with torch.inference_mode():
            torch.ops.aten.gru_cell(*inp)
        ts = [t.detach().clone().requires_grad_(True) for t in inp]
        hy = torch.ops.aten.gru_cell(*ts)
        hy.sum().backward()
        for t in ts:
            self.assertFalse(t.grad is None)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_enable_grad_in_no_grad(self):
        """enable_grad nested inside no_grad re-enables recording."""
        inp = self.make_gru_cell_inputs()
        with torch.no_grad():
            with torch.enable_grad():
                ts = [t.detach().clone().requires_grad_(True) for t in inp[:4]]
                hy = torch.ops.aten.gru_cell(*ts)
                self.assertTrue(hy.requires_grad)
                hy.sum().backward()
                for t in ts:
                    self.assertFalse(t.grad is None)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_grad_mode_switch(self):
        """default -> no_grad -> default produces identical outputs."""
        inp = self.make_gru_cell_inputs()
        hy1 = torch.ops.aten.gru_cell(*inp)
        with torch.no_grad():
            hy2 = torch.ops.aten.gru_cell(*inp)
        hy3 = torch.ops.aten.gru_cell(*inp)
        self.assertRtolEqual(hy1.cpu().numpy(), hy2.cpu().numpy(), prec=1.e-5)
        self.assertRtolEqual(hy1.cpu().numpy(), hy3.cpu().numpy(), prec=1.e-5)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_no_grad_output_inplace(self):
        """no_grad outputs are plain (non-inference) tensors and can be mutated."""
        inp = self.make_gru_cell_inputs()
        with torch.no_grad():
            hy = torch.ops.aten.gru_cell(*inp)
        hy.add_(1.0)
        self.assertFalse(hy.requires_grad)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_no_grad(self):
        """Fused-cell forward under no_grad matches the default-mode output."""
        x, hx, w_ih, w_hh = self.make_thnn_inputs()
        gi = x @ w_ih.t()
        gh = hx @ w_hh.t()
        hy_d = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        with torch.no_grad():
            hy_n = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        self.assertRtolEqual(hy_d[0].cpu().numpy(), hy_n[0].cpu().numpy(), prec=1.e-5)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_inference_mode(self):
        x, hx, w_ih, w_hh = self.make_thnn_inputs()
        gi = x @ w_ih.t()
        gh = hx @ w_hh.t()
        with torch.inference_mode():
            hy = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        self.assertFalse(hy[0].requires_grad)


if __name__ == "__main__":
    run_tests()
