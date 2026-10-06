import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestGruCellForwardBackward(TestCase):
    def make_training_inputs(self, batch_size, input_size, hidden_size, dtype):
        device = "npu"
        input = torch.randn(batch_size, input_size, dtype=dtype, device=device)
        hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        w_ih = torch.randn(3 * hidden_size, input_size, dtype=dtype, device=device)
        w_hh = torch.randn(3 * hidden_size, hidden_size, dtype=dtype, device=device)
        b_ih = torch.randn(3 * hidden_size, dtype=dtype, device=device)
        b_hh = torch.randn(3 * hidden_size, dtype=dtype, device=device)
        return input, hx, w_ih, w_hh, b_ih, b_hh

    def run_backward(self, input, hx, w_ih, w_hh, b_ih=None, b_hh=None, create_graph=False):
        tensors = [t.requires_grad_(True) for t in (input, hx, w_ih, w_hh)]
        if b_ih is not None:
            tensors.append(b_ih.requires_grad_(True))
            tensors.append(b_hh.requires_grad_(True))
        hy = torch.ops.aten.gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh)
        loss = (hy.float() * hy.float()).sum()
        grads = torch.autograd.grad(loss, tensors, create_graph=create_graph)
        return hy, tensors, grads

    def gru_cell_reference_grads(self, input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
        cpu_tensors = [t.detach().cpu().float().requires_grad_(True) for t in (input, hx, w_ih, w_hh)]
        cpu_input, cpu_hx, cpu_w_ih, cpu_w_hh = cpu_tensors
        cpu_b_ih = cpu_b_hh = None
        if b_ih is not None:
            cpu_b_ih = b_ih.detach().cpu().float().requires_grad_(True)
            cpu_b_hh = b_hh.detach().cpu().float().requires_grad_(True)
            cpu_tensors += [cpu_b_ih, cpu_b_hh]
        hy = torch.ops.aten.gru_cell(cpu_input, cpu_hx, cpu_w_ih, cpu_w_hh, cpu_b_ih, cpu_b_hh)
        (hy * hy).sum().backward()
        return [t.grad for t in cpu_tensors]

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_forward_backward_float32(self):
        """Forward + backward with float32, grads vs CPU."""
        inputs = self.make_training_inputs(8, 16, 32, torch.float32)
        hy, _, grads = self.run_backward(*inputs)
        ref = self.gru_cell_reference_grads(*inputs)
        for npu_g, cpu_g in zip(grads, ref):
            self.assertRtolEqual(npu_g.cpu().numpy(), cpu_g.numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_forward_backward_float16(self):
        """Forward + backward with float16."""
        inputs = self.make_training_inputs(4, 8, 16, torch.float16)
        hy, _, grads = self.run_backward(*inputs)
        ref = self.gru_cell_reference_grads(*inputs)
        for npu_g, cpu_g in zip(grads, ref):
            self.assertRtolEqual(npu_g.cpu().float().numpy(), cpu_g.numpy(), prec=5.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_forward_backward_no_bias(self):
        """Forward + backward without bias."""
        inputs = self.make_training_inputs(4, 32, 64, torch.float32)
        input, hx, w_ih, w_hh, _, _ = inputs
        hy, tensors, grads = self.run_backward(input, hx, w_ih, w_hh)
        ref = self.gru_cell_reference_grads(input, hx, w_ih, w_hh)
        for npu_g, cpu_g in zip(grads, ref):
            self.assertRtolEqual(npu_g.cpu().numpy(), cpu_g.numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_double_backward(self):
        """Double backward via create_graph=True (aten derivative formula routes
        the second order through _thnn_differentiable_gru_cell_backward)."""
        inputs = self.make_training_inputs(4, 8, 16, torch.float32)
        tensors = [t.requires_grad_(True) for t in inputs]
        input, hx, w_ih, w_hh, b_ih, b_hh = tensors
        hy = torch.ops.aten.gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh)
        loss = (hy * hy).sum()
        first_grads = torch.autograd.grad(loss, tensors, create_graph=True)
        grad_norm = sum(g.float().pow(2).sum() for g in first_grads)
        second_grads = torch.autograd.grad(grad_norm, tensors)
        for g in second_grads:
            self.assertFalse(g is None)
            self.assertFalse(torch.all(torch.isnan(g.cpu())))


if __name__ == "__main__":
    run_tests()
