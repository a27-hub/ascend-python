import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestGruCellForwardBackward(TestCase):
    def make_training_inputs(self, batch_size, input_size, hidden_size, dtype):
        """Generate training inputs for gru_cell forward-backward tests.

        Returns:
            (input, hx, w_ih, w_hh, b_ih, b_hh) tensors on CPU with the given dtype.
        """
        input = (torch.randn(batch_size, input_size, dtype=torch.float32) * 0.8).to(dtype)
        hx = (torch.randn(batch_size, hidden_size, dtype=torch.float32) * 0.8).to(dtype)
        w_ih = (torch.randn(3 * hidden_size, input_size, dtype=torch.float32) * 0.1).to(dtype)
        w_hh = (torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32) * 0.1).to(dtype)
        b_ih = (torch.randn(3 * hidden_size, dtype=torch.float32) * 0.1).to(dtype)
        b_hh = (torch.randn(3 * hidden_size, dtype=torch.float32) * 0.1).to(dtype)
        return input, hx, w_ih, w_hh, b_ih, b_hh

    def run_backward(self, input, hx, w_ih, w_hh, b_ih=None, b_hh=None, create_graph=False):
        """Run gru_cell forward + backward, returning (hy, [grads]).

        Bias tensors that are None are excluded from requires_grad tracking.
        """
        tensors = [t.detach().clone().requires_grad_(True) for t in (input, hx, w_ih, w_hh)]
        if b_ih is not None:
            tensors.append(b_ih.detach().clone().requires_grad_(True))
        if b_hh is not None:
            tensors.append(b_hh.detach().clone().requires_grad_(True))
        hy = torch.ops.aten.gru_cell(*tensors)
        loss = (hy * hy).sum()
        loss.backward(create_graph=create_graph)
        return hy.detach(), [t.grad for t in tensors]

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_forward_backward_float32(self):
        """Test gru_cell training (forward + backward) with float32, grads vs CPU."""
        batch_size = 32
        hidden_size = 16
        input_size = 64

        npu_inputs = self.make_training_inputs(
            batch_size, input_size, hidden_size, torch.float32)
        cpu_inputs = [t.cpu().float() for t in npu_inputs]

        npu_hy, npu_grads = self.run_backward(*[t.to("npu") for t in npu_inputs])
        cpu_hy, cpu_grads = self.run_backward(*cpu_inputs)

        # Compare results
        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)
        for i, (npu_g, cpu_g) in enumerate(zip(npu_grads, cpu_grads)):
            self.assertRtolEqual(npu_g.cpu().numpy(), cpu_g.numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_forward_backward_float16(self):
        """Test gru_cell training with float16."""
        batch_size = 16
        hidden_size = 32
        input_size = 64

        npu_inputs = self.make_training_inputs(
            batch_size, input_size, hidden_size, torch.float16)
        cpu_inputs = [t.cpu().float() for t in npu_inputs]

        npu_hy, npu_grads = self.run_backward(*[t.to("npu") for t in npu_inputs])
        cpu_hy, cpu_grads = self.run_backward(*cpu_inputs)

        # Compare results (with higher tolerance for float16)
        self.assertRtolEqual(npu_hy.cpu().float().numpy(), cpu_hy.numpy(), prec=1.e-2)
        for npu_g, cpu_g in zip(npu_grads, cpu_grads):
            self.assertRtolEqual(npu_g.cpu().float().numpy(), cpu_g.numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_forward_backward_no_bias(self):
        """Test gru_cell training without bias."""
        batch_size = 4
        hidden_size = 128
        input_size = 256

        input, hx, w_ih, w_hh, _, _ = self.make_training_inputs(
            batch_size, input_size, hidden_size, torch.float32)
        npu_inputs = (input, hx, w_ih, w_hh)
        cpu_inputs = (input.cpu().float(), hx.cpu().float(),
                      w_ih.cpu().float(), w_hh.cpu().float())

        npu_hy, npu_grads = self.run_backward(*[t.to("npu") for t in npu_inputs])
        cpu_hy, cpu_grads = self.run_backward(*cpu_inputs)

        # Compare results (bias grads absent: only 4 tensors tracked)
        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)
        for i in range(4):
            self.assertRtolEqual(npu_grads[i].cpu().numpy(), cpu_grads[i].numpy(), prec=1.e-3)
        self.assertEqual(len(npu_grads), 4)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_double_backward(self):
        """Test gru_cell double backward (create_graph=True)."""
        batch_size = 4
        hidden_size = 32
        input_size = 64

        npu_inputs = self.make_training_inputs(
            batch_size, input_size, hidden_size, torch.float32)
        cpu_inputs = [t.cpu().float() for t in npu_inputs]

        def run_double(inputs):
            tensors = [t.detach().clone().requires_grad_(True) for t in inputs]
            hy = torch.ops.aten.gru_cell(*tensors)
            loss = (hy * hy).sum()
            first_grads = torch.autograd.grad(loss, tensors, create_graph=True)
            grad_norm = sum((g * g).sum() for g in first_grads)
            return torch.autograd.grad(grad_norm, tensors)

        npu_double_grads = run_double([t.to("npu") for t in npu_inputs])
        cpu_double_grads = run_double(cpu_inputs)

        # Compare results
        for npu_g, cpu_g in zip(npu_double_grads, cpu_double_grads):
            self.assertRtolEqual(npu_g.cpu().numpy(), cpu_g.float().numpy(), prec=1.e-3)


if __name__ == "__main__":
    run_tests()
