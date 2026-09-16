import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

import unittest


class TestThnnFusedGruCell(TestCase):
    def thnn_fused_gru_cell_reference(self, input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
        """Reference implementation of _thnn_fused_gru_cell for CPU comparison.

        GRU cell equations (gate order: reset, input, new):
        r = sigmoid(ir + hr)
        z = sigmoid(iz + hz)
        n = tanh(in + r * hn)
        hy = (1 - z) * n + z * hx

        Args:
            input_gates: Tensor of shape (batch_size, 3*hidden_size) containing input gates
            hidden_gates: Tensor of shape (batch_size, 3*hidden_size) containing hidden gates
            hx: Tensor of shape (batch_size, hidden_size) - input hidden state
            input_bias: Optional bias for input gates, shape (3*hidden_size,)
            hidden_bias: Optional bias for hidden gates, shape (3*hidden_size,)

        Returns:
            hy: Tensor of shape (batch_size, hidden_size) - new hidden state
        """
        # Add biases if provided
        ig = input_gates.float()
        hg = hidden_gates.float()
        if input_bias is not None:
            ig = ig + input_bias.float()
        if hidden_bias is not None:
            hg = hg + hidden_bias.float()

        ir, iz, i_new = ig.chunk(3, dim=1)
        hr, hz, h_new = hg.chunk(3, dim=1)

        r = torch.sigmoid(ir + hr)
        z = torch.sigmoid(iz + hz)
        n = torch.tanh(i_new + r * h_new)
        hy = (1 - z) * n + z * hx.float()

        dtype = input_gates.dtype
        return hy.to(dtype)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_float32(self):
        """Test _thnn_fused_gru_cell with float32 dtype."""
        batch_size = 32
        hidden_size = 16

        input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float32, device="npu")
        hidden_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float32, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")
        input_bias = torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")
        hidden_bias = torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")

        # NPU forward
        npu_hy, _ = torch.ops.aten._thnn_fused_gru_cell(
            input_gates, hidden_gates, hx, input_bias, hidden_bias)
        # CPU reference
        cpu_hy = self.thnn_fused_gru_cell_reference(
            input_gates.cpu(), hidden_gates.cpu(), hx.cpu(), input_bias.cpu(), hidden_bias.cpu())

        # Compare results
        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_float16(self):
        """Test _thnn_fused_gru_cell with float16 dtype."""
        batch_size = 64
        hidden_size = 32

        input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float16, device="npu")
        hidden_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float16, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="npu")
        input_bias = torch.randn(3 * hidden_size, dtype=torch.float16, device="npu")
        hidden_bias = torch.randn(3 * hidden_size, dtype=torch.float16, device="npu")

        # NPU forward
        npu_hy, _ = torch.ops.aten._thnn_fused_gru_cell(
            input_gates, hidden_gates, hx, input_bias, hidden_bias)
        # CPU reference
        cpu_hy = self.thnn_fused_gru_cell_reference(
            input_gates.cpu(), hidden_gates.cpu(), hx.cpu(), input_bias.cpu(), hidden_bias.cpu())

        # Compare results (with higher tolerance for float16)
        self.assertRtolEqual(npu_hy.cpu().float().numpy(), cpu_hy.float().numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_bfloat16(self):
        """Test _thnn_fused_gru_cell with bfloat16 dtype."""
        batch_size = 16
        hidden_size = 64

        input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.bfloat16, device="npu")
        hidden_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.bfloat16, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")

        # NPU forward
        npu_hy, _ = torch.ops.aten._thnn_fused_gru_cell(input_gates, hidden_gates, hx)
        # CPU reference
        cpu_hy = self.thnn_fused_gru_cell_reference(
            input_gates.cpu(), hidden_gates.cpu(), hx.cpu())

        # Compare results (with higher tolerance for bfloat16)
        self.assertRtolEqual(npu_hy.cpu().float().numpy(), cpu_hy.float().numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_no_bias(self):
        """Test _thnn_fused_gru_cell without bias."""
        batch_size = 8
        hidden_size = 128

        input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float32, device="npu")
        hidden_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float32, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")

        # NPU forward
        npu_hy, _ = torch.ops.aten._thnn_fused_gru_cell(input_gates, hidden_gates, hx)
        # CPU reference
        cpu_hy = self.thnn_fused_gru_cell_reference(
            input_gates.cpu(), hidden_gates.cpu(), hx.cpu())

        # Compare results
        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_large_batch(self):
        """Test _thnn_fused_gru_cell with a large batch size."""
        batch_size = 512
        hidden_size = 1024

        input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float32, device="npu")
        hidden_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float32, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")

        # NPU forward
        npu_hy, _ = torch.ops.aten._thnn_fused_gru_cell(input_gates, hidden_gates, hx)
        # CPU reference
        cpu_hy = self.thnn_fused_gru_cell_reference(
            input_gates.cpu(), hidden_gates.cpu(), hx.cpu())

        # Compare results
        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_non_contiguous(self):
        """Test _thnn_fused_gru_cell with non-contiguous inputs."""
        batch_size = 16
        hidden_size = 32

        # Create non-contiguous tensors via slicing
        input_gates_base = torch.randn(batch_size, 2, 3 * hidden_size, dtype=torch.float32, device="npu")
        input_gates = input_gates_base[:, 1, :]
        hidden_gates_base = torch.randn(batch_size, 2, 3 * hidden_size, dtype=torch.float32, device="npu")
        hidden_gates = hidden_gates_base[:, 1, :]
        hx_base = torch.randn(batch_size, 2, hidden_size, dtype=torch.float32, device="npu")
        hx = hx_base[:, 1, :]

        # NPU forward
        npu_hy, _ = torch.ops.aten._thnn_fused_gru_cell(input_gates, hidden_gates, hx)
        # CPU reference
        cpu_hy = self.thnn_fused_gru_cell_reference(
            input_gates.contiguous().cpu(), hidden_gates.contiguous().cpu(), hx.contiguous().cpu())

        # Compare results
        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_fused_workspace: "
                   "aclnnThnnFusedGruCell not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_fused_workspace(self):
        """Test _thnn_fused_gru_cell workspace layout (fused path, Ascend950 only)."""
        batch_size = 32
        hidden_size = 16

        input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float32, device="npu")
        hidden_gates = torch.randn(batch_size, 3 * hidden_size, dtype=torch.float32, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")
        input_bias = torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")
        hidden_bias = torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")

        npu_hy, npu_workspace = torch.ops.aten._thnn_fused_gru_cell(
            input_gates, hidden_gates, hx, input_bias, hidden_bias)
        self.assertTrue(npu_workspace is not None and npu_workspace.numel() > 0)
        self.assertEqual(npu_workspace.shape, (batch_size, 5 * hidden_size))

    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_empty_batch: "
                   "aclnnThnnFusedGruCell not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_empty_batch(self):
        """Test _thnn_fused_gru_cell with an empty batch (B=0, fused path)."""
        hidden_size = 64

        input_gates = torch.randn(0, 3 * hidden_size, dtype=torch.float32, device="npu")
        hidden_gates = torch.randn(0, 3 * hidden_size, dtype=torch.float32, device="npu")
        hx = torch.randn(0, hidden_size, dtype=torch.float32, device="npu")

        npu_hy, _ = torch.ops.aten._thnn_fused_gru_cell(input_gates, hidden_gates, hx)
        self.assertEqual(npu_hy.shape, (0, hidden_size))

    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_empty_hidden: "
                   "aclnnThnnFusedGruCell not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_empty_hidden(self):
        """Test _thnn_fused_gru_cell with an empty hidden size (H=0, fused path)."""
        batch_size = 4

        input_gates = torch.randn(batch_size, 0, dtype=torch.float32, device="npu")
        hidden_gates = torch.randn(batch_size, 0, dtype=torch.float32, device="npu")
        hx = torch.randn(batch_size, 0, dtype=torch.float32, device="npu")

        npu_hy, _ = torch.ops.aten._thnn_fused_gru_cell(input_gates, hidden_gates, hx)
        self.assertEqual(npu_hy.shape, (batch_size, 0))


if __name__ == "__main__":
    run_tests()
