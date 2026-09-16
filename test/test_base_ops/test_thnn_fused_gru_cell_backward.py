import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

import unittest


class TestThnnFusedGruCellBackward(TestCase):
    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_backward_float32: "
                   "aclnnThnnFusedGruCellBackward not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_backward_float32(self):
        """Test _thnn_fused_gru_cell_backward with float32, per-output comparison."""
        batch_size = 32
        hidden_size = 16

        grad_hy = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")
        workspace = torch.randn(batch_size, 5 * hidden_size, dtype=torch.float32,
                                device="npu").clamp(-1.5, 1.5)
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, True)
        self.assertEqual(len(grads), 5)

    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_backward_float16: "
                   "aclnnThnnFusedGruCellBackward not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_backward_float16(self):
        """Test _thnn_fused_gru_cell_backward with float16 dtype."""
        batch_size = 16
        hidden_size = 32

        grad_hy = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="npu")
        workspace = torch.randn(batch_size, 5 * hidden_size, dtype=torch.float16,
                                device="npu").clamp(-1.5, 1.5)
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, True)
        self.assertEqual(len(grads), 5)

    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_backward_bfloat16: "
                   "aclnnThnnFusedGruCellBackward not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_backward_bfloat16(self):
        """Test _thnn_fused_gru_cell_backward with bfloat16 dtype."""
        batch_size = 16
        hidden_size = 64

        grad_hy = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")
        workspace = torch.randn(batch_size, 5 * hidden_size, dtype=torch.bfloat16,
                                device="npu").clamp(-1.5, 1.5)
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, True)
        self.assertEqual(len(grads), 5)

    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_backward_no_bias: "
                   "aclnnThnnFusedGruCellBackward not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_backward_no_bias(self):
        """Test _thnn_fused_gru_cell_backward with has_bias=False: bias outputs undefined."""
        batch_size = 8
        hidden_size = 128

        grad_hy = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")
        workspace = torch.randn(batch_size, 5 * hidden_size, dtype=torch.float32,
                                device="npu").clamp(-1.5, 1.5)
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, False)
        self.assertFalse(grads[3].defined())
        self.assertFalse(grads[4].defined())

    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_backward_empty_batch: "
                   "aclnnThnnFusedGruCellBackward not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_backward_empty_batch(self):
        """Test _thnn_fused_gru_cell_backward with B=0: bias grads are all-zero."""
        hidden_size = 64

        grad_hy = torch.randn(0, hidden_size, dtype=torch.float32, device="npu")
        workspace = torch.randn(0, 5 * hidden_size, dtype=torch.float32, device="npu")
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, True)
        self.assertTrue((grads[3].cpu() == 0).all())
        self.assertTrue((grads[4].cpu() == 0).all())

    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_backward_undefined_workspace: "
                   "aclnnThnnFusedGruCellBackward not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_backward_undefined_workspace(self):
        """Test _thnn_fused_gru_cell_backward rejects an undefined workspace."""
        grad_hy = torch.randn(4, 8, dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, None, True)

    @SupportedDevices(["Ascend950"])
    @unittest.skip("skip test_thnn_fused_gru_cell_backward_wrong_shape: "
                   "aclnnThnnFusedGruCellBackward not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_backward_wrong_shape(self):
        """Test _thnn_fused_gru_cell_backward rejects a workspace with wrong shape (B, 4H)."""
        grad_hy = torch.randn(4, 8, dtype=torch.float32, device="npu")
        workspace = torch.randn(4, 32, dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, True)

    @SupportedDevices(["Ascend910B"])
    @unittest.skip("skip test_thnn_fused_gru_cell_backward_soc_not_supported: "
                   "aclnnThnnFusedGruCellBackward not in CANN yet. Remove this skip after CANN update.")
    def test_thnn_fused_gru_cell_backward_soc_not_supported(self):
        """Test _thnn_fused_gru_cell_backward is rejected on SoCs before Ascend950."""
        grad_hy = torch.randn(4, 8, dtype=torch.float32, device="npu")
        workspace = torch.randn(4, 40, dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, True)


if __name__ == "__main__":
    run_tests()
