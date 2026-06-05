import os
os.environ["TORCH_NPU_USE_COMPATIBLE_IMPL"] = "1"

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestMvCompatible(TestCase):
    """Test mv when the matrix is transposed-non-contiguous and vec requires_grad.

    In compatible mode, _matmul_impl's should_fold branch transposes the 3D tensor
    (via mT) before reshaping and calling mv, making the matrix transposed-non-contiguous
    (stride(0)==1, stride(1)>1). The MvKernelNpuOpApi redirects to aclnnMatmul
    when vec.requires_grad && is_transposed_2d(self) to avoid redundant CANN Transpose.
    """

    def test_mv_transposed_matrix_vec_requires_grad(self):
        """Matrix is transposed, vec requires_grad -> triggers aclnnMatmul"""
        for dtype in [torch.float32, torch.float16]:
            m, k = 8, 16
            cpu_mat_base = torch.randn(k, m, dtype=dtype)
            cpu_vec_base = torch.randn(k, dtype=dtype)

            cpu_mat = cpu_mat_base.t()
            cpu_vec = cpu_vec_base.clone().requires_grad_(True)
            npu_mat = cpu_mat_base.npu().t()
            npu_vec = cpu_vec_base.npu().requires_grad_(True)

            self.assertEqual(npu_mat.stride(0), 1)
            self.assertGreater(npu_mat.stride(1), 1)

            cpu_out = torch.mv(cpu_mat, cpu_vec)
            npu_out = torch.mv(npu_mat, npu_vec).cpu()

            cpu_out.backward(torch.ones_like(cpu_out))
            npu_out.backward(torch.ones_like(npu_out))

            cpu_out_np = cpu_out.detach().numpy().astype(npu_out.detach().numpy().dtype)
            npu_out_np = npu_out.detach().numpy()
            self.assertRtolEqual(cpu_out_np, npu_out_np)
            self.assertRtolEqual(cpu_vec.grad.numpy(), npu_vec.grad.cpu().numpy())

    def test_mv_transposed_matrix_vec_no_grad(self):
        """Matrix is transposed but vec doesn't require grad -> branch NOT triggered,
        still uses aclnnMv. Verify correctness anyway."""
        m, k = 8, 16
        cpu_mat_base = torch.randn(k, m, dtype=torch.float32)
        cpu_vec = torch.randn(k, dtype=torch.float32)
        cpu_mat = cpu_mat_base.t()

        npu_mat = cpu_mat_base.npu().t()
        npu_vec = cpu_vec.npu()

        self.assertEqual(npu_mat.stride(0), 1)
        self.assertGreater(npu_mat.stride(1), 1)

        cpu_out = torch.mv(cpu_mat, cpu_vec).numpy()
        npu_out = torch.mv(npu_mat, npu_vec).cpu().numpy()
        cpu_out = cpu_out.astype(npu_out.dtype)

        self.assertRtolEqual(cpu_out, npu_out)

    def test_mv_contiguous_matrix_vec_requires_grad(self):
        """Matrix is contiguous, vec requires_grad -> branch NOT triggered,
        still uses aclnnMv. Verify correctness anyway."""
        m, k = 8, 16
        cpu_mat = torch.randn(m, k, dtype=torch.float32)
        cpu_vec_base = torch.randn(k, dtype=torch.float32)

        cpu_vec = cpu_vec_base.clone().requires_grad_(True)
        npu_mat = cpu_mat.npu()
        npu_vec = cpu_vec_base.npu().requires_grad_(True)

        cpu_out = torch.mv(cpu_mat, cpu_vec)
        npu_out = torch.mv(npu_mat, npu_vec).cpu()

        cpu_out.backward(torch.ones_like(cpu_out))
        npu_out.backward(torch.ones_like(npu_out))

        cpu_out_np = cpu_out.detach().numpy().astype(npu_out.detach().numpy().dtype)
        npu_out_np = npu_out.detach().numpy()
        self.assertRtolEqual(cpu_out_np, npu_out_np)
        self.assertRtolEqual(cpu_vec.grad.numpy(), npu_vec.grad.cpu().numpy())

    def test_mv_transposed_matrix_backward(self):
        """Transposed matrix with full backward pass"""
        m, k = 12, 20
        cpu_mat_base = torch.randn(k, m, dtype=torch.float32)
        cpu_vec_base = torch.randn(k, dtype=torch.float32)

        cpu_mat_leaf = cpu_mat_base.clone().requires_grad_(True)
        cpu_mat = cpu_mat_leaf.t()
        cpu_vec = cpu_vec_base.clone().requires_grad_(True)
        npu_mat_leaf = cpu_mat_base.npu().requires_grad_(True)
        npu_mat = npu_mat_leaf.t()
        npu_vec = cpu_vec_base.npu().requires_grad_(True)

        self.assertEqual(npu_mat.stride(0), 1)
        self.assertGreater(npu_mat.stride(1), 1)

        cpu_out = torch.mv(cpu_mat, cpu_vec)
        npu_out = torch.mv(npu_mat, npu_vec).cpu()

        cpu_out.backward(torch.ones_like(cpu_out))
        npu_out.backward(torch.ones_like(npu_out))

        cpu_out_np = cpu_out.detach().numpy().astype(npu_out.detach().numpy().dtype)
        npu_out_np = npu_out.detach().numpy()
        self.assertRtolEqual(cpu_out_np, npu_out_np)
        self.assertRtolEqual(cpu_vec.grad.numpy(), npu_vec.grad.cpu().numpy())


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
