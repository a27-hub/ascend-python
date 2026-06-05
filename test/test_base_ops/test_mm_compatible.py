import os
os.environ["TORCH_NPU_USE_COMPATIBLE_IMPL"] = "1"

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestMmCompatible(TestCase):
    """Test mm when input tensors are transposed-non-contiguous with requires_grad.

    In compatible mode, _matmul_impl's should_fold branch transposes the small
    tensor before calling mm, making it non-contiguous (stride(0)==1, stride(1)>1).
    The MmKernelNpuOpApi redirects to aclnnMatmul to avoid redundant CANN Transpose.
    """

    def test_mm_self_transposed_requires_grad(self):
        """self is transposed and requires_grad -> triggers aclnnMatmul"""
        for dtype in [torch.float32, torch.float16]:
            m, k, n = 8, 16, 12
            cpu_self_base = torch.randn(k, m, dtype=dtype)
            cpu_mat2_base = torch.randn(k, n, dtype=dtype)

            cpu_self_leaf = cpu_self_base.clone().requires_grad_(True)
            cpu_mat2_leaf = cpu_mat2_base.clone().requires_grad_(True)
            cpu_self = cpu_self_leaf.t()
            cpu_mat2 = cpu_mat2_leaf

            npu_self_leaf = cpu_self_base.npu().requires_grad_(True)
            npu_mat2_leaf = cpu_mat2_base.npu().requires_grad_(True)
            npu_self = npu_self_leaf.t()
            npu_mat2 = npu_mat2_leaf

            self.assertEqual(npu_self.stride(0), 1)
            self.assertGreater(npu_self.stride(1), 1)

            cpu_out = torch.mm(cpu_self, cpu_mat2)
            npu_out = torch.mm(npu_self, npu_mat2).cpu()

            cpu_out.backward(torch.ones_like(cpu_out))
            npu_out.backward(torch.ones_like(npu_out))

            cpu_out_np = cpu_out.detach().numpy().astype(npu_out.detach().numpy().dtype)
            npu_out_np = npu_out.detach().numpy()
            self.assertRtolEqual(cpu_out_np, npu_out_np)
            self.assertRtolEqual(cpu_self_leaf.grad.t().numpy(), npu_self_leaf.grad.t().cpu().numpy())
            self.assertRtolEqual(cpu_mat2_leaf.grad.numpy(), npu_mat2_leaf.grad.cpu().numpy())

    def test_mm_mat2_transposed_requires_grad(self):
        """mat2 is transposed and requires_grad -> triggers aclnnMatmul"""
        for dtype in [torch.float32, torch.float16]:
            m, k, n = 8, 16, 12
            cpu_self_base = torch.randn(m, k, dtype=dtype)
            cpu_mat2_base = torch.randn(n, k, dtype=dtype)

            cpu_self_leaf = cpu_self_base.clone().requires_grad_(True)
            cpu_mat2_leaf = cpu_mat2_base.clone().requires_grad_(True)
            cpu_self = cpu_self_leaf
            cpu_mat2 = cpu_mat2_leaf.t()

            npu_self_leaf = cpu_self_base.npu().requires_grad_(True)
            npu_mat2_leaf = cpu_mat2_base.npu().requires_grad_(True)
            npu_self = npu_self_leaf
            npu_mat2 = npu_mat2_leaf.t()

            self.assertEqual(npu_mat2.stride(0), 1)
            self.assertGreater(npu_mat2.stride(1), 1)

            cpu_out = torch.mm(cpu_self, cpu_mat2)
            npu_out = torch.mm(npu_self, npu_mat2).cpu()

            cpu_out.backward(torch.ones_like(cpu_out))
            npu_out.backward(torch.ones_like(npu_out))

            cpu_out_np = cpu_out.detach().numpy().astype(npu_out.detach().numpy().dtype)
            npu_out_np = npu_out.detach().numpy()
            self.assertRtolEqual(cpu_out_np, npu_out_np)
            self.assertRtolEqual(cpu_self_leaf.grad.numpy(), npu_self_leaf.grad.cpu().numpy())
            self.assertRtolEqual(cpu_mat2_leaf.grad.t().numpy(), npu_mat2_leaf.grad.t().cpu().numpy())

    def test_mm_both_transposed_requires_grad(self):
        """Both self and mat2 are transposed and require_grad"""
        m, k, n = 8, 16, 12
        cpu_self_base = torch.randn(k, m, dtype=torch.float32)
        cpu_mat2_base = torch.randn(n, k, dtype=torch.float32)

        cpu_self_leaf = cpu_self_base.clone().requires_grad_(True)
        cpu_mat2_leaf = cpu_mat2_base.clone().requires_grad_(True)
        cpu_self = cpu_self_leaf.t()
        cpu_mat2 = cpu_mat2_leaf.t()

        npu_self_leaf = cpu_self_base.npu().requires_grad_(True)
        npu_mat2_leaf = cpu_mat2_base.npu().requires_grad_(True)
        npu_self = npu_self_leaf.t()
        npu_mat2 = npu_mat2_leaf.t()

        self.assertEqual(npu_self.stride(0), 1)
        self.assertGreater(npu_self.stride(1), 1)
        self.assertEqual(npu_mat2.stride(0), 1)
        self.assertGreater(npu_mat2.stride(1), 1)

        cpu_out = torch.mm(cpu_self, cpu_mat2)
        npu_out = torch.mm(npu_self, npu_mat2).cpu()

        cpu_out.backward(torch.ones_like(cpu_out))
        npu_out.backward(torch.ones_like(npu_out))

        cpu_out_np = cpu_out.detach().numpy().astype(npu_out.detach().numpy().dtype)
        npu_out_np = npu_out.detach().numpy()
        self.assertRtolEqual(cpu_out_np, npu_out_np)
        self.assertRtolEqual(cpu_self_leaf.grad.t().numpy(), npu_self_leaf.grad.t().cpu().numpy())
        self.assertRtolEqual(cpu_mat2_leaf.grad.t().numpy(), npu_mat2_leaf.grad.t().cpu().numpy())

    def test_mm_transposed_no_grad_not_triggered(self):
        """Transposed but no requires_grad -> should NOT trigger aclnnMatmul branch,
        still uses aclnnMm. Verify correctness anyway."""
        m, k, n = 8, 16, 12
        cpu_self_base = torch.randn(k, m, dtype=torch.float32)
        cpu_mat2 = torch.randn(k, n, dtype=torch.float32)
        cpu_self = cpu_self_base.t()

        npu_self = cpu_self_base.npu().t()
        npu_mat2 = cpu_mat2.npu()

        self.assertEqual(npu_self.stride(0), 1)
        self.assertGreater(npu_self.stride(1), 1)

        cpu_out = torch.mm(cpu_self, cpu_mat2).numpy()
        npu_out = torch.mm(npu_self, npu_mat2).cpu().numpy()
        cpu_out = cpu_out.astype(npu_out.dtype)

        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
