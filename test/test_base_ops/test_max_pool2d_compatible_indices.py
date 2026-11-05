import os
os.environ["TORCH_NPU_USE_COMPATIBLE_IMPL"] = "1"

import torch
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False

POOL_2D = {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1, "ceil_mode": False}
POOL_2D_ATEN = ([2, 2], [2, 2], [0, 0], [1, 1], False)


class TestMaxPool2dCompatibleIndices(TestCase):
    """Compatible implementation returns real indices; they must follow the ATen Long contract."""

    def _pair(self, shape, dtype):
        # Strictly increasing values keep the per-window maximum unique, so indices compare exactly.
        numel = 1
        for dim in shape:
            numel *= dim
        cpu = torch.arange(numel, dtype=dtype).reshape(shape)
        return cpu, cpu.npu()

    def _check_forward(self, shape, dtype):
        cpu_input, npu_input = self._pair(shape, dtype)
        cpu_out, cpu_idx = F.max_pool2d(cpu_input, return_indices=True, **POOL_2D)
        npu_out, npu_idx = F.max_pool2d(npu_input, return_indices=True, **POOL_2D)

        self.assertEqual(npu_idx.dtype, torch.int64)
        self.assertEqual(npu_idx.shape, cpu_idx.shape)
        self.assertEqual(npu_idx.cpu(), cpu_idx)
        self.assertEqual(npu_out.cpu(), cpu_out)
        return cpu_input, npu_input, cpu_out, npu_out, cpu_idx, npu_idx

    def test_forward_long_indices_2d(self):
        for dtype in (torch.float32, torch.float16):
            for shape in ((1, 1, 6, 6), (2, 3, 4, 4)):
                self._check_forward(shape, dtype)

    def test_forward_out_long_writeback(self):
        cpu_input, npu_input = self._pair((1, 1, 6, 6), torch.float32)
        cpu_out, cpu_idx = F.max_pool2d(cpu_input, return_indices=True, **POOL_2D)

        npu_out = torch.empty_like(cpu_out).npu()
        npu_idx = torch.empty(cpu_idx.shape, dtype=torch.int64).npu()
        ret_out, ret_idx = torch.ops.aten.max_pool2d_with_indices.out(
            npu_input, *POOL_2D_ATEN, out=npu_out, indices=npu_idx)

        self.assertEqual(ret_out.data_ptr(), npu_out.data_ptr())
        self.assertEqual(ret_idx.data_ptr(), npu_idx.data_ptr())
        self.assertEqual(npu_idx.dtype, torch.int64)
        self.assertEqual(npu_idx.cpu(), cpu_idx)
        self.assertEqual(npu_out.cpu(), cpu_out)

    def test_max_unpool2d_consumes_pool_indices(self):
        for dtype in (torch.float32, torch.float16):
            cpu_input, _, cpu_pooled, npu_pooled, cpu_idx, npu_idx = self._check_forward(
                (1, 1, 6, 6), dtype)
            cpu_unpooled = F.max_unpool2d(cpu_pooled, cpu_idx, kernel_size=2, stride=2,
                                          output_size=cpu_input.shape[-2:])
            npu_unpooled = F.max_unpool2d(npu_pooled, npu_idx, kernel_size=2, stride=2,
                                          output_size=cpu_input.shape[-2:])
            self.assertEqual(npu_unpooled.cpu(), cpu_unpooled)

    def test_max_unpool1d_consumes_pool_indices(self):
        cpu_input, npu_input = self._pair((1, 2, 6), torch.float32)
        cpu_pooled, cpu_idx = F.max_pool1d(cpu_input, kernel_size=2, stride=2, return_indices=True)
        npu_pooled, npu_idx = F.max_pool1d(npu_input, kernel_size=2, stride=2, return_indices=True)

        self.assertEqual(npu_idx.dtype, torch.int64)
        self.assertEqual(npu_idx.cpu(), cpu_idx)

        cpu_unpooled = F.max_unpool1d(cpu_pooled, cpu_idx, kernel_size=2, stride=2,
                                      output_size=cpu_input.shape[-1:])
        npu_unpooled = F.max_unpool1d(npu_pooled, npu_idx, kernel_size=2, stride=2,
                                      output_size=cpu_input.shape[-1:])
        self.assertEqual(npu_unpooled.cpu(), cpu_unpooled)

    def test_backward_autograd_with_long_indices(self):
        # The compatible backward kernel supports float32; forward/unpool cover float16 above.
        cpu_input, npu_input = self._pair((1, 1, 6, 6), torch.float32)
        cpu_input = cpu_input.detach().requires_grad_(True)
        npu_input = npu_input.detach().requires_grad_(True)

        cpu_out = F.max_pool2d(cpu_input, **POOL_2D)
        npu_out = F.max_pool2d(npu_input, **POOL_2D)
        cpu_out.backward(torch.ones_like(cpu_out))
        npu_out.backward(torch.ones_like(npu_out))

        self.assertEqual(npu_input.grad.cpu(), cpu_input.grad)

    def test_backward_grad_input_with_long_indices(self):
        cpu_input, npu_input = self._pair((1, 1, 6, 6), torch.float32)
        cpu_out, cpu_idx = F.max_pool2d(cpu_input, return_indices=True, **POOL_2D)
        npu_out, npu_idx = F.max_pool2d(npu_input, return_indices=True, **POOL_2D)

        cpu_grad = torch.ones_like(cpu_out)
        cpu_gi = torch.ops.aten.max_pool2d_with_indices_backward.default(
            cpu_grad, cpu_input, *POOL_2D_ATEN, cpu_idx)

        npu_gi = torch.empty_like(npu_input)
        ret = torch.ops.aten.max_pool2d_with_indices_backward.grad_input(
            torch.ones_like(npu_out), npu_input, *POOL_2D_ATEN, npu_idx, grad_input=npu_gi)

        self.assertEqual(ret.data_ptr(), npu_gi.data_ptr())
        self.assertEqual(npu_gi.cpu(), cpu_gi)

    def test_backward_still_accepts_int32_indices(self):
        """Callers holding pre-existing int32 indices must keep working."""
        cpu_input, npu_input = self._pair((1, 1, 6, 6), torch.float32)
        cpu_out, cpu_idx = F.max_pool2d(cpu_input, return_indices=True, **POOL_2D)
        npu_out, npu_idx = F.max_pool2d(npu_input, return_indices=True, **POOL_2D)

        cpu_gi = torch.ops.aten.max_pool2d_with_indices_backward.default(
            torch.ones_like(cpu_out), cpu_input, *POOL_2D_ATEN, cpu_idx)
        npu_gi = torch.ops.aten.max_pool2d_with_indices_backward.default(
            torch.ones_like(npu_out), npu_input, *POOL_2D_ATEN, npu_idx.to(torch.int32))

        self.assertEqual(npu_gi.cpu(), cpu_gi)


if __name__ == "__main__":
    run_tests()
