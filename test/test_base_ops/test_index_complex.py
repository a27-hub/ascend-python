import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestIndexComplex(TestCase):
    """Advanced indexing on complex tensors (aclnnIndex has no complex kernel; op_api::index
    gathers through the real view instead)."""

    def _check(self, cpu_input, index_fn):
        npu_input = cpu_input.npu()
        cpu_output = index_fn(cpu_input)
        npu_output = index_fn(npu_input)
        self.assertEqual(npu_output.dtype, cpu_output.dtype)
        self.assertEqual(npu_output.shape, cpu_output.shape)
        self.assertRtolEqual(torch.view_as_real(npu_output.cpu()).numpy(),
                             torch.view_as_real(cpu_output).numpy())

    def test_index_tensor_complex64(self):
        cpu_input = torch.randn(64, 32, dtype=torch.complex64)
        idx = torch.randint(0, 64, (128,))
        self._check(cpu_input, lambda x: x[idx.to(x.device)])

    def test_index_tensor_complex128(self):
        cpu_input = torch.randn(16, 8, dtype=torch.complex128)
        idx = torch.randint(0, 16, (32,))
        self._check(cpu_input, lambda x: x[idx.to(x.device)])

    def test_index_mixed_slice_complex64(self):
        cpu_input = torch.randn(64, 32, 4, dtype=torch.complex64)
        idx = torch.randint(0, 64, (8,))
        self._check(cpu_input, lambda x: x[idx.to(x.device), :, 1:3])
        self._check(cpu_input, lambda x: x[:, idx.to(x.device) % 32])

    def test_index_bool_mask_complex64(self):
        cpu_input = torch.randn(64, 32, dtype=torch.complex64)
        mask = torch.rand(64, 32) > 0.5
        self._check(cpu_input, lambda x: x[mask.to(x.device)])

    def test_index_two_tensors_complex64(self):
        cpu_input = torch.randn(64, 32, dtype=torch.complex64)
        i0 = torch.randint(0, 64, (16,))
        i1 = torch.randint(0, 32, (16,))
        self._check(cpu_input, lambda x: x[i0.to(x.device), i1.to(x.device)])

    def test_index_noncontiguous_complex64(self):
        cpu_input = torch.randn(32, 64, dtype=torch.complex64).transpose(0, 1)
        idx = torch.randint(0, 64, (16,))
        self._check(cpu_input, lambda x: x[idx.to(x.device)])


if __name__ == "__main__":
    run_tests()
