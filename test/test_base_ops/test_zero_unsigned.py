import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestZeroUnsigned(TestCase):
    """zero_ / zeros on the dtypes aclnnInplaceZero has no kernel for: the unsigned
    16/32/64-bit integers and the 8-bit floats. op_api::zero_ zeroes the storage
    through an integer dtype of equal width; an all-zero byte pattern is 0 for the
    integer dtypes and +0.0 for both float8 formats."""

    dtypes = [torch.uint16, torch.uint32, torch.uint64]
    float8_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2]

    def test_zero_(self):
        for dtype in self.dtypes:
            # inputs are built on CPU: aclnnInplaceOne / fill_ have no unsigned kernels either
            npu_input = torch.ones(4, 3, dtype=dtype).npu()
            npu_input.zero_()
            self.assertEqual(npu_input.dtype, dtype)
            self.assertTrue(bool((npu_input.cpu() == 0).all()), msg=str(dtype))

    def test_zeros(self):
        for dtype in self.dtypes:
            npu_output = torch.zeros((2, 5), dtype=dtype, device="npu")
            self.assertEqual(npu_output.dtype, dtype)
            self.assertEqual(npu_output.shape, torch.Size([2, 5]))
            self.assertTrue(bool((npu_output.cpu() == 0).all()), msg=str(dtype))

    def test_zero_float8(self):
        for dtype in self.float8_dtypes:
            # built on CPU then moved: aclnnInplaceOne has no float8 kernel either
            npu_input = torch.ones(4, 3, dtype=torch.float32).to(dtype).npu()
            npu_input.zero_()
            self.assertEqual(npu_input.dtype, dtype)
            self.assertTrue(bool((npu_input.float().cpu() == 0).all()), msg=str(dtype))

    def test_zeros_float8(self):
        for dtype in self.float8_dtypes:
            npu_output = torch.zeros((2, 5), dtype=dtype, device="npu")
            self.assertEqual(npu_output.dtype, dtype)
            self.assertTrue(bool((npu_output.float().cpu() == 0).all()), msg=str(dtype))

    def test_zero_noncontiguous(self):
        for dtype in self.dtypes:
            npu_input = torch.ones(6, 4, dtype=dtype).npu().transpose(0, 1)
            npu_input.zero_()
            self.assertTrue(bool((npu_input.cpu() == 0).all()), msg=str(dtype))


if __name__ == "__main__":
    run_tests()
