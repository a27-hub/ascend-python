import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


torch.npu.set_compile_mode(jit_compile=True)


class TestConstantPadNdFp4(TestCase):
    """Tests for constant_pad_nd with FP4 (float4_e2m1fn_x2) dtype support.

    FP4 stores 2 values per byte, so:
    - All pad values must be even
    - The padded dimension = (original_dim * 2 + pad_left + pad_right) / 2
    """

    def _create_fp4_tensor(self, shape):
        """Create an FP4 tensor on NPU with given logical shape.

        Since float4_e2m1fn_x2 packs 2 fp4 values per byte,
        the last dim of the storage shape is shape[-1] // 2.
        """
        storage_shape = list(shape)
        storage_shape[-1] = shape[-1] // 2
        x = torch.randint(0, 16, storage_shape, dtype=torch.uint8)
        return x.npu().to(torch_npu.float4_e2m1fn_x2)

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_1d_even(self):
        """FP4 1D padding with even pad values."""
        x = self._create_fp4_tensor((8,))
        pad = (2, 4)  # both even
        result = torch.constant_pad_nd(x, pad, 0)
        # expected last dim = (8 + 2 + 4) // 2 = 7
        self.assertEqual(result.shape, (7,))

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_2d_even(self):
        """FP4 2D padding with even pad values."""
        x = self._create_fp4_tensor((4, 8))
        pad = (2, 4, 0, 2)  # all even
        result = torch.constant_pad_nd(x, pad, 0)
        # dim1 = (8 + 2 + 4) // 2 = 7, dim0 = (4 + 0 + 2) = 6
        self.assertEqual(result.shape, (6, 7,))

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_3d_even(self):
        """FP4 3D padding with even pad values."""
        x = self._create_fp4_tensor((2, 4, 8))
        pad = (0, 2, 2, 2, 0, 0)  # all even
        result = torch.constant_pad_nd(x, pad, 0)
        # dim2 = (8 + 0 + 2) // 2 = 5, dim1 = (4 + 2 + 2) = 8
        self.assertEqual(result.shape, (2, 8, 5))

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_zero(self):
        """FP4 padding with all zeros — shape unchanged (divided by 2)."""
        x = self._create_fp4_tensor((4, 8))
        pad = (0, 0)
        result = torch.constant_pad_nd(x, pad, 0)
        # dim1 = (8 + 0 + 0) // 2 = 4
        self.assertEqual(result.shape, (4, 4))

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_odd_raises(self):
        """FP4 padding with odd pad value must raise an error."""
        x = self._create_fp4_tensor((4, 8))
        pad = (1, 2)  # 1 is odd
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x, pad, 0)

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_all_odd_raises(self):
        """FP4 padding with all odd pad values must raise an error."""
        x = self._create_fp4_tensor((4, 8))
        pad = (3, 5)  # both odd
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x, pad, 0)

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_negative_even(self):
        """FP4 negative padding with even values — shrinks dimension."""
        x = self._create_fp4_tensor((4, 8))
        pad = (-2, 0)  # even negative
        result = torch.constant_pad_nd(x, pad, 0)
        # dim1 = (8 + (-2) + 0) // 2 = 3
        self.assertEqual(result.shape, (4, 3))

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_negative_odd_raises(self):
        """FP4 negative odd padding must raise an error (not divisible by 2)."""
        x = self._create_fp4_tensor((4, 8))
        pad = (-1, 0)  # -1 is odd
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x, pad, 0)

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_4d(self):
        """FP4 4D tensor padding."""
        x = self._create_fp4_tensor((1, 2, 4, 8))
        pad = (0, 2, 2, 0)
        result = torch.constant_pad_nd(x, pad, 0)
        # dim3 = (8 + 0 + 2) // 2 = 5, dim2 = (4 + 2 + 0) = 6
        self.assertEqual(result.shape, (1, 2, 6, 5))


if __name__ == "__main__":
    run_tests()
