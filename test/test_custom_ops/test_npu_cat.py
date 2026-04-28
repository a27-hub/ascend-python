import numpy
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

class TestNpuCat(TestCase):

    @SupportedDevices(['Ascend950'])
    def test_npu_cat_fp4_e2m1_dim0(self):
        """FP4(E2M1) 沿 dim=0 拼接，uint8 模拟 FP4 打包存储"""
        a_np = numpy.random.randint(0, 256, size=(2, 4), dtype=numpy.uint8)
        b_np = numpy.random.randint(0, 256, size=(3, 4), dtype=numpy.uint8)
        a = torch.from_numpy(a_np).npu()
        b = torch.from_numpy(b_np).npu()

        result = torch_npu.npu_cat([a, b], dim=0, input_dtype=297)

        expected = numpy.concatenate([a_np, b_np], axis=0)
        self.assertEqual(result.shape, torch.Size([5, 4]))
        self.assertEqual(result.dtype, torch.uint8)
        self.assertEqual(result.cpu().numpy(), expected)

    @SupportedDevices(['Ascend950'])
    def test_npu_cat_fallback_no_input_dtype(self):
        """No input_dtype: fallback to torch.cat"""
        a = torch.randn(2, 4, dtype=torch.float16).npu()
        b = torch.randn(3, 4, dtype=torch.float16).npu()

        result = torch_npu.npu_cat([a, b], dim=0)
        expected = torch.cat([a, b], dim=0)

        self.assertRtolEqual(result.cpu(), expected.cpu())

if __name__ == "__main__":
    run_tests()