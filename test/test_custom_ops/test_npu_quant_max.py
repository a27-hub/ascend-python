import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuQuantMax(TestCase):

    # dst_dtype int value -> expected torch dtype of quantized output
    DST_DTYPE_MAP = {
        290: torch.uint8,           # torch_npu.hifloat8
        291: torch.float8_e5m2,
        292: torch.float8_e4m3fn,
    }

    def cpu_op_exec(self, x, scale, dst_torch_dtype):
        # reference: y = quant(x * scale) cast to dst dtype; amax = max(|x|) with x's dtype
        x_fp32 = x.to(torch.float32)
        scale_fp32 = scale.to(torch.float32)
        y_fp32 = x_fp32 * scale_fp32
        y = y_fp32.to(dst_torch_dtype)
        amax = x_fp32.abs().max().to(x.dtype)
        return y, amax

    def npu_op_exec(self, x, scale, round_mode, dst_dtype):
        y, amax = torch_npu.npu_quant_max(x, scale, round_mode=round_mode, dst_dtype=dst_dtype)
        return y.cpu(), amax.cpu()

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max(self):
        shape_format = [
            # (x_dtype, shape, round_mode, dst_dtype)
            (torch.float16, (16, 128), "rint", 291),
            (torch.float16, (16, 128), "round", 292),
            (torch.float16, (16, 128), "hybrid", 290),
            (torch.bfloat16, (8, 32, 64), "rint", 291),
            (torch.bfloat16, (8, 32, 64), "round", 292),
            (torch.bfloat16, (8, 32, 64), "hybrid", 290),
        ]

        for x_dtype, shape, round_mode, dst_dtype in shape_format:
            cpu_x = torch.randn(shape).to(x_dtype)
            cpu_scale = torch.randn(shape[-1:]).to(torch.float32)
            npu_x = cpu_x.npu()
            npu_scale = cpu_scale.npu()

            npu_y, npu_amax = self.npu_op_exec(npu_x, npu_scale, round_mode, dst_dtype)

            # shape / dtype checks
            self.assertEqual(npu_y.shape, cpu_x.shape)
            self.assertEqual(npu_y.dtype, self.DST_DTYPE_MAP[dst_dtype])
            self.assertEqual(list(npu_amax.shape), [1])
            self.assertEqual(npu_amax.dtype, x_dtype)

            # numeric check on amax (absolute-max of x)
            expected_amax = cpu_x.to(torch.float32).abs().max()
            self.assertRtolEqual(npu_amax.to(torch.float32), expected_amax)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_default_args(self):
        cpu_x = torch.randn((4, 16)).to(torch.float16)
        cpu_scale = torch.randn((16,)).to(torch.float32)
        npu_x = cpu_x.npu()
        npu_scale = cpu_scale.npu()

        # defaults: round_mode="rint", dst_dtype=291
        y, amax = torch_npu.npu_quant_max(npu_x, npu_scale)
        self.assertEqual(y.shape, cpu_x.shape)
        self.assertEqual(y.dtype, torch.float8_e5m2)
        self.assertEqual(list(amax.shape), [1])
        self.assertEqual(amax.dtype, torch.float16)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_invalid_round_mode(self):
        npu_x = torch.randn((4, 16), dtype=torch.float16).npu()
        npu_scale = torch.randn((16,), dtype=torch.float32).npu()
        try:
            torch_npu.npu_quant_max(npu_x, npu_scale, round_mode="invalid", dst_dtype=291)
            self.fail("Do not throw the expected exception")
        except Exception as e:
            self.assertTrue("round_mode" in str(e), f"err message mismatch, exception:{e}")

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_invalid_dst_dtype(self):
        npu_x = torch.randn((4, 16), dtype=torch.float16).npu()
        npu_scale = torch.randn((16,), dtype=torch.float32).npu()
        try:
            torch_npu.npu_quant_max(npu_x, npu_scale, round_mode="rint", dst_dtype=-1)
            self.fail("Do not throw the expected exception")
        except Exception as e:
            self.assertTrue("dst_dtype" in str(e), f"err message mismatch, exception:{e}")


if __name__ == "__main__":
    run_tests()
