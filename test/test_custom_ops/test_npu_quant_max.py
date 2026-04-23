import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuQuantMax(TestCase):

    # dst_dtype 支持的形式：
    # 1. torch.dtype 对象：torch.float8_e5m2, torch.float8_e4m3fn
    # 2. torch_npu 自定义 dtype：torch_npu.hifloat8
    # 3. 整数值：290(hifloat8), 291(float8_e5m2), 292(float8_e4m3fn)
    #
    # dst_dtype 到输出 dtype 的映射：
    DST_DTYPE_MAP = {
        290: torch.uint8,           # torch_npu.hifloat8
        291: torch.float8_e5m2,
        292: torch.float8_e4m3fn,
        torch.float8_e5m2: torch.float8_e5m2,
        torch.float8_e4m3fn: torch.float8_e4m3fn,
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
    def test_npu_quant_max_with_int_value(self):
        """测试使用整数值作为 dst_dtype 参数"""
        shape_format = [
            # (x_dtype, shape, round_mode, dst_dtype_int)
            (torch.float16, (16, 128), "rint", 291),    # float8_e5m2
            (torch.float16, (16, 128), "round", 292),   # float8_e4m3fn
            (torch.float16, (16, 128), "hybrid", 290),  # hifloat8
            (torch.bfloat16, (8, 32, 64), "rint", 291),
            (torch.bfloat16, (8, 32, 64), "round", 292),
            (torch.bfloat16, (8, 32, 64), "hybrid", 290),
        ]

        for x_dtype, shape, round_mode, dst_dtype in shape_format:
            cpu_x = torch.randn(shape).to(x_dtype)
            cpu_scale = torch.randn((1,)).to(torch.float32)
            npu_x = cpu_x.npu()
            npu_scale = cpu_scale.npu()

            npu_y, npu_amax = self.npu_op_exec(npu_x, npu_scale, round_mode, dst_dtype)

            # shape / dtype checks
            self.assertEqual(npu_y.shape, cpu_x.shape)
            self.assertEqual(npu_y.dtype, self.DST_DTYPE_MAP[dst_dtype])
            self.assertEqual(list(npu_amax.shape), [1])  # amax 必须为一维 tensor，shape [1]
            self.assertEqual(npu_amax.dtype, x_dtype)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_with_torch_dtype(self):
        """测试使用 torch.dtype 对象作为 dst_dtype 参数"""
        shape_format = [
            # (x_dtype, shape, round_mode, dst_dtype_torch)
            (torch.float16, (16, 128), "rint", torch.float8_e5m2),
            (torch.float16, (16, 128), "round", torch.float8_e4m3fn),
            (torch.bfloat16, (8, 32, 64), "rint", torch.float8_e5m2),
            (torch.bfloat16, (8, 32, 64), "round", torch.float8_e4m3fn),
        ]

        for x_dtype, shape, round_mode, dst_dtype in shape_format:
            cpu_x = torch.randn(shape).to(x_dtype)
            cpu_scale = torch.randn((1,)).to(torch.float32)
            npu_x = cpu_x.npu()
            npu_scale = cpu_scale.npu()

            npu_y, npu_amax = self.npu_op_exec(npu_x, npu_scale, round_mode, dst_dtype)

            # shape / dtype checks
            self.assertEqual(npu_y.shape, cpu_x.shape)
            self.assertEqual(npu_y.dtype, dst_dtype)
            self.assertEqual(list(npu_amax.shape), [1])
            self.assertEqual(npu_amax.dtype, x_dtype)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_with_torch_npu_dtype(self):
        """测试使用 torch_npu.hifloat8 作为 dst_dtype 参数"""
        # hifloat8 只支持 round 和 hybrid 舍入模式
        shape_format = [
            (torch.float16, (16, 128), "round", torch_npu.hifloat8),
            (torch.float16, (16, 128), "hybrid", torch_npu.hifloat8),
            (torch.bfloat16, (8, 32, 64), "round", torch_npu.hifloat8),
            (torch.bfloat16, (8, 32, 64), "hybrid", torch_npu.hifloat8),
        ]

        for x_dtype, shape, round_mode, dst_dtype in shape_format:
            cpu_x = torch.randn(shape).to(x_dtype)
            cpu_scale = torch.randn((1,)).to(torch.float32)
            npu_x = cpu_x.npu()
            npu_scale = cpu_scale.npu()

            npu_y, npu_amax = self.npu_op_exec(npu_x, npu_scale, round_mode, dst_dtype)

            # shape / dtype checks
            self.assertEqual(npu_y.shape, cpu_x.shape)
            self.assertEqual(npu_y.dtype, torch.uint8)  # hifloat8 用 torch.uint8 表示
            self.assertEqual(list(npu_amax.shape), [1])
            self.assertEqual(npu_amax.dtype, x_dtype)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_default_args(self):
        cpu_x = torch.randn((4, 16)).to(torch.float16)
        cpu_scale = torch.randn((1,)).to(torch.float32)
        npu_x = cpu_x.npu()
        npu_scale = cpu_scale.npu()

        # defaults: round_mode="rint", dst_dtype=291 (torch.float8_e5m2)
        y, amax = torch_npu.npu_quant_max(npu_x, npu_scale)
        self.assertEqual(y.shape, cpu_x.shape)
        self.assertEqual(y.dtype, torch.float8_e5m2)
        self.assertEqual(list(amax.shape), [1])
        self.assertEqual(amax.dtype, torch.float16)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_invalid_round_mode(self):
        npu_x = torch.randn((4, 16), dtype=torch.float16).npu()
        npu_scale = torch.randn((1,), dtype=torch.float32).npu()
        try:
            torch_npu.npu_quant_max(npu_x, npu_scale, round_mode="invalid", dst_dtype=291)
            self.fail("Do not throw the expected exception")
        except Exception as e:
            self.assertTrue("round_mode" in str(e), f"err message mismatch, exception:{e}")

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_invalid_dst_dtype(self):
        npu_x = torch.randn((4, 16), dtype=torch.float16).npu()
        npu_scale = torch.randn((1,), dtype=torch.float32).npu()
        try:
            torch_npu.npu_quant_max(npu_x, npu_scale, round_mode="rint", dst_dtype=-1)
            self.fail("Do not throw the expected exception")
        except Exception as e:
            self.assertTrue("dst_dtype" in str(e), f"err message mismatch, exception:{e}")

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_invalid_dst_dtype_torch(self):
        """测试传入不支持的 torch.dtype"""
        npu_x = torch.randn((4, 16), dtype=torch.float16).npu()
        npu_scale = torch.randn((1,), dtype=torch.float32).npu()
        try:
            # torch.float32 不是支持的输出类型
            torch_npu.npu_quant_max(npu_x, npu_scale, round_mode="rint", dst_dtype=torch.float32)
            self.fail("Do not throw the expected exception")
        except Exception as e:
            self.assertTrue("dst_dtype" in str(e), f"err message mismatch, exception:{e}")

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_invalid_scale_dim(self):
        # scale 必须为一维 tensor
        npu_x = torch.randn((4, 16), dtype=torch.float16).npu()
        npu_scale = torch.randn((2, 1), dtype=torch.float32).npu()  # 2D tensor，不符合要求
        try:
            torch_npu.npu_quant_max(npu_x, npu_scale, round_mode="rint", dst_dtype=291)
            self.fail("Do not throw the expected exception")
        except Exception as e:
            self.assertTrue("1D tensor" in str(e) or "dim" in str(e), f"err message mismatch, exception:{e}")

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_invalid_scale_shape(self):
        # scale shape 必须为 [1]
        npu_x = torch.randn((4, 16), dtype=torch.float16).npu()
        npu_scale = torch.randn((2,), dtype=torch.float32).npu()  # shape [2]，不符合要求
        try:
            torch_npu.npu_quant_max(npu_x, npu_scale, round_mode="rint", dst_dtype=291)
            self.fail("Do not throw the expected exception")
        except Exception as e:
            self.assertTrue("[1]" in str(e) or "shape" in str(e), f"err message mismatch, exception:{e}")

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_invalid_scale_dtype(self):
        # scale dtype 必须为 torch.float32
        npu_x = torch.randn((4, 16), dtype=torch.float16).npu()
        npu_scale = torch.randn((1,), dtype=torch.float16).npu()  # float16，不符合要求
        try:
            torch_npu.npu_quant_max(npu_x, npu_scale, round_mode="rint", dst_dtype=291)
            self.fail("Do not throw the expected exception")
        except Exception as e:
            self.assertTrue("float32" in str(e) or "dtype" in str(e), f"err message mismatch, exception:{e}")

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_empty_scale(self):
        # scale 不支持空 tensor
        npu_x = torch.randn((4, 16), dtype=torch.float16).npu()
        npu_scale = torch.randn((0,), dtype=torch.float32).npu()  # empty tensor
        try:
            torch_npu.npu_quant_max(npu_x, npu_scale, round_mode="rint", dst_dtype=291)
            self.fail("Do not throw the expected exception")
        except Exception as e:
            self.assertTrue("empty" in str(e), f"err message mismatch, exception:{e}")


if __name__ == "__main__":
    run_tests()
