import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices



class TestScaledMm(TestCase):

    # ========== 入参约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_mat_a_dtype(self):
        """测试 mat_a 不是 float8 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.float16)  # 使用 float16 而不是 float8
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("mat_a must be float8 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_mat_b_dtype(self):
        """测试 mat_b 不是 float8 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.float16)  # 使用 float16 而不是 float8
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("mat_b must be float8 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_two_float8_e5m2_not_supported(self):
        """测试两个 float8_e5m2 矩阵相乘时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e5m2)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e5m2)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("Multiplication of two Float8_e5m2 matrices is not supported" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_scale_a_dtype(self):
        """测试 scale_a 不是 float32 或 float8_e8m0 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float16)  # 使用 float16 而不是 float32
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("scale_a must be float32 or float8_e8m0 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_scale_b_dtype(self):
        """测试 scale_b 不是 float32 或 float8_e8m0 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float16)  # 使用 float16 而不是 float32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("scale_b must be float32 or float8_e8m0 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_out_dtype(self):
        """测试 out_dtype 不支持的类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.int8)
        self.assertTrue("out_dtype must be Float32, BFloat16, or Float16" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_mat_a_dim(self):
        """测试 mat_a 不是 2 维时报错"""
        x1 = torch.rand(2, 16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3维张量
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("mat_a must be a matrix" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_mat_b_dim(self):
        """测试 mat_b 不是 2 维时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(2, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3维张量
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("mat_b must be a matrix" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_shape_mismatch(self):
        """测试 mat_a 的列数与 mat_b 的行数不匹配时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 行数是64, 不是32
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("shapes cannot be multiplied" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_scale_result(self):
        """测试传入 scale_result 时报错（当前不支持输出 float8 类型）"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        scale_result = torch.rand(1, dtype=torch.float32)  # 传入 scale_result
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=None, scale_result=scale_result.npu(), out_dtype=torch.float32)
        self.assertTrue("scale_result is not supported currently" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_bias_size(self):
        """测试 bias 大小与 mat_b 的输出维度不匹配时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(32, dtype=torch.bfloat16)  # bias 大小应该是 64，但传入 32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.bfloat16)
        self.assertTrue("Bias must be size" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_bias_not_supported_with_float32_out(self):
        """测试 out_dtype 为 Float32 时传入 bias 报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(64, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.float32)
        self.assertTrue("Bias is not supported when out_dtype is set to Float32" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_bias_dtype(self):
        """测试 bias 数据类型不是 BFloat16 或 Half 时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(64, dtype=torch.float32)  # bias 应该是 BFloat16 或 Half
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.bfloat16)
        self.assertTrue("Bias must be BFloat16 or Half" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_bias_out_dtype_mismatch_bf16(self):
        """测试 out_dtype 为 BFloat16 但 bias 为 Float16 时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(64, dtype=torch.float16)  # out_dtype 为 bfloat16 时 bias 也应为 bfloat16
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.bfloat16)
        self.assertTrue("Bias must be BFloat16 to compute" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_bias_out_dtype_mismatch_fp16(self):
        """测试 out_dtype 为 Float16 但 bias 为 BFloat16 时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(64, dtype=torch.bfloat16)  # out_dtype 为 float16 时 bias 也应为 float16
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.float16)
        self.assertTrue("Bias must be Float16 to compute" in str(context.exception))


if __name__ == "__main__":
    run_tests()