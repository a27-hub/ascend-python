import math
import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestScaledGroupedMm(TestCase):

    # ========== 入参约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_a_dim_1d(self):
        """测试 mat_a 为 1 维时报错"""
        mat_a = torch.rand(128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_a has to be 2 or 3d" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_a_dim_4d(self):
        """测试 mat_a 为 4 维时报错"""
        mat_a = torch.rand(4, 2, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 2, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 2, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 2, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_a has to be 2 or 3d" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_b_dim_1d(self):
        """测试 mat_b 为 1 维时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_b has to be 2 or 3d" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_b_dim_4d(self):
        """测试 mat_b 为 4 维时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 2, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 2, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_b has to be 2 or 3d" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_contraction_dim_mismatch(self):
        """测试 mat_a 和 mat_b 的收缩维度不匹配时报错"""
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D [M, K=64]
        mat_b = torch.rand(4, 128, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D [G, K=128, N]
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        offs = torch.tensor([256], dtype=torch.int32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     offs.npu(), None, None, torch.bfloat16, False)
        self.assertTrue("contraction dimension of mat_a and mat_b must match" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_bias_not_supported(self):
        """测试传入 bias 时报错（当前不支持）"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        bias = torch.rand(4, 32, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, bias.npu(), None, torch.bfloat16, False)
        self.assertTrue("Bias not supported" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_scale_result_not_supported(self):
        """测试传入 scale_result 时报错（当前不支持）"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        scale_result = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, scale_result.npu(), torch.bfloat16, False)
        self.assertTrue("Scale result not supported" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_missing_offsets_for_2d(self):
        """测试有 2D 矩阵但未提供 offs 时报错"""
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("Have to provide offsets" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_offs_dim(self):
        """测试 offs 不是 1D 时报错"""
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        offs = torch.tensor([[256]], dtype=torch.int32)  # 2D tensor
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     offs.npu(), None, None, torch.bfloat16, False)
        self.assertTrue("offs has to be 1D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_offs_dtype(self):
        """测试 offs 不是 int32 时报错"""
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        offs = torch.tensor([256], dtype=torch.int64)  # int64 instead of int32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     offs.npu(), None, None, torch.bfloat16, False)
        self.assertTrue("Offsets have to be int32" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_out_dtype(self):
        """测试 out_dtype 不是 bfloat16 时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.float32, False)
        self.assertTrue("Only bf16 high precision output types are supported" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_3d_group_mismatch(self):
        """测试两个 3D 矩阵组数不一致时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 4 groups
        mat_b = torch.rand(8, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 8 groups
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(8, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_a and mat_b must have same number of groups" in str(context.exception))


if __name__ == "__main__":
    run_tests()