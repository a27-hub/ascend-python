import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestScaledGroupedMmV2(TestCase):

    # ========== 入参约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_mat_a_dim(self):
        """测试 mat_a 不是 2D 或 3D 时报错"""
        mat_a = torch.rand(2, 16, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 4D
        mat_b = torch.rand(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(4, 32, dtype=torch.float32)
        scale_b = torch.rand(4, 1, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("mat_a has to be 2 or 3d" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_mat_b_dim(self):
        """测试 mat_b 不是 2D 或 3D 时报错"""
        mat_a = torch.rand(4, 16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        mat_b = torch.rand(2, 32, 64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 4D
        scale_a = torch.rand(4, 16, dtype=torch.float32)
        scale_b = torch.rand(2, 1, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("mat_b has to be 2 or 3d" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_contraction_dim_mismatch(self):
        """测试 mat_a 和 mat_b 的收缩维度不匹配时报错"""
        mat_a = torch.rand(4, 16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=32
        mat_b = torch.rand(4, 64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=64
        scale_a = torch.rand(4, 16, dtype=torch.float32)
        scale_b = torch.rand(4, 1, 128, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("contraction dimension of mat_a and mat_b must match" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_mat_a_trailing_dim_not_divisible_16(self):
        """测试 mat_a 的最后一个维度不能被 16 整除时报错"""
        mat_a = torch.rand(4, 16, 33, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=33 不能被16整除
        mat_b = torch.rand(4, 33, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 16, dtype=torch.float32)
        scale_b = torch.rand(4, 1, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("Expected trailing dimension of mat_a to be divisible by 16" in str(context.exception))


    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_bias_not_supported(self):
        """测试传入 bias 时报错（当前不支持）"""
        mat_a = torch.rand(4, 16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 16, dtype=torch.float32)
        scale_b = torch.rand(4, 1, 64, dtype=torch.float32)
        bias = torch.rand(4, 64, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=None, bias=bias.npu(), out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("Bias not supported in _scaled_grouped_mm_v2 NPU adapter yet" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_missing_offsets_for_2d_matrix(self):
        """测试当有 2D 矩阵时未提供 offs 时报错"""
        mat_a = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(16, dtype=torch.float32)
        scale_b = torch.rand(4, 1, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(1,),
                use_fast_accum=False
            )
        self.assertTrue("Have to provide offsets if there is a 2d matrix" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_offs_dim(self):
        """测试 offs 不是 1D 时报错"""
        mat_a = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(16, dtype=torch.float32)
        scale_b = torch.rand(4, 1, 64, dtype=torch.float32)
        offs = torch.randint(0, 16, (2, 4), dtype=torch.int32)  # 2D offs
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=offs.npu(), bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(1,),
                use_fast_accum=False
            )
        self.assertTrue("offs has to be 1D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_offs_dtype(self):
        """测试 offs 数据类型不是 int32 时报错"""
        mat_a = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(16, dtype=torch.float32)
        scale_b = torch.rand(4, 1, 64, dtype=torch.float32)
        offs = torch.tensor([4, 8, 12, 16], dtype=torch.int64)  # int64 而不是 int32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=offs.npu(), bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(1,),
                use_fast_accum=False
            )
        self.assertTrue("Offsets have to be int32" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_out_dtype(self):
        """测试 out_dtype 不是 bfloat16 时报错"""
        mat_a = torch.rand(4, 16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 16, dtype=torch.float32)
        scale_b = torch.rand(4, 1, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=None, bias=None, out_dtype=torch.float32,  # 只支持 bfloat16
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("Only bf16 high precision output types are supported for _scaled_grouped_mm_v2" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_group_count_mismatch(self):
        """测试 mat_a 和 mat_b 都是 3D 时组数不匹配时报错"""
        mat_a = torch.rand(4, 16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 4 groups
        mat_b = torch.rand(8, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 8 groups
        scale_a = torch.rand(4, 16, dtype=torch.float32)
        scale_b = torch.rand(8, 1, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("mat_a and mat_b must have same number of groups when both are 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_both_2d_without_offs(self):
        """测试 mat_a 和 mat_b 都是 2D 时未提供 offs 时报错"""
        mat_a = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        scale_a = torch.rand(16, dtype=torch.float32)
        scale_b = torch.rand(1, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [0], [0],
                [scale_b.npu()], [0], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(1,),
                use_fast_accum=False
            )
        self.assertTrue("Have to provide offsets if there is a 2d matrix" in str(context.exception))


if __name__ == "__main__":
    run_tests()