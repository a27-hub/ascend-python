import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

# ScalingType enum values (row-wise = 0)
SCALING_TYPE_ROWWISE = 0


class TestScaledGroupedMmV2(TestCase):

    # ========== 实现选择测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_implementation(self):
        """测试无法找到有效的 scaled gemm 实现时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16)  # 非 FP8 类型
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(ValueError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("failed to find valid scaled gemm implementation" in str(context.exception))

    # ========== 维度约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_mat_a_dim_1d(self):
        """测试 mat_a 为 1D 时报错"""
        mat_a = torch.rand(128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(0,),
                use_fast_accum=False
            )
        self.assertTrue("Input tensor mat_a must be 2D or 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_mat_a_dim_4d(self):
        """测试 mat_a 为 4D 时报错"""
        mat_a = torch.rand(2, 4, 16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 16, dtype=torch.float32)
        scale_b = torch.rand(4, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("Input tensor mat_a must be 2D or 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_mat_b_dim_1d(self):
        """测试 mat_b 为 1D 时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("Input tensor mat_b must be 2D or 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_mat_b_dim_4d(self):
        """测试 mat_b 为 4D 时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(2, 4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("Input tensor mat_b must be 2D or 3D" in str(context.exception))

    # ========== Scale维度约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_scale_a_dtype(self):
        """测试 scale_a 数据类型不支持时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.bfloat16)  # 不支持的dtype
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("scale_a must be float32 or float8_e8m0fnu" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_scale_b_dtype(self):
        """测试 scale_b 数据类型不支持时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.bfloat16)  # 不支持的dtype
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("scale_b must be float32 or float8_e8m0fnu" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_scale_a_dim_fp8(self):
        """测试 FP8 模式下 scale_a 维度不正确时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, 1, dtype=torch.float32)  # 3D for fp8 is invalid
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("scale_a dimension must be 1D or 2D for fp8" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_scale_b_dim_fp8(self):
        """测试 FP8 模式下 scale_b 维度不正确时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, 1, dtype=torch.float32)  # 3D for fp8 is invalid
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("scale_b dimension must be 1D or 2D for fp8" in str(context.exception))

    # ========== 收缩维度约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_contraction_dim_mismatch(self):
        """测试 mat_a 和 mat_b 的收缩维度不匹配时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=64
        mat_b = torch.rand(4, 128, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=128
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("contraction dimension mismatch between mat_a and mat_b" in str(context.exception))

    # ========== 16对齐约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_mat_a_trailing_dim_not_divisible_16(self):
        """测试 mat_a 的最后一个维度不能被 16 整除时报错"""
        mat_a = torch.rand(4, 128, 33, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=33 不能被16整除
        mat_b = torch.rand(4, 33, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("mat_a trailing dimension must be multiple of 16" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_mat_b_n_dim_not_divisible_16(self):
        """测试 mat_b 的 N 维度不能被 16 整除时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 33, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # N=33 不能被16整除
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 33, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("mat_b dimensions must be multiple of 16" in str(context.exception))

    # ========== 不支持特性测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_bias_not_supported(self):
        """测试传入 bias 时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        bias = torch.rand(4, 32, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=bias.npu(), out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("NPU _scaled_grouped_mm_v2 does not support bias parameter" in str(context.exception))

    # ========== Offsets约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_missing_offsets_for_2d_mat_a(self):
        """测试 mat_a 为 2D 但未提供 offs 时报错"""
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(1,),
                use_fast_accum=False
            )
        self.assertTrue("offsets required when using 2D input tensor" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_missing_offsets_for_2d_mat_b(self):
        """测试 mat_b 为 2D 但未提供 offs 时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        mat_b = torch.rand(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("offsets required when using 2D input tensor" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_offs_dim(self):
        """测试 offs 不是 1D 时报错"""
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        offs = torch.tensor([[256]], dtype=torch.int32)  # 2D tensor
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=offs.npu(), bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(1,),
                use_fast_accum=False
            )
        self.assertTrue("offsets tensor must be 1D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_offs_dtype(self):
        """测试 offs 不是 int32 时报错"""
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        offs = torch.tensor([256], dtype=torch.int64)  # int64 instead of int32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=offs.npu(), bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(1,),
                use_fast_accum=False
            )
        self.assertTrue("offsets data type must be int32" in str(context.exception))

    # ========== 输出类型约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_invalid_out_dtype(self):
        """测试 out_dtype 不是 bfloat16 时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.float32,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("NPU _scaled_grouped_mm_v2 only supports BF16 output type" in str(context.exception))

    # ========== K_SPLIT 不支持测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_v2_k_split_not_supported_single_weight(self):
        """测试 mat_b 只有 1 个 weight 时触发 K_SPLIT 报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 4 groups
        mat_b = torch.rand(1, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 1 weight shared
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(1, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm_v2(
                mat_a.npu(), mat_b.npu(),
                [scale_a.npu()], [SCALING_TYPE_ROWWISE], [0],
                [scale_b.npu()], [SCALING_TYPE_ROWWISE], [0],
                offs=None, bias=None, out_dtype=torch.bfloat16,
                contraction_dim=(2,),
                use_fast_accum=False
            )
        self.assertTrue("K_SPLIT (group_type=2) is not supported yet" in str(context.exception))


if __name__ == "__main__":
    run_tests()