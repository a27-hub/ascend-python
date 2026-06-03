import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestScaledGroupedMm(TestCase):

    # ========== 基本功能测试 ==========
    # 注意：由于 split_item 固定为 IN_NOT_SPLIT_OUT_SPLIT(2)，
    # group_type 不能为 DEFAULT_SPLIT(-1)，因此基本功能测试使用 M_SPLIT 场景

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_basic_function_2d_a_3d_b(self):
        """测试基本功能 - 2D mat_a 和 3D mat_b (M_SPLIT 场景)"""
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D [M, K]
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D [G, K, N]
        scale_a = torch.rand(256, dtype=torch.float32)  # KC mode: (M,)
        scale_b = torch.rand(4, 32, dtype=torch.float32)  # KC mode: (G, N)
        offs = torch.tensor([64, 128, 192, 256], dtype=torch.int32)  # offsets for 2D mat_a
        result = torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                          offs.npu(), None, None, torch.bfloat16, False)
        self.assertEqual(result.shape, (256, 32))

    # ========== 维度约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_a_dim_1d(self):
        """测试 mat_a 为 1D 时报错 """
        mat_a = torch.rand(128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(128, dtype=torch.float32)
        scale_b = torch.rand(4, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_a dimension must be 2D or 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_a_dim_4d(self):
        """测试 mat_a 为 4D 时报错 """
        mat_a = torch.rand(4, 2, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 2, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_a dimension must be 2D or 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_b_dim_1d(self):
        """测试 mat_b 为 1D 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_b dimension must be 2D or 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_b_dim_4d(self):
        """测试 mat_b 为 4D 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 2, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 2, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_b dimension must be 2D or 3D" in str(context.exception))

    # ========== Scale维度约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_scale_a_dtype(self):
        """测试 scale_a 数据类型不支持时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.bfloat16)  # 不支持的dtype
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("scale_a must be float32 or float8_e8m0fnu" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_scale_b_dtype(self):
        """测试 scale_b 数据类型不支持时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.bfloat16)  # 不支持的dtype
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("scale_b must be float32 or float8_e8m0fnu" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_scale_a_dim_fp8(self):
        """测试 FP8 模式下 scale_a 维度不正确时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, 1, dtype=torch.float32)  # 3D for fp8 is invalid
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("scale_a dimension must be 1D or 2D for fp8" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_scale_b_dim_fp8(self):
        """测试 FP8 模式下 scale_b 维度不正确时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, 1, dtype=torch.float32)  # 3D for fp8 is invalid
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("scale_b dimension must be 1D or 2D for fp8" in str(context.exception))

    # ========== 收缩维度约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_contraction_dim_mismatch(self):
        """测试 mat_a 和 mat_b 的收缩维度不匹配时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=64
        mat_b = torch.rand(4, 128, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=128
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("contraction dimension mismatch between mat_a and mat_b" in str(context.exception))

    # ========== 不支持特性测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_bias_not_supported(self):
        """测试传入 bias 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        bias = torch.rand(4, 32, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, bias.npu(), None, torch.bfloat16, False)
        self.assertTrue("NPU _scaled_grouped_mm does not support bias yet" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_scale_result_not_supported(self):
        """测试传入 scale_result 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        scale_result = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, scale_result.npu(), torch.bfloat16, False)
        self.assertTrue("NPU _scaled_grouped_mm does not support scale_result yet" in str(context.exception))

    # ========== Offsets约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_missing_offsets_for_2d_mat_a(self):
        """测试 mat_a 为 2D 但未提供 offs 时报错 """
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("offsets required when using 2D input tensor" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_missing_offsets_for_2d_mat_b(self):
        """测试 mat_b 为 2D 但未提供 offs 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        mat_b = torch.rand(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("offsets required when using 2D input tensor" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_offs_dim(self):
        """测试 offs 不是 1D 时报错 """
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        offs = torch.tensor([[256]], dtype=torch.int32)  # 2D tensor
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     offs.npu(), None, None, torch.bfloat16, False)
        self.assertTrue("offsets tensor must be 1D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_offs_dtype(self):
        """测试 offs 不是 int32 时报错 """
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        offs = torch.tensor([256], dtype=torch.int64)  # int64 instead of int32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     offs.npu(), None, None, torch.bfloat16, False)
        self.assertTrue("offsets data type must be int32" in str(context.exception))

    # ========== 输出类型约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_out_dtype(self):
        """测试 out_dtype 不是 bfloat16 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.float32, False)
        self.assertTrue("_scaled_grouped_mm on NPU only supports BF16 output type" in str(context.exception))

    # ========== K_SPLIT 不支持测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_k_split_not_supported_single_weight(self):
        """测试 mat_b 只有 1 个 weight 时触发 K_SPLIT 报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 4 groups
        mat_b = torch.rand(1, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 1 weight shared
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(1, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("K_SPLIT (group_type=2) is not supported yet" in str(context.exception))


if __name__ == "__main__":
    run_tests()