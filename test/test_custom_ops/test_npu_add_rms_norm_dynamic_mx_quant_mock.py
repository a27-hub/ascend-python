import math
import unittest
from unittest.mock import patch, MagicMock
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


def _mock_npu_add_rms_norm(x1, x2, gamma, epsilon=1e-06):
    """Mock: 返回与真实算子相同 shape/dtype 的假数据"""
    x_out = x1 + x2
    y_norm = torch.randn_like(x1)
    # rstd shape: x1 前几维 + 尾维=1
    rstd_shape = list(x1.shape[:-1]) + [1]
    rstd = torch.randn(rstd_shape, dtype=x1.dtype, device=x1.device)
    return y_norm, rstd, x_out


def _mock_npu_dynamic_mx_quant(input, axis=-1, round_mode="rint",
                                dst_type=296, block_size=32, scale_alg=0):
    """Mock: 返回与真实算子相同 shape/dtype 的假数据"""
    dim_num = input.dim()
    last_dim = input.size(dim_num - 1)

    # y shape/dtype 取决于 dst_type
    fp4_types = {torch_npu.float4_e2m1fn_x2, torch_npu.float4_e1m2fn_x2}
    if dst_type in fp4_types or dst_type in {296, 297, 40, 41}:
        y_shape = list(input.shape[:-1]) + [last_dim // 2]
        y = torch.randint(0, 255, y_shape, dtype=torch.uint8, device=input.device)
    else:
        y = torch.randn_like(input).to(torch.float8_e5m2)

    # mxscale shape: 前几维 + ceil(last_dim/32) 偶数对齐 + 尾维=2
    scale_dim = math.ceil(last_dim / 32)
    scale_dim = (scale_dim + 1) // 2  # 偶数 pad
    mxscale_shape = list(input.shape[:-1]) + [scale_dim, 2]
    mxscale = torch.randint(0, 255, mxscale_shape, dtype=torch.uint8, device=input.device)

    return y, mxscale


def _mock_npu_add_rms_norm_dynamic_mx_quant(x1, x2, gamma, beta=None, epsilon=1e-06,
                                             scale_alg=0, round_mode="rint", dst_type=296):
    """Mock: 复用上面两个 mock 的逻辑，模拟融合算子输出"""
    y_norm, rstd, x_out = _mock_npu_add_rms_norm(x1, x2, gamma, epsilon=epsilon)
    if beta is not None:
        y_norm = y_norm + beta
    y, mxscale = _mock_npu_dynamic_mx_quant(
        y_norm, axis=-1, round_mode=round_mode,
        dst_type=dst_type, block_size=32, scale_alg=scale_alg
    )
    return y, x_out, mxscale, rstd


class TestAddRmsNormDynamicMxQuantMock(TestCase):
    """Mock 测试：不依赖 NPU，验证测试框架调用链和 assert 结构"""

    def custom_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-06,
                       scale_alg=0, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2):
        return torch_npu.npu_add_rms_norm_dynamic_mx_quant(
            x1, x2, gamma, beta=beta, epsilon=epsilon,
            scale_alg=scale_alg, round_mode=round_mode, dst_type=dst_type
        )

    def supported_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-06,
                          scale_alg=0, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2):
        y_norm, rstd_golden, x_out_golden = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=epsilon)
        if beta is not None:
            y_norm = y_norm + beta
        y_golden, mxscale_golden = torch_npu.npu_dynamic_mx_quant(
            y_norm, axis=-1, round_mode=round_mode,
            dst_type=dst_type, block_size=32, scale_alg=scale_alg
        )
        return y_golden, x_out_golden, mxscale_golden, rstd_golden

    def generate_input(self, shape, dtype="float16"):
        data_type = torch.float16 if dtype == "float16" else torch.bfloat16
        return torch.randn(shape, dtype=data_type)

    @patch("torch_npu.npu_add_rms_norm_dynamic_mx_quant", side_effect=_mock_npu_add_rms_norm_dynamic_mx_quant)
    @patch("torch_npu.npu_dynamic_mx_quant", side_effect=_mock_npu_dynamic_mx_quant)
    @patch("torch_npu.npu_add_rms_norm", side_effect=_mock_npu_add_rms_norm)
    def test_npu_add_rms_norm_dynamic_mx_quant_fp8_e5m2(self, mock_arm, mock_dmq, mock_fused):
        shape = [2, 64]
        x1 = self.generate_input(shape, "float16")
        x2 = self.generate_input(shape, "float16")
        gamma = torch.ones(64, dtype=torch.float16)

        golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), dst_type=torch_npu.float8_e5m2
        )
        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), dst_type=torch_npu.float8_e5m2
        )

        # 验证 shape/dtype 一致
        self.assertEqual(custom_y.shape, golden_y.shape)
        self.assertEqual(custom_x_out.shape, golden_x_out.shape)
        self.assertEqual(custom_mxscale.shape, golden_mxscale.shape)
        self.assertEqual(custom_rstd.shape, golden_rstd.shape)
        self.assertEqual(custom_y.dtype, golden_y.dtype)
        self.assertEqual(custom_x_out.dtype, golden_x_out.dtype)

        # 验证 mock 被正确调用
        mock_fused.assert_called_once()
        mock_arm.assert_called_once()
        mock_dmq.assert_called_once()

    @patch("torch_npu.npu_add_rms_norm_dynamic_mx_quant", side_effect=_mock_npu_add_rms_norm_dynamic_mx_quant)
    @patch("torch_npu.npu_dynamic_mx_quant", side_effect=_mock_npu_dynamic_mx_quant)
    @patch("torch_npu.npu_add_rms_norm", side_effect=_mock_npu_add_rms_norm)
    def test_npu_add_rms_norm_dynamic_mx_quant_bf16(self, mock_arm, mock_dmq, mock_fused):
        shape = [4, 8, 128]
        x1 = self.generate_input(shape, "bfloat16")
        x2 = self.generate_input(shape, "bfloat16")
        gamma = torch.ones(128, dtype=torch.bfloat16)
        beta = torch.zeros(128, dtype=torch.bfloat16)

        golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), beta=beta.clone(), dst_type=torch_npu.float8_e5m2
        )
        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), beta=beta.clone(), dst_type=torch_npu.float8_e5m2
        )

        self.assertEqual(custom_y.shape, golden_y.shape)
        self.assertEqual(custom_x_out.shape, golden_x_out.shape)
        self.assertEqual(custom_mxscale.shape, golden_mxscale.shape)
        self.assertEqual(custom_rstd.shape, golden_rstd.shape)

        mock_fused.assert_called_once()
        mock_arm.assert_called_once()
        mock_dmq.assert_called_once()

    @patch("torch_npu.npu_add_rms_norm_dynamic_mx_quant", side_effect=_mock_npu_add_rms_norm_dynamic_mx_quant)
    @patch("torch_npu.npu_dynamic_mx_quant", side_effect=_mock_npu_dynamic_mx_quant)
    @patch("torch_npu.npu_add_rms_norm", side_effect=_mock_npu_add_rms_norm)
    def test_npu_add_rms_norm_dynamic_mx_quant_fp4(self, mock_arm, mock_dmq, mock_fused):
        shape = [2, 64]
        x1 = self.generate_input(shape, "float16")
        x2 = self.generate_input(shape, "float16")
        gamma = torch.ones(64, dtype=torch.float16)

        golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), dst_type=torch_npu.float4_e2m1fn_x2
        )
        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), dst_type=torch_npu.float4_e2m1fn_x2
        )

        # FP4: y 应为 uint8，尾轴减半
        self.assertEqual(custom_y.shape, golden_y.shape)
        self.assertEqual(custom_y.shape[-1], shape[-1] // 2)
        self.assertEqual(custom_y.dtype, torch.uint8)
        self.assertEqual(custom_x_out.shape, golden_x_out.shape)
        self.assertEqual(custom_mxscale.shape, golden_mxscale.shape)

        mock_fused.assert_called_once()
        mock_arm.assert_called_once()
        mock_dmq.assert_called_once()


if __name__ == "__main__":
    run_tests()
