import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


class TestAddRmsNormDynamicMxQuant(TestCase):
    def custom_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-06,
                       scale_alg=0, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2):
        return torch_npu.npu_add_rms_norm_dynamic_mx_quant(
            x1, x2, gamma, beta=beta, epsilon=epsilon,
            scale_alg=scale_alg, round_mode=round_mode, dst_type=dst_type
        )

    def supported_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-06,
                          scale_alg=0, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2):
        # 拼接已有 NPU 算子作为 golden：
        # Step 1: npu_add_rms_norm -> (y_norm, rstd, x_out)
        y_norm, rstd_golden, x_out_golden = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=epsilon)

        # Step 1.5: beta 偏置（npu_add_rms_norm 不支持 beta，需手动加）
        if beta is not None:
            y_norm = y_norm + beta

        # Step 2: npu_dynamic_mx_quant -> (y_quant, mxscale)
        y_golden, mxscale_golden = torch_npu.npu_dynamic_mx_quant(
            y_norm, axis=-1, round_mode=round_mode,
            dst_type=dst_type, block_size=32, scale_alg=scale_alg
        )

        return y_golden, x_out_golden, mxscale_golden, rstd_golden

    def generate_input(self, shape, dtype="float16"):
        data_type = torch.float16 if dtype == "float16" else torch.bfloat16
        return torch.randn(shape, dtype=data_type)

    @SupportedDevices(['Ascend950'])
    def test_npu_add_rms_norm_dynamic_mx_quant_fp8_e5m2(self, device="npu"):
        shape = [2, 64]
        x1 = self.generate_input(shape, "float16").to(device)
        x2 = self.generate_input(shape, "float16").to(device)
        gamma = torch.ones(64, dtype=torch.float16, device=device)

        golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), dst_type=torch_npu.float8_e5m2
        )
        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), dst_type=torch_npu.float8_e5m2
        )

        # x_out 对比 (Add 结果)
        assert_close(custom_x_out.float(), golden_x_out.float(), atol=1e-3, rtol=1e-3)
        # y 对比 (量化结果, 用 uint8 视图)
        assert torch.all(custom_y.view(torch.uint8) == golden_y.view(torch.uint8))
        # mxscale 对比
        assert_close(custom_mxscale, golden_mxscale, atol=0.01, rtol=0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_add_rms_norm_dynamic_mx_quant_bf16(self, device="npu"):
        shape = [4, 8, 128]
        x1 = self.generate_input(shape, "bfloat16").to(device)
        x2 = self.generate_input(shape, "bfloat16").to(device)
        gamma = torch.ones(128, dtype=torch.bfloat16, device=device)
        beta = torch.zeros(128, dtype=torch.bfloat16, device=device)

        golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), beta=beta.clone(), dst_type=torch_npu.float8_e5m2
        )
        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), beta=beta.clone(), dst_type=torch_npu.float8_e5m2
        )

        assert_close(custom_x_out.float(), golden_x_out.float(), atol=1e-3, rtol=1e-3)
        assert torch.all(custom_y.view(torch.uint8) == golden_y.view(torch.uint8))
        assert_close(custom_mxscale, golden_mxscale, atol=0.01, rtol=0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_add_rms_norm_dynamic_mx_quant_fp4(self, device="npu"):
        shape = [2, 64]
        x1 = self.generate_input(shape, "float16").to(device)
        x2 = self.generate_input(shape, "float16").to(device)
        gamma = torch.ones(64, dtype=torch.float16, device=device)

        golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), dst_type=torch_npu.float4_e2m1fn_x2
        )
        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), dst_type=torch_npu.float4_e2m1fn_x2
        )

        assert_close(custom_x_out.float(), golden_x_out.float(), atol=1e-3, rtol=1e-3)
        assert torch.all(custom_y == golden_y)
        assert_close(custom_mxscale, golden_mxscale, atol=0.01, rtol=0.001)


if __name__ == "__main__":
    run_tests()
