import math
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


class TestNpuAddRmsNormDynamicMxQuant(TestCase):
    def custom_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2):
        return torch_npu.npu_add_rms_norm_dynamic_mx_quant(
            x1, x2, gamma, beta=beta, epsilon=epsilon, scale_alg=scale_alg,
            round_mode=round_mode, dst_type=dst_type
        )

    def supported_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2):
        # 使用子算子拼接作为 golden
        # Step 1: Add + RmsNorm
        y_norm, rstd, x_out = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=epsilon)

        # Step 2: DynamicMxQuant
        # 注意：npu_dynamic_mx_quant 需要确定 axis 和 block_size
        axis = -1  # 最后一维
        block_size = 32
        y_quant, mxscale = torch_npu.npu_dynamic_mx_quant(
            y_norm, axis=axis, round_mode=round_mode, dst_type=dst_type,
            block_size=block_size, scale_alg=scale_alg
        )

        return y_quant, x_out, mxscale, rstd

    def generate_input(self, shape, dtype=torch.float16):
        return torch.randn(shape, dtype=dtype)

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_dynamic_mx_quant_fp16_fp4(self, device="npu"):
        """测试 FP16 输入，FP4 输出"""
        shape = (2, 4, 64)
        x1 = self.generate_input(shape, torch.float16).to(device)
        x2 = self.generate_input(shape, torch.float16).to(device)
        gamma = self.generate_input((shape[-1],), torch.float16).to(device)

        golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
            x1.clone(), x2.clone(), gamma.clone(),
            dst_type=torch_npu.float4_e2m1fn_x2
        )
        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(),
            dst_type=torch_npu.float4_e2m1fn_x2
        )

        # x_out 对比（非量化输出）
        assert_close(custom_x_out.float(), golden_x_out.float(), atol=1e-3, rtol=1e-3)
        # y 对比（uint8 视图）
        assert torch.all(custom_y.view(torch.uint8) == golden_y.view(torch.uint8))
        # mxscale 对比
        assert_close(custom_mxscale.float(), golden_mxscale.float(), atol=0.01, rtol=0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_dynamic_mx_quant_bf16_fp8(self, device="npu"):
        """测试 BF16 输入，FP8 输出"""
        shape = (2, 4, 64)
        x1 = self.generate_input(shape, torch.bfloat16).to(device)
        x2 = self.generate_input(shape, torch.bfloat16).to(device)
        gamma = self.generate_input((shape[-1],), torch.bfloat16).to(device)

        golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
            x1.clone(), x2.clone(), gamma.clone(),
            dst_type=torch_npu.float8_e5m2
        )
        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(),
            dst_type=torch_npu.float8_e5m2
        )

        # x_out 对比（非量化输出）
        assert_close(custom_x_out.float(), golden_x_out.float(), atol=1e-3, rtol=1e-3)
        # y 对比（uint8 视图）
        assert torch.all(custom_y.view(torch.uint8) == golden_y.view(torch.uint8))
        # mxscale 对比
        assert_close(custom_mxscale.float(), golden_mxscale.float(), atol=0.01, rtol=0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_dynamic_mx_quant_with_beta(self, device="npu"):
        """测试带 beta 参数"""
        shape = (2, 4, 64)
        x1 = self.generate_input(shape, torch.float16).to(device)
        x2 = self.generate_input(shape, torch.float16).to(device)
        gamma = self.generate_input((shape[-1],), torch.float16).to(device)
        beta = self.generate_input((shape[-1],), torch.float16).to(device)

        golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), beta=beta.clone(),
            dst_type=torch_npu.float4_e2m1fn_x2
        )
        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), beta=beta.clone(),
            dst_type=torch_npu.float4_e2m1fn_x2
        )

        # x_out 对比（非量化输出）
        assert_close(custom_x_out.float(), golden_x_out.float(), atol=1e-3, rtol=1e-3)
        # y 对比（uint8 视图）
        assert torch.all(custom_y.view(torch.uint8) == golden_y.view(torch.uint8))
        # mxscale 对比
        assert_close(custom_mxscale.float(), golden_mxscale.float(), atol=0.01, rtol=0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_dynamic_mx_quant_different_dst_types(self, device="npu"):
        """测试不同的 dst_type"""
        shape = (2, 4, 64)
        x1 = self.generate_input(shape, torch.float16).to(device)
        x2 = self.generate_input(shape, torch.float16).to(device)
        gamma = self.generate_input((shape[-1],), torch.float16).to(device)

        dst_types = [
            torch_npu.float8_e5m2,
            torch_npu.float8_e4m3fn,
            torch_npu.float4_e2m1fn_x2,
            torch_npu.float4_e1m2fn_x2,
        ]

        for dst_type in dst_types:
            golden_y, golden_x_out, golden_mxscale, golden_rstd = self.supported_op_exec(
                x1.clone(), x2.clone(), gamma.clone(), dst_type=dst_type
            )
            custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
                x1.clone(), x2.clone(), gamma.clone(), dst_type=dst_type
            )

            # x_out 对比（非量化输出）
            assert_close(custom_x_out.float(), golden_x_out.float(), atol=1e-3, rtol=1e-3)
            # y 对比（uint8 视图）
            assert torch.all(custom_y.view(torch.uint8) == golden_y.view(torch.uint8))
            # mxscale 对比
            assert_close(custom_mxscale.float(), golden_mxscale.float(), atol=0.01, rtol=0.001)


if __name__ == "__main__":
    run_tests()
