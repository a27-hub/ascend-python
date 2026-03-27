import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


class TestNpuAddRmsNormDynamicMxQuant(TestCase):
    def custom_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-6, scale_alg=0,
                       round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2):
        return torch_npu.npu_add_rms_norm_dynamic_mx_quant(
            x1, x2, gamma, beta=beta, epsilon=epsilon, scale_alg=scale_alg,
            round_mode=round_mode, dst_type=dst_type
        )

    def supported_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-6, dst_type=torch_npu.float4_e2m1fn_x2):
        """使用已有 NPU 算子拼接作为 golden"""
        # Add + RmsNorm
        y_norm, rstd, x_out = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=epsilon)
        # DynamicMxQuant
        if dst_type == torch_npu.float4_e2m1fn_x2 or dst_type == torch_npu.float4_e1m2fn_x2:
            # FP4 不支持 npu_dynamic_mx_quant，使用 FP8 近似
            y_quant, mxscale = torch_npu.npu_dynamic_mx_quant(
                y_norm, axis=-1, round_mode="rint",
                dst_type=torch_npu.float8_e5m2, block_size=32, scale_alg=0
            )
        else:
            y_quant, mxscale = torch_npu.npu_dynamic_mx_quant(
                y_norm, axis=-1, round_mode="rint",
                dst_type=dst_type, block_size=32, scale_alg=0
            )
        return y_quant, x_out, mxscale, rstd

    def generate_input(self, shape, dtype=torch.float16):
        return torch.randn(shape, dtype=dtype)

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_dynamic_mx_quant_fp16_fp4(self, device="npu"):
        """FP16输入，FP4输出"""
        x1 = self.generate_input((8, 64), torch.float16).to(device)
        x2 = self.generate_input((8, 64), torch.float16).to(device)
        gamma = self.generate_input((64,), torch.float16).to(device)

        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(),
            dst_type=torch_npu.float4_e2m1fn_x2
        )

        # 基本检查
        self.assertEqual(custom_y.shape, (8, 32))  # FP4 尾轴减半
        self.assertEqual(custom_x_out.shape, (8, 64))
        self.assertEqual(custom_mxscale.shape, (8, 1, 2))  # (8, ceil(64/32)/2, 2) = (8, 1, 2)
        self.assertEqual(custom_rstd.shape, (8, 1))

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_dynamic_mx_quant_bf16_fp4(self, device="npu"):
        """BF16输入，FP4输出"""
        x1 = self.generate_input((16, 128), torch.bfloat16).to(device)
        x2 = self.generate_input((16, 128), torch.bfloat16).to(device)
        gamma = self.generate_input((128,), torch.bfloat16).to(device)

        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(),
            dst_type=torch_npu.float4_e2m1fn_x2
        )

        self.assertEqual(custom_y.shape, (16, 64))  # FP4 尾轴减半
        self.assertEqual(custom_x_out.shape, (16, 128))
        self.assertEqual(custom_mxscale.shape, (16, 2, 2))  # (16, ceil(128/32)/2, 2) = (16, 2, 2)
        self.assertEqual(custom_rstd.shape, (16, 1))

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_dynamic_mx_quant_fp8(self, device="npu"):
        """FP16输入，FP8输出"""
        x1 = self.generate_input((4, 256), torch.float16).to(device)
        x2 = self.generate_input((4, 256), torch.float16).to(device)
        gamma = self.generate_input((256,), torch.float16).to(device)

        # FP8_E5M2
        custom_y1, custom_x_out1, custom_mxscale1, _ = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(),
            dst_type=torch_npu.float8_e5m2
        )
        self.assertEqual(custom_y1.shape, (4, 256))  # FP8 shape 不变
        self.assertEqual(custom_y1.dtype, torch.float8_e5m2)

        # FP8_E4M3FN
        custom_y2, custom_x_out2, custom_mxscale2, _ = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(),
            dst_type=torch_npu.float8_e4m3fn
        )
        self.assertEqual(custom_y2.shape, (4, 256))
        self.assertEqual(custom_y2.dtype, torch.float8_e4m3fn)

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_dynamic_mx_quant_3d(self, device="npu"):
        """3D输入"""
        x1 = self.generate_input((2, 8, 64), torch.float16).to(device)
        x2 = self.generate_input((2, 8, 64), torch.float16).to(device)
        gamma = self.generate_input((64,), torch.float16).to(device)

        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(),
            dst_type=torch_npu.float4_e2m1fn_x2
        )

        self.assertEqual(custom_y.shape, (2, 8, 32))
        self.assertEqual(custom_x_out.shape, (2, 8, 64))
        self.assertEqual(custom_mxscale.shape, (2, 8, 1, 2))
        self.assertEqual(custom_rstd.shape, (2, 8, 1))

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_dynamic_mx_quant_with_beta(self, device="npu"):
        """带 beta 参数"""
        x1 = self.generate_input((8, 64), torch.float16).to(device)
        x2 = self.generate_input((8, 64), torch.float16).to(device)
        gamma = self.generate_input((64,), torch.float16).to(device)
        beta = self.generate_input((64,), torch.float16).to(device)

        custom_y, custom_x_out, custom_mxscale, custom_rstd = self.custom_op_exec(
            x1.clone(), x2.clone(), gamma.clone(), beta=beta.clone(),
            dst_type=torch_npu.float4_e2m1fn_x2
        )

        self.assertEqual(custom_y.shape, (8, 32))
        self.assertEqual(custom_x_out.shape, (8, 64))


if __name__ == "__main__":
    run_tests()
