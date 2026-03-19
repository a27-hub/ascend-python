import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


class TestNpuAddRmsNormDynamicMxQuant(TestCase):
    def supported_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-06, scale_alg=0, round_mode="rint", dst_type=296):
        """Golden 参考实现 — 需用户提供具体逻辑"""
        # TODO: 用户需补充 golden 计算逻辑
        # 示例框架：
        # 1. add_out = x1 + x2
        # 2. rms_norm_out = rms_norm(add_out, gamma, beta, epsilon)
        # 3. y, mxscale = dynamic_mx_quant(rms_norm_out, ...)
        # 4. rstd = 1 / sqrt(mean(add_out^2) + epsilon)
        raise NotImplementedError("请补充 golden 参考实现")

    def custom_op_exec(self, x1, x2, gamma, beta=None, epsilon=1e-06, scale_alg=0, round_mode="rint", dst_type=296):
        return torch_npu.npu_add_rms_norm_dynamic_mx_quant(
            x1, x2, gamma, beta=beta, epsilon=epsilon,
            scale_alg=scale_alg, round_mode=round_mode, dst_type=dst_type
        )

    def generate_input(self, shape, dtype="float16"):
        data_type = torch.float16 if dtype == "float16" else torch.bfloat16
        return torch.randn(shape, dtype=data_type)

    @SupportedDevices(['Ascend950'])
    def test_npu_add_rms_norm_dynamic_mx_quant_fp16(self, device="npu"):
        shape = (2, 64)
        x1 = self.generate_input(shape, "float16").to(device)
        x2 = self.generate_input(shape, "float16").to(device)
        gamma = torch.ones(64, dtype=torch.float16).to(device)
        y, x_out, mxscale_out, rstd_out = self.custom_op_exec(x1, x2, gamma, dst_type=296)
        # 基本 shape/dtype 校验
        self.assertEqual(y.shape, (2, 32))  # FP4: 尾轴减半
        self.assertEqual(y.dtype, torch.uint8)
        self.assertEqual(x_out.shape, shape)
        self.assertEqual(x_out.dtype, torch.float16)
        self.assertEqual(mxscale_out.dtype, torch.uint8)
        self.assertEqual(rstd_out.dtype, torch.float32)

    @SupportedDevices(['Ascend950'])
    def test_npu_add_rms_norm_dynamic_mx_quant_bf16(self, device="npu"):
        shape = (4, 8, 128)
        x1 = self.generate_input(shape, "bfloat16").to(device)
        x2 = self.generate_input(shape, "bfloat16").to(device)
        gamma = torch.ones(128, dtype=torch.bfloat16).to(device)
        beta = torch.zeros(128, dtype=torch.bfloat16).to(device)
        y, x_out, mxscale_out, rstd_out = self.custom_op_exec(
            x1, x2, gamma, beta=beta, dst_type=296
        )
        self.assertEqual(y.shape, (4, 8, 64))
        self.assertEqual(y.dtype, torch.uint8)
        self.assertEqual(x_out.shape, shape)
        self.assertEqual(x_out.dtype, torch.bfloat16)
        self.assertEqual(mxscale_out.dtype, torch.uint8)
        self.assertEqual(mxscale_out.dim(), 4)  # rank(x1)+1
        self.assertEqual(mxscale_out.shape[-1], 2)

    @SupportedDevices(['Ascend950'])
    def test_npu_add_rms_norm_dynamic_mx_quant_fp8(self, device="npu"):
        shape = (2, 64)
        x1 = self.generate_input(shape, "float16").to(device)
        x2 = self.generate_input(shape, "float16").to(device)
        gamma = torch.ones(64, dtype=torch.float16).to(device)
        y, x_out, mxscale_out, rstd_out = self.custom_op_exec(
            x1, x2, gamma, dst_type=torch_npu.float8_e5m2
        )
        self.assertEqual(y.shape, shape)
        self.assertEqual(y.dtype, torch.float8_e5m2)


if __name__ == "__main__":
    run_tests()
