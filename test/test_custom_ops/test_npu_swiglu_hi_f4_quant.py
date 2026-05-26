import unittest
import math
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

class TestSwigluHiF4Quant(TestCase):
    def npu_op_exec(self, x, group_index=None, activate_dim=-1, activate_left=False, swiglu_mode=0,
                    clamp_limit=7, glu_alpha=1.702, glu_bias=1, axis=-1):
        return torch_npu.npu_swiglu_hi_f4_quant(x, group_index=group_index, activate_dim=activate_dim,
                                                 activate_left=activate_left, swiglu_mode=swiglu_mode,
                                                 clamp_limit=clamp_limit, glu_alpha=glu_alpha,
                                                 glu_bias=glu_bias, axis=axis)

    def generate_input(self, input_shape, value, dtype="bfloat16"):
        if dtype == "float16":
            data_type = torch.float16
        elif dtype == "bfloat16":
            data_type = torch.bfloat16
        input_tensor = torch.full(input_shape, value, dtype=data_type)
        return input_tensor

    @SupportedDevices(['Ascend960'])
    def test_npu_swiglu_hi_f4_quant_basic(self, device="npu"):
        x = self.generate_input(input_shape=[1, 128], value=1.0, dtype="bfloat16")
        x = x.to(device).requires_grad_(False)

        y, scale = self.npu_op_exec(x)

        self.assertEqual(y.shape, torch.Size([1, 32]))
        self.assertEqual(y.dtype, torch.uint8)
        self.assertEqual(scale.shape, torch.Size([1, 1, 1]))
        self.assertEqual(scale.dtype, torch.float32)

    @SupportedDevices(['Ascend960'])
    def test_npu_swiglu_hi_f4_quant_mode1(self, device="npu"):
        x = self.generate_input(input_shape=[2, 128], value=2.0, dtype="bfloat16")
        x = x.to(device)

        y, scale = self.npu_op_exec(x, swiglu_mode=1, activate_left=True)

        self.assertEqual(y.shape, torch.Size([2, 32]))
        self.assertEqual(y.dtype, torch.uint8)
        self.assertEqual(scale.shape, torch.Size([2, 1, 1]))

    @SupportedDevices(['Ascend960'])
    def test_npu_swiglu_hi_f4_quant_with_group_index(self, device="npu"):
        x = self.generate_input(input_shape=[8, 128], value=1.0, dtype="bfloat16")
        x = x.to(device)
        group_index = torch.tensor([4, 4], dtype=torch.int64, device=device)

        y, scale = self.npu_op_exec(x, group_index=group_index)

        self.assertEqual(y.shape[0], 8)
        self.assertEqual(y.shape[-1], 32)


if __name__ == "__main__":
    run_tests()