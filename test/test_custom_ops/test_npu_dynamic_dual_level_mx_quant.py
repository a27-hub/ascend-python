import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


class TestDynamicDualLevelMxQuant(TestCase):
    def custom_op_exec(self, input_tensor, smooth_scale=None, round_mode="rint", level0_block_size=512, level1_block_size=32):
        return torch_npu.npu_dynamic_dual_level_mx_quant(input_tensor, smooth_scale=smooth_scale, round_mode=round_mode, 
                level0_block_size=level0_block_size, level1_block_size=level1_block_size)

    def supported_op_exec(self, input_tensor):
        if torch.all(torch.eq(input_tensor, 0.0)) and input_tensor.shape == torch.Size([1, 1, 512]):
            device = input_tensor.device
            y = torch.zeros([1, 1, 256], dtype=torch.uint8, device=device)
            level0_scale = torch.tensor([[[5.87747e-39]]], dtype=torch.float32, device=device)
            level1_scale = torch.tensor([[[[5.87747e-39, 5.87747e-39] * 8]]], dtype=torch.uint8, device=device)
            return y, level0_scale, level1_scale

    def generate_input(self, input, dtype="float16"):
        data_type = torch.float16 if dtype == "float16" else torch.bfloat16
        value = 0.0
        input_tensor = torch.full(input, value, dtype=data_type)
        return input_tensor
    
    @SupportedDevices(['Ascend910_95'])
    def test_npu_dynamic_dual_level_mx_quant(self, device="npu"):
        input_tensor = self.generate_input(input=[1, 1, 512], dtype="bfloat16")
        input_tensor = input_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone())
        custom_output = self.custom_op_exec(input_tensor.clone(), None, "rint", 512, 32)

        y = custom_output[0].view([1, 1, 256]).view(torch.uint8)
        level0_scale = custom_output[1].view([1, 1, 1])
        level1_scale = custom_output[2].view([1, 1, 8, 2]).to(torch.uint8)

        assert torch.all(y == supported_output[0].view(torch.uint8))
        assert_close(supported_output[1], level0_scale, atol=0.01, rtol=0.001)
        assert_close(supported_output[2], level1_scale, atol=0.01, rtol=0.001)
        
if __name__ == "__main__":
    run_tests()