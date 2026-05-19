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


class TestAntiMxQuant(TestCase):
    def custom_op_exec(self, input_tensor, mxscale_tensor, axis=-1, dst_type=15):
        return torch_npu.npu_anti_mx_quant(
            input_tensor, mxscale_tensor, axis=-1, dst_type=dst_type)

    def supported_op_exec(self, input_tensor, mxscale_tensor):
        if (torch.all(torch.eq(input_tensor, 0.0)) and input_tensor.shape == torch.Size([1, 2])
            and torch.all(torch.eq(mxscale_tensor, 0)) and mxscale_tensor.shape == torch.Size([1, 1, 2])):
            device = input_tensor.device
            y = torch.tensor([[0, 0]], dtype=torch.bfloat16, device=device)
            return y

    def generate_input(self, input, mxscale, dtype="float8_e5m2"):
        if dtype == "float8_e5m2":
            input_data_type = torch.float8_e5m2
        elif dtype == "float8_e4m3fn":
            input_data_type = torch.float8_e4m3fn
        elif dtype == "float4_e1m2":
            input_data_type = torch.float4_e1m2
        elif dtype == "float4_e2m1":
            input_data_type = torch.float4_e2m1
        input_value = 0.0
        input_tensor = torch.full(input, input_value, dtype=input_data_type)
        mxscale_data_type = torch_npu.float8_e8m0
        mxscale_value = 0
        mxscale_tensor = torch.full(mxscale, mxscale_value, dtype=mxscale_data_type)
        return input_tensor, mxscale_tensor
    
    @SupportedDevices(['Ascend950'])
    def test_npu_anti_mx_quant_(self, device="npu"):
        input_tensor, mxscale_tensor = self.generate_input(input=[1, 2], mxscale=[1, 1, 2], dtype="float8_e5m2")
        input_tensor = input_tensor.to(device)
        mxscale_tensor = mxscale_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone(), mxscale_tensor.clone())
        custom_output = self.custom_op_exec(input_tensor.clone(), mxscale_tensor.clone(), -1, 15)
        y = custom_output[0].view([1, 2]).view(torch.bfloat16)

        assert torch.all(y == supported_output[0].view(torch.bfloat16))
        
if __name__ == "__main__":
    run_tests()