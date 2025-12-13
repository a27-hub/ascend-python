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


class TestDynamicMxQuantWithDualAxis(TestCase):
    def custom_op_exec(self, input_tensor, round_mode="rint", dst_type=296, scale_alg=0):
        return torch_npu.npu_dynamic_mx_quant_with_dual_axis(input_tensor, round_mode=round_mode, dst_type=dst_type, scale_alg=scale_alg)

    def supported_op_exec(self, input_tensor):
        if torch.all(torch.eq(input_tensor, 1.0)) and input_tensor.shape == torch.Size([1, 1, 2]):
            device = input_tensor.device
            dtype = input_tensor.dtype

            y1 = torch.tensor([32768, 32768, 32768, 32768], dtype=dtype, device=device)
            mxscale1 = torch.tensor([3.05176e-05, 5.87747e-39, 3.05176e-05, 5.87747e-39], dtype=dtype, device=device)
            y2 = torch.tensor([32768, 32768, 32768, 32768], dtype=dtype, device=device)
            mxscale2 = torch.tensor([3.05176e-05, 5.87747e-39, 3.05176e-05, 5.87747e-39], dtype=dtype, device=device)
            return y1, mxscale1, y2, mxscale2

    def generate_input(self, input, dtype="float16"):
        data_type = torch.float16 if dtype == "float16" else torch.bfloat16
        value = 1.0
        input_tensor = torch.full(input, value, dtype=data_type)
        return input_tensor
    
    @SupportedDevices(['Ascend910_95'])
    def test_npu_dynamic_mx_quant_with_dual_axis(self, device="npu"):
        input_tensor = self.generate_input(input=[1, 1, 2], dtype="bfloat16")
        input_tensor = input_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone())
        custom_output = self.custom_op_exec(input_tensor.clone(), "rint", 23, 0)
        y1 = custom_output[0].view([1, 1, 2])
        mxscale1 = custom_output[1].view([1, 1, 1, 2])
        y2 = custom_output[2].view([1, 1, 2])
        mxscale2 = custom_output[3].view([1, 1, 2, 2])
        assert_close(supported_output[0], y1, atol=0.01, rtol=0.001)
        assert_close(supported_output[1], mxscale1, atol=0.01, rtol=0.001)
        assert_close(supported_output[2], y2, atol=0.01, rtol=0.001)
        assert_close(supported_output[3], mxscale2, atol=0.01, rtol=0.001)
        
if __name__ == "__main__":
    run_tests()