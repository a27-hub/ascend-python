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


std::tuple<at::Tensor, at::Tensor> npu_grouped_dynamic_block_quant(
    const at::Tensor& x,
    const at::Tensor& group_list,
    double min_scale,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t row_block_size,
    int64_t col_block_size,
    int64_t group_list_type)

class TestGroupedDynamicBlockQuant(TestCase):
    def custom_op_exec(self, input_tensor, group_list, min_scale=0.0, round_mode="rint", dst_type=torch.float8_e5m2, row_block_size=1, col_block_size=128, group_list_type=0):
        return torch_npu.npu_grouped_dynamic_block_quant(input_tensor, 
                                                         group_list, 
                                                         min_scale=min_scale, 
                                                         round_mode=round_mode, 
                                                         dst_type=dst_type, 
                                                         row_block_size=row_block_size, 
                                                         col_block_size=col_block_size,
                                                         group_list_type=group_list_type)

    def supported_op_exec(self, input_tensor):
        if torch.all(torch.eq(input_tensor, 0.0)) and input_tensor.shape == torch.Size([1, 2]):
            device = input_tensor.device
            y = torch.tensor([[0, 0]], dtype=torch.float8_e5m2, device=device)
            scale = torch.tensor([[0.0], [1.0]], dtype=torch.uint8, device=device)
    
            return y, scale

    def generate_input(self, input, input_dtype="float16", group_list):
        input_data_type = torch.float16 if input_dtype == "float16" else torch.bfloat16
        input_value = 0.0
        input_tensor = torch.full(input, input_value, input_dtype=input_data_type)
        group_list_data_type = torch.int32
        group_list_value = 1
        group_list_tensor = torch.full(input, group_list_value, input_dtype=input_data_type)

        return input_tensor, group_list_tensor
    
    @SupportedDevices(['Ascend910_95'])
    def test_npu_grouped_dynamic_block_quant(self, device="npu"):
        input_tensor, group_list_tensor = self.generate_input(input=[1, 2], input_dtype="float16", group_list=[1])
        input_tensor = input_tensor.to(device)
        group_list_tensor = group_list_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone())
        custom_output = self.custom_op_exec(input_tensor.clone(), group_list_tensor.clone(), 0.0, "rint", 35, 1, 128, 0)
        y = custom_output[0].view([1, 2]).view(torch.uint8)
        scale = custom_output[1].view([1])
        scale_uint8 = scale.to(torch.uint8)

        assert torch.all(y == supported_output[0].view(torch.uint8))
        assert_close(supported_output[1], scale_uint8, atol=0.01, rtol=0.001)
        
if __name__ == "__main__":
    run_tests()