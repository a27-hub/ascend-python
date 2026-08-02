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
    def custom_op_exec(self, input_tensor, round_mode="rint", dst_type=torch.float8_e5m2, scale_alg=0, dst_type_max=0.0):
        return torch_npu.npu_dynamic_mx_quant_with_dual_axis(input_tensor, round_mode=round_mode, dst_type=dst_type, scale_alg=scale_alg, dst_type_max=dst_type_max)

    def supported_op_exec(self, input_tensor):
        if torch.all(torch.eq(input_tensor, 0.0)) and input_tensor.shape == torch.Size([1, 2, 2]):
            device = input_tensor.device
            y1 = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.float8_e5m2, device=device)
            mxscale1 = torch.tensor([[[[5.87747e-39, 5.87747e-39]], [[5.87747e-39, 5.87747e-39]]]], dtype=torch.uint8, device=device)
            y2 = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.float8_e5m2, device=device)
            mxscale2 = torch.tensor([[[[5.87747e-39, 5.87747e-39], [5.87747e-39, 5.87747e-39]]]], dtype=torch.uint8, device=device)
            return y1, mxscale1, y2, mxscale2

    def generate_input(self, input, dtype="float16"):
        data_type = torch.float16 if dtype == "float16" else torch.bfloat16
        value = 0.0
        input_tensor = torch.full(input, value, dtype=data_type)
        return input_tensor

    def generate_out_tensors(self, input_tensor):
        y1_out = torch.empty_like(input_tensor, dtype=torch.float8_e5m2)
        y2_out = torch.empty_like(input_tensor, dtype=torch.float8_e5m2)
        mxscale1_shape = list(input_tensor.shape) + [2]
        mxscale2_shape = list(input_tensor.shape) + [2]
        mxscale1_shape[-2] = (math.ceil(mxscale1_shape[-2] / 32) + 1) // 2
        mxscale2_shape[-3] = (math.ceil(mxscale2_shape[-3] / 32) + 1) // 2
        mxscale1_out = torch.empty(mxscale1_shape, dtype=torch.uint8, device=input_tensor.device)
        mxscale2_out = torch.empty(mxscale2_shape, dtype=torch.uint8, device=input_tensor.device)
        return y1_out, mxscale1_out, y2_out, mxscale2_out

    @SupportedDevices(['Ascend950'])
    def test_npu_dynamic_mx_quant_with_dual_axis(self, device="npu"):
        input_tensor = self.generate_input(input=[1, 2, 2], dtype="bfloat16")
        input_tensor = input_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone())
        custom_output = self.custom_op_exec(input_tensor.clone(), "rint", 23, 0, 0.0)
        y1 = custom_output[0].view([1, 2, 2]).view(torch.uint8)
        mxscale1 = custom_output[1].view([1, 2, 1, 2])
        mxscale1_uint8 = mxscale1.to(torch.uint8)
        y2 = custom_output[2].view([1, 2, 2]).view(torch.uint8)
        mxscale2 = custom_output[3].view([1, 1, 2, 2])
        mxscale2_uint8 = mxscale2.to(torch.uint8)

        assert torch.all(y1 == supported_output[0].view(torch.uint8))
        assert torch.all(y2 == supported_output[2].view(torch.uint8))
        assert_close(supported_output[1], mxscale1_uint8, atol=0.01, rtol=0.001)
        assert_close(supported_output[3], mxscale2_uint8, atol=0.01, rtol=0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_dynamic_mx_quant_with_dual_axis_out(self, device="npu"):
        input_tensor = torch.randn([1, 64, 64], dtype=torch.bfloat16, device=device)
        input_before = input_tensor.clone()
        expected = self.custom_op_exec(input_tensor, "rint", 23, 0, 0.0)
        _, mxscale1_out, _, mxscale2_out = self.generate_out_tensors(input_tensor)
        payload_storage = torch.empty_like(input_tensor)
        payload_bytes = payload_storage.view(torch.uint8).reshape(-1)
        payload_numel = input_tensor.numel()
        y1_out = payload_bytes[:payload_numel].reshape(input_tensor.shape).view(torch.float8_e5m2)
        y2_out = payload_bytes[payload_numel:].reshape(input_tensor.shape).view(torch.float8_e5m2)
        out_tensors = (y1_out, mxscale1_out, y2_out, mxscale2_out)

        actual = torch_npu.npu_dynamic_mx_quant_with_dual_axis_out(
            input_tensor, *out_tensors, round_mode="rint", dst_type=23, scale_alg=0, dst_type_max=0.0)

        for returned_tensor, supplied_tensor in zip(actual, out_tensors):
            self.assertEqual(returned_tensor.data_ptr(), supplied_tensor.data_ptr())
        self.assertEqual(y1_out.untyped_storage().data_ptr(), y2_out.untyped_storage().data_ptr())
        self.assertEqual(input_tensor, input_before)
        self.assertEqual(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
        self.assertEqual(actual[1], expected[1].view(torch.uint8))
        self.assertEqual(actual[2].view(torch.uint8), expected[2].view(torch.uint8))
        self.assertEqual(actual[3], expected[3].view(torch.uint8))

    @SupportedDevices(['Ascend950'])
    def test_npu_dynamic_mx_quant_with_dual_axis_out_rejects_overlap(self, device="npu"):
        input_tensor = torch.randn([1, 64, 64], dtype=torch.bfloat16, device=device)
        y1_out, mxscale1_out, y2_out, mxscale2_out = self.generate_out_tensors(input_tensor)
        input_bytes = input_tensor.view(torch.uint8).reshape(-1)
        y1_overlap = input_bytes[:input_tensor.numel()].reshape(input_tensor.shape).view(torch.float8_e5m2)

        with self.assertRaisesRegex(RuntimeError, "input and y1_out must not overlap"):
            torch_npu.npu_dynamic_mx_quant_with_dual_axis_out(
                input_tensor, y1_overlap, mxscale1_out, y2_out, mxscale2_out, dst_type=23)

        with self.assertRaisesRegex(RuntimeError, "y1_out and y2_out must not overlap"):
            torch_npu.npu_dynamic_mx_quant_with_dual_axis_out(
                input_tensor, y1_out, mxscale1_out, y1_out, mxscale2_out, dst_type=23)

    @SupportedDevices(['Ascend950'])
    def test_npu_dynamic_mx_quant_with_dual_axis_out_validates_outputs(self, device="npu"):
        input_tensor = torch.randn([1, 64, 64], dtype=torch.bfloat16, device=device)
        y1_out, mxscale1_out, y2_out, mxscale2_out = self.generate_out_tensors(input_tensor)
        wrong_y1_out = torch.empty([1, 64, 63], dtype=torch.float8_e5m2, device=device)

        with self.assertRaisesRegex(RuntimeError, "y1_out and y2_out have an invalid shape"):
            torch_npu.npu_dynamic_mx_quant_with_dual_axis_out(
                input_tensor, wrong_y1_out, mxscale1_out, y2_out, mxscale2_out, dst_type=23)

        with self.assertRaisesRegex(RuntimeError, "mxscale1_out and mxscale2_out must use uint8 storage"):
            torch_npu.npu_dynamic_mx_quant_with_dual_axis_out(
                input_tensor, y1_out, mxscale1_out.to(torch.int16), y2_out, mxscale2_out, dst_type=23)

if __name__ == "__main__":
    run_tests()
