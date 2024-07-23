import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestSilentCheck(TestCase):
    @SupportedDevices(['Ascend910B'])
    def test_npu_silent_check(self, device="npu"):
        input_grad = torch.rand((3, 5, 10), dtype=torch.half).npu()
        val = torch.Tensor((160)).float().npu()
        pre_val = torch.Tensor((70)).float().npu()
        min_val = torch.Tensor((200)).float().npu()
        max_val = torch.Tensor((400)).float().npu()
        val_counter = torch.Tensor((7)).float().npu()

        c_min_steps = 2
        c_thresh_l1 = 300
        c_coeff_l1 = 1
        c_thresh_l2 = 0.8
        c_coeff_l2 = 0.5

        supported_output = torch.Tensor((2)).int().type(torch.int32).npu()
        custom_output = torch_npu._npu_silent_check(input_grad, val, pre_val, min_val, max_val, val_counter, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2)
        self.assertRtolEqual(supported_output, custom_output)

if __name__ == "__main__":
    run_tests()