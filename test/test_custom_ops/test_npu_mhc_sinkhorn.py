from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestMhcSinkhorn(TestCase):
   
    def mhc_sinkhorn_impl(self, h_res, out_flag, eps, num_iters):
        y = torch.softmax(h_res, dim=-1) + eps
        y = y / (y.sum(dim=-2, keepdim=True) + eps)
        for i in range(max(num_iters - 1, 0)):
            # Row Norm
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
            # Col Norm
            y = y / (y.sum(dim=-2, keepdim=True) + eps)
        return y
   
    def custom_op_exec(self, h_res, out_flag, eps, num_iters):
        y, norm_out, sum_out = torch_npu.npu_mhc_sinkhorn(h_res, out_flag, eps, num_iters)
        return y
    
    @SupportedDevices(['Ascend910B'])
    def test_mhc_sinkhorn(self, device="npu"):
        h_res = torch.randn(1, 128, 4, 4, dtype=torch.float32).npu()
        out_flag = 0
        eps = 1e-6
        num_iters = 20

        expected_output = self.mhc_sinkhorn_impl(h_res, out_flag, eps, num_iters)
        custom_output = self.custom_op_exec(h_res, out_flag, eps, num_iters)
        self.assertRtolEqual(expected_output, custom_output)


if __name__ == "__main__":
    run_tests()        