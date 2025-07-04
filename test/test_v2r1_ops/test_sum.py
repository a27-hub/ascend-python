import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestMaxPool2d(TestCase):

    def test_sum_dim_none(self):
        data = torch.tensor(-6.6523, dtype=torch.float16)
        data_npu = data.npu()
        dim = None
        keepdim = False
        cpu_output = data.sum(dim=dim, keepdim=keepdim)
        npu_output = data_npu.sum(dim=dim, keepdim=keepdim)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_sum_out(self):
        inputs = torch.randn([3, 3])
        inputs_npu = inputs.npu()
        out_tensor = torch.zeros_like(inputs)
        out_tensor_npu = out_tensor.npu()
        cpu_out = torch.sum(inputs, dim=0, keepdim=True, out=out_tensor)
        npu_out = torch.sum(inputs_npu, dim=0, keepdim=True, out=out_tensor_npu)
        self.assertRtolEqual(cpu_out, npu_out)
    
    def test_sum_out_diff_size(self):
        inputs = torch.randn([3, 3])
        inputs_npu = inputs.npu()
        out_tensor = torch.zeros([3, 1])
        out_tensor_npu = out_tensor.npu()
        cpu_out = torch.sum(inputs, dim=0, keepdim=True, out=out_tensor)
        npu_out = torch.sum(inputs_npu, dim=0, keepdim=True, out=out_tensor_npu)
        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
