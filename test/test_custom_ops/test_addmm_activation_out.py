import copy
import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAddmmActivationOut(TestCase):
    def generate_scalar(self, dtype, min_x, max_x):
        if dtype == "float32" or "float16":
            scalar = np.random.uniform(min_x, max_x)
        if dtype == "int32":
            scalar = np.random.randint(min_x, max_x)
        return scalar

    def cpu_reference_out(self, input1, input2, input3, beta, alpha, use_gelu, out):
        # perform addmm into out on CPU, then apply activation in-place
        torch._addmm_activation(input1, input2, input3, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out)
        return out.numpy()

    def npu_op_exec_out(self, input1, input2, input3, beta, alpha, use_gelu, out):
        input1 = input1.to('npu')
        input2 = input2.to('npu')
        input3 = input3.to('npu')
        out_npu = out.to('npu')
        torch._addmm_activation(input1, input2, input3, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out_npu)
        out_res = out_npu.to('cpu').numpy()
        return out_res

    def test_addmm_activation_out_fp32(self):
        # shapes: use small sizes for coverage
        shape_a = (3, 3)
        shape_b = (3, 4)
        shape_c = (4, 3)

        fmt = [np.float32, 0, shape_a]
        cpu_input1, npu_input1 = create_common_tensor(fmt, 0, 1)
        cpu_input2, npu_input2 = create_common_tensor([np.float32, 0, shape_b], 0, 1)
        cpu_input3, npu_input3 = create_common_tensor([np.float32, 0, shape_c], 0, 1)
        cpu_out, npu_out = create_common_tensor([np.float32, 0, shape_a], 0, 1)

        beta = self.generate_scalar('float32', 0, 2)
        alpha = self.generate_scalar('float32', 0, 2)

        # test with relu
        cpu_out_copy = copy.deepcopy(cpu_out)
        cpu_ref = self.cpu_reference_out(cpu_input1, cpu_input2, cpu_input3, beta, alpha, False, cpu_out_copy)
        npu_res = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3, beta, alpha, False, npu_out)
        self.assertRtolEqual(cpu_ref, npu_res)

        # test with gelu
        cpu_out_copy = copy.deepcopy(cpu_out)
        cpu_ref = self.cpu_reference_out(cpu_input1, cpu_input2, cpu_input3, beta, alpha, True, cpu_out_copy)
        npu_res = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3, beta, alpha, True, npu_out)
        self.assertRtolEqual(cpu_ref, npu_res)

    def test_addmm_activation_out_fp16(self):
        shape_a = (3, 3)
        shape_b = (3, 4)
        shape_c = (4, 3)

        cpu_input1, npu_input1 = create_common_tensor([np.float16, 0, shape_a], 0, 2)
        cpu_input2, npu_input2 = create_common_tensor([np.float16, 0, shape_b], 0, 2)
        cpu_input3, npu_input3 = create_common_tensor([np.float16, 0, shape_c], 0, 2)
        cpu_out, npu_out = create_common_tensor([np.float16, 0, shape_a], 0, 2)

        # CPU compute in float32 then cast back to float16 to match NPU behaviour
        cpu_input1 = cpu_input1.to(torch.float32)
        cpu_input2 = cpu_input2.to(torch.float32)
        cpu_input3 = cpu_input3.to(torch.float32)
        cpu_out = cpu_out.to(torch.float32)

        beta = self.generate_scalar('float16', 0, 10)
        alpha = self.generate_scalar('float16', 0, 10)

        # relu path
        cpu_out_copy = copy.deepcopy(cpu_out)
        cpu_ref = self.cpu_reference_out(cpu_input1, cpu_input2, cpu_input3, beta, alpha, False, cpu_out_copy)
        cpu_ref = cpu_ref.astype(np.float16)
        npu_res = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3, beta, alpha, False, npu_out)
        self.assertRtolEqual(cpu_ref, npu_res)

        # gelu path
        cpu_out_copy = copy.deepcopy(cpu_out)
        cpu_ref = self.cpu_reference_out(cpu_input1, cpu_input2, cpu_input3, beta, alpha, True, cpu_out_copy)
        cpu_ref = cpu_ref.astype(np.float16)
        npu_res = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3, beta, alpha, True, npu_out)
        self.assertRtolEqual(cpu_ref, npu_res)


if __name__ == '__main__':
    run_tests()