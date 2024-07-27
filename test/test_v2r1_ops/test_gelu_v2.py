import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGelu(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1, mode):
        output = torch.nn.functional.gelu(input1, approximate=mode)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, mode):
        input1_npu = input1.to('npu')
        output = torch.nn.functional.gelu(input1_npu, approximate=mode)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_gelu_float32_1(self):
        approximate = "tanh"
        input1 = self.generate_data(0, 100, (4, 3), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1, approximate)
        npu_output = self.npu_op_exec(input1, approximate)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_float32_2(self):
        approximate = "none"
        input1 = self.generate_data(0, 1000, (4, 3), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1, approximate)
        npu_output = self.npu_op_exec(input1, approximate)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
