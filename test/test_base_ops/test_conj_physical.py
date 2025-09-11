import unittest
import random
import torch
import torch_npu
import hypothesis
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestConjPhysical(TestCase):

    torch_dtypes = {
        "float16": torch.float16,
        "float32": torch.float32
    }

    def create_tensors(self, tensor_nums, dtype):
        cpu_tensors = []
        npu_tensors = []
        for i in range(tensor_nums):
            m = random.randint(1, 100)
            n = random.randint(1, 100)
            t = torch.randn((m, n), dtype=self.torch_dtypes.get(dtype))
            cpu_tensors.append(t)
            npu_tensors.append(t.npu())
        return tuple(cpu_tensors), tuple(npu_tensors)

    def test_conj_physical_out_float32_shpae_tensor_num(self):
        tensor_num_list = [3, 4]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float32")
            cpu_output = torch.conj_physical(cpu_tensors)
            npu_output = torch.conj_physical(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_conj_physical_out_float16_shpae_tensor_num(self):
        tensor_num_list = [3, 4]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float16")
            cpu_output = torch.conj_physical(cpu_tensors)
            npu_output = torch.conj_physical(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)

if __name__ == "__main__":
    run_tests()
