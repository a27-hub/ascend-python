import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIsinfV2(TestCase):

    torch_dtypes = {
        "float16" : torch.float16,
        "float32" : torch.float32,
        "bfloat16" : torch.bfloat16,
        "int32" : torch.int32,
    }

    def create_tensors(self, tensor_shape, dtype):
        data_count = np.prod(tensor_shape)
        tmp_ininf_array = np.array([np.inf, np.nan])
        if (dtype == "int32"):
            tmp_norm_array = np.random.randint(-100, 100, 10)
        else:
            tmp_norm_array = np.random.randn(10)
        conc_array = np.concatenate((tmp_ininf_array, tmp_norm_array))
        input_array = np.random.choice(conc_array, size=data_count)
        cpu_tensor = torch.tensor(input_array, dtype=self.torch_dtypes.get(dtype)).reshape(tensor_shape)
        npu_tensor = cpu_tensor.npu()
        return cpu_tensor, npu_tensor


    def cpu_op_exec(self, input1):
        output = torch.isinf(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.isinf(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @SupportedDevices(['Ascend910B'])
    def test_is_inf_v2_float16_tensor(self):
        tensor_shape = [256, 1024]
        cpu_tensor, npu_tensor = self.create_tensors(tensor_shape, "float16")
        cpu_output = self.cpu_op_exec(cpu_tensor)
        npu_output = self.npu_op_exec(npu_tensor)

        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_is_inf_v2_float32_tensor(self):
        tensor_shape = [512, 1111]
        cpu_tensor, npu_tensor = self.create_tensors(tensor_shape, "float32")
        cpu_output = self.cpu_op_exec(cpu_tensor)
        npu_output = self.npu_op_exec(npu_tensor)

        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_is_inf_v2_bfloat16_tensor(self):
        tensor_shape = [256, 1024, 3]
        cpu_tensor, npu_tensor = self.create_tensors(tensor_shape, "bfloat16")
        cpu_output = self.cpu_op_exec(cpu_tensor)
        npu_output = self.npu_op_exec(npu_tensor)

        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_is_inf_v2_int32_tensor(self):
        tensor_shape = [512, 1024]
        cpu_tensor, npu_tensor = self.create_tensors(tensor_shape, "int32")
        cpu_output = self.cpu_op_exec(cpu_tensor)
        npu_output = self.npu_op_exec(npu_tensor)

        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
