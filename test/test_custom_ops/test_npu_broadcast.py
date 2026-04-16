import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuBroadcast(TestCase):
    def supported_op_exec(self, input1, size):
        output = torch.broadcast_to(input1, size)
        output = output.to("cpu")
        return output.numpy()

    def custom_op_exec(self, input1, size):
        output = torch_npu.npu_broadcast(input1, size)
        output = output.to("cpu")
        return output.numpy()

    def custom_op_out_exec(self, input1, size, out):
        return torch_npu.npu_broadcast.out(input1, size, out=out).to("cpu").numpy()

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_1d_to_2d(self):
        cpu_input = torch.tensor([1, 2, 3])
        npu_input = cpu_input.npu()
        size = (3, 3)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_output = self.custom_op_exec(npu_input, size)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_2d_to_2d(self):
        cpu_input = torch.tensor([[1], [2], [3]])
        npu_input = cpu_input.npu()
        size = (3, 3)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_output = self.custom_op_exec(npu_input, size)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_scalar_to_2d(self):
        cpu_input = torch.tensor(5.0)
        npu_input = cpu_input.npu()
        size = (2, 3)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_output = self.custom_op_exec(npu_input, size)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_1d_to_3d(self):
        cpu_input = torch.tensor([1.0, 2.0])
        npu_input = cpu_input.npu()
        size = (2, 2, 2)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_output = self.custom_op_exec(npu_input, size)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_2d_to_4d(self):
        cpu_input = torch.randn(1, 3)
        npu_input = cpu_input.npu()
        size = (2, 1, 3, 3)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_output = self.custom_op_exec(npu_input, size)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_dtype_float16(self):
        cpu_input = torch.randn(2, 1).half()
        npu_input = cpu_input.npu()
        size = (2, 4)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_output = self.custom_op_exec(npu_input, size)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_dtype_int32(self):
        cpu_input = torch.randint(0, 10, (2, 1), dtype=torch.int32)
        npu_input = cpu_input.npu()
        size = (2, 4)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_output = self.custom_op_exec(npu_input, size)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_out(self):
        cpu_input = torch.tensor([1.0, 2.0, 3.0])
        npu_input = cpu_input.npu()
        size = (2, 3)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_out = torch.empty(size, device="npu")
        npu_output = self.custom_op_out_exec(npu_input, size, npu_out)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_out_2d(self):
        cpu_input = torch.tensor([[1], [2], [3]], dtype=torch.float32)
        npu_input = cpu_input.npu()
        size = (3, 4)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_out = torch.empty(size, device="npu")
        npu_output = self.custom_op_out_exec(npu_input, size, npu_out)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend950'])
    def test_npu_broadcast_same_shape(self):
        cpu_input = torch.randn(2, 3)
        npu_input = cpu_input.npu()
        size = (2, 3)
        cpu_output = self.supported_op_exec(cpu_input, size)
        npu_output = self.custom_op_exec(npu_input, size)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
