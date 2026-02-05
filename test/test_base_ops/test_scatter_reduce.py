import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestScatterReduce(TestCase):

    def cpu_op_exec(self, input_tensor, dim, index, src, reduce):
        output = input_tensor.scatter_reduce(dim, index, src, reduce=reduce)
        return output.numpy()

    def npu_op_exec(self, input_tensor, dim, index, src, reduce):
        input_npu = input_tensor.npu()
        index_npu = index.npu()
        src_npu = src.npu()
        output = input_npu.scatter_reduce(dim, index_npu, src_npu, reduce=reduce)
        return output.cpu().numpy()

    def cpu_op_exec_inplace(self, input_tensor, dim, index, src, reduce):
        input_tensor.scatter_reduce_(dim, index, src, reduce=reduce)
        return input_tensor.numpy()

    def npu_op_exec_inplace(self, input_tensor, dim, index, src, reduce):
        input_npu = input_tensor.npu()
        index_npu = index.npu()
        src_npu = src.npu()
        input_npu.scatter_reduce_(dim, index_npu, src_npu, reduce=reduce)
        return input_npu.cpu().numpy()

    def test_scatter_reduce_prod_basic(self):
        input_cpu = torch.ones(3, dtype=torch.float32)
        index = torch.tensor([0, 0, 0], dtype=torch.int64)
        src = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

        cpu_output = self.cpu_op_exec(input_cpu.clone(), 0, index, src, "prod")
        npu_output = self.npu_op_exec(input_cpu.clone(), 0, index, src, "prod")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_reduce_amax_basic(self):
        input_cpu = torch.zeros(3, dtype=torch.float32)
        index = torch.tensor([0, 0, 1], dtype=torch.int64)
        src = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32)

        cpu_output = self.cpu_op_exec(input_cpu.clone(), 0, index, src, "amax")
        npu_output = self.npu_op_exec(input_cpu.clone(), 0, index, src, "amax")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_reduce_amin_basic(self):
        input_cpu = torch.full((3,), 10.0, dtype=torch.float32)
        index = torch.tensor([0, 0, 1], dtype=torch.int64)
        src = torch.tensor([5.0, 3.0, 8.0], dtype=torch.float32)

        cpu_output = self.cpu_op_exec(input_cpu.clone(), 0, index, src, "amin")
        npu_output = self.npu_op_exec(input_cpu.clone(), 0, index, src, "amin")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_reduce_amax_negative(self):
        input_cpu = torch.tensor([-5.0, -5.0, -5.0], dtype=torch.float32)
        index = torch.tensor([0, 0, 1], dtype=torch.int64)
        src = torch.tensor([-1.0, -3.0, -2.0], dtype=torch.float32)

        cpu_output = self.cpu_op_exec(input_cpu.clone(), 0, index, src, "amax")
        npu_output = self.npu_op_exec(input_cpu.clone(), 0, index, src, "amax")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_reduce_2d(self):
        for reduce in ["prod", "amax", "amin"]:
            input_cpu = torch.randn(3, 5, dtype=torch.float32)
            if reduce == "prod":
                input_cpu = torch.ones(3, 5, dtype=torch.float32)
            index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64)
            src = torch.randn(2, 5, dtype=torch.float32)

            cpu_output = self.cpu_op_exec(input_cpu.clone(), 0, index, src, reduce)
            npu_output = self.npu_op_exec(input_cpu.clone(), 0, index, src, reduce)

            self.assertRtolEqual(cpu_output, npu_output, prec=1e-4)

    def test_scatter_reduce_float16(self):
        for reduce in ["prod", "amax", "amin"]:
            input_cpu = torch.randn(3, 5, dtype=torch.float16)
            if reduce == "prod":
                input_cpu = torch.ones(3, 5, dtype=torch.float16)
            index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64)
            src = torch.randn(2, 5, dtype=torch.float16)

            cpu_input_f32 = input_cpu.clone().float()
            src_f32 = src.clone().float()
            cpu_output = self.cpu_op_exec(cpu_input_f32, 0, index, src_f32, reduce)

            npu_output = self.npu_op_exec(input_cpu.clone(), 0, index, src, reduce)

            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output, prec=1e-3)

    def test_scatter_reduce_int32(self):
        for reduce in ["prod", "amax", "amin"]:
            if reduce == "prod":
                input_cpu = torch.ones(3, 5, dtype=torch.int32)
            else:
                input_cpu = torch.randint(-10, 10, (3, 5), dtype=torch.int32)
            index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64)
            src = torch.randint(-10, 10, (2, 5), dtype=torch.int32)

            cpu_output = self.cpu_op_exec(input_cpu.clone(), 0, index, src, reduce)
            npu_output = self.npu_op_exec(input_cpu.clone(), 0, index, src, reduce)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_reduce_inplace(self):
        for reduce in ["prod", "amax", "amin"]:
            if reduce == "prod":
                input_cpu = torch.ones(3, 5, dtype=torch.float32)
                input_npu = torch.ones(3, 5, dtype=torch.float32)
            else:
                input_cpu = torch.randn(3, 5, dtype=torch.float32)
                input_npu = input_cpu.clone()
            index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64)
            src = torch.randn(2, 5, dtype=torch.float32)

            cpu_output = self.cpu_op_exec_inplace(input_cpu, 0, index, src, reduce)
            npu_output = self.npu_op_exec_inplace(input_npu, 0, index, src, reduce)

            self.assertRtolEqual(cpu_output, npu_output, prec=1e-4)

    def test_scatter_reduce_multidim(self):
        for reduce in ["prod", "amax", "amin"]:
            if reduce == "prod":
                input_cpu = torch.ones(2, 3, 4, 5, dtype=torch.float32)
            else:
                input_cpu = torch.randn(2, 3, 4, 5, dtype=torch.float32)
            index = torch.randint(0, 3, (2, 2, 4, 5), dtype=torch.int64)
            src = torch.randn(2, 2, 4, 5, dtype=torch.float32)

            cpu_output = self.cpu_op_exec(input_cpu.clone(), 1, index, src, reduce)
            npu_output = self.npu_op_exec(input_cpu.clone(), 1, index, src, reduce)

            self.assertRtolEqual(cpu_output, npu_output, prec=1e-4)

    def test_scatter_reduce_large_tensor(self):
        for reduce in ["prod", "amax", "amin"]:
            if reduce == "prod":
                input_cpu = torch.ones(128, 256, dtype=torch.float32)
            else:
                input_cpu = torch.randn(128, 256, dtype=torch.float32)
            index = torch.randint(0, 128, (64, 256), dtype=torch.int64)
            src = torch.randn(64, 256, dtype=torch.float32)

            cpu_output = self.cpu_op_exec(input_cpu.clone(), 0, index, src, reduce)
            npu_output = self.npu_op_exec(input_cpu.clone(), 0, index, src, reduce)

            self.assertRtolEqual(cpu_output, npu_output, prec=1e-4)


if __name__ == "__main__":
    run_tests()
