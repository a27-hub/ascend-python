import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGather(TestCase):
    def cpu_op_exec(self, input1, dim, index, sparse_grad=False):
        input1.requires_grad = True
        output = torch.gather(input1, dim, index, sparse_grad=sparse_grad)
        output.backward(torch.ones_like(output))
        output = output.detach().numpy()
        grad = input1.grad.to_dense() if sparse_grad else input1.grad
        return output, grad

    def npu_op_exec(self, input1, dim, index, sparse_grad=False):
        input1.requires_grad = True
        output = torch.gather(input1, dim, index, sparse_grad=sparse_grad)
        output.backward(torch.ones_like(output))
        output = output.to("cpu")
        output = output.detach().numpy()
        grad = input1.grad.to_dense() if sparse_grad else input1.grad
        return output, grad

    def test_gather_backward_shape_format(self):
        shape_format = [
            [[np.float32, 0, (3, 3, 5)], -3, torch.LongTensor([[[0, 1, 2, 0, 2], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                                                               [[1, 2, 2, 2, 2], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2]]])]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
            npu_idx = item[2].to("npu")
            npu_output = self.npu_op_exec(npu_input1, item[1], npu_idx)
            self.assertRtolEqual(cpu_output, npu_output)


    def test_gather_gather_saprse_grad(self):
        shape_format = [
            [[np.float32, 0, (3, 3, 5)], -3, torch.LongTensor([[[0, 1, 2, 0, 2], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                                                               [[1, 2, 2, 2, 2], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2]]])]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2], True)
            npu_idx = item[2].to("npu")
            npu_output = self.npu_op_exec(npu_input1, item[1], npu_idx, True)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_gather_backward_with_empty_index_tensor_sparse_grad(self):
        dim = -1
        cpu_input = torch.rand([10, 5], dtype=torch.float32, requires_grad=True)
        npu_input = cpu_input.detach().clone().to("npu").requires_grad_(True)
        cpu_index = torch.randint(0, 2, [3, 0], dtype=torch.int64)
        npu_index = cpu_index.to("npu")

        cpu_output = torch.gather(cpu_input, dim, cpu_index, sparse_grad=True)
        cpu_output.sum().backward()
        cpu_grad = cpu_input.grad.to_dense()

        npu_output = torch.gather(npu_input, dim, npu_index, sparse_grad=True)
        npu_output.sum().backward()
        npu_grad = npu_input.grad.to_dense().cpu()

        expected_grad = torch.zeros_like(cpu_input, requires_grad=False)
        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.cpu().detach().numpy())
        self.assertRtolEqual(cpu_grad, npu_grad)
        self.assertRtolEqual(npu_grad, expected_grad)


if __name__ == "__main__":
    run_tests()
