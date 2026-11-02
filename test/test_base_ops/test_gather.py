import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGather(TestCase):
    def test_gather_empty_index(self):
        cases = [
            ((5,), (2, 0), 0),
            ((5,), (2, 0), -1),
            ((2, 3), (0,), 1),
            ((2, 3), (2, 0), -1),
            ((0,), (3, 0), 0),
            ((), (0,), 0),
        ]
        for shape, index_shape, dim in cases:
            for index_dtype in (torch.int64, torch.float32):
                with self.subTest(shape=shape, index_shape=index_shape, dim=dim, index_dtype=index_dtype):
                    cpu_input = torch.zeros(shape, dtype=torch.float32)
                    cpu_index = torch.empty(index_shape, dtype=index_dtype)
                    expected = torch.gather(cpu_input, dim, cpu_index)
                    npu_input, npu_index = cpu_input.npu(), cpu_index.npu()
                    for sparse_grad in (False, True):
                        actual = torch.gather(npu_input, dim, npu_index, sparse_grad=sparse_grad)
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertEqual(actual.device, npu_input.device)
                        self.assertEqual(actual.numel(), 0)

    def test_gather_empty_index_out(self):
        cpu_input = torch.arange(5, dtype=torch.float32)
        cpu_index = torch.empty((2, 0), dtype=torch.int64)
        for out_shape in ((0,), (2, 0), (2, 3)):
            with self.subTest(out_shape=out_shape):
                cpu_out = torch.empty(out_shape)
                expected = torch.gather(cpu_input, 0, cpu_index, out=cpu_out)
                npu_input, npu_index = cpu_input.npu(), cpu_index.npu()
                npu_out = torch.empty(out_shape, device=npu_input.device)
                actual = torch.gather(npu_input, 0, npu_index, out=npu_out)
                self.assertIs(actual, npu_out)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.device, npu_input.device)
                self.assertEqual(actual.numel(), 0)

    def test_gather_empty_index_invalid_dim(self):
        for device in ('cpu', 'npu'):
            x = torch.zeros(5, device=device)
            index = torch.empty((2, 0), dtype=torch.int64, device=device)
            for dim in (-2, 1):
                with self.subTest(device=device, dim=dim):
                    with self.assertRaises(IndexError):
                        torch.gather(x, dim, index)
                    out = torch.empty(0, device=device)
                    with self.assertRaises(IndexError):
                        torch.gather(x, dim, index, out=out)

    def test_gather_empty_index_out_dtype(self):
        for device in ('cpu', 'npu'):
            x = torch.zeros(5, device=device)
            index = torch.empty((2, 0), dtype=torch.int64, device=device)
            out = torch.empty(0, dtype=torch.int64, device=device)
            with self.subTest(device=device):
                with self.assertRaises(RuntimeError):
                    torch.gather(x, 0, index, out=out)

    def test_gather_nonempty_invalid_index(self):
        for device in ('cpu', 'npu'):
            x = torch.zeros(5, device=device)
            for shape, dtype in (((2, 1), torch.int64), ((1,), torch.float32)):
                with self.subTest(device=device, shape=shape, dtype=dtype):
                    index = torch.zeros(shape, dtype=dtype, device=device)
                    with self.assertRaises(RuntimeError):
                        torch.gather(x, 0, index)

    def cpu_op_exec(self, input1, dim, index):
        output = torch.gather(input1, dim, index)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim, index):
        output = torch.gather(input1, dim, index)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_gather_shape_format(self):
        shape_format = [
            [[np.int32, 0, (4, 3)], 0, torch.LongTensor([[0, 1, 1], [2, 0, 1]])],
            [[np.int64, 0, (2, 3)], 1, torch.LongTensor([[0, 1, 1], [0, 0, 1]])],
            [[np.float16, 0, (2, 3, 5)], 2, torch.LongTensor([[[0, 1, 2, 0, 2], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                                                              [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]])],
            [[np.float32, 0, (3, 3, 5)], -3, torch.LongTensor([[[0, 1, 2, 0, 2], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                                                               [[1, 2, 2, 2, 2], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2]]])]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
            npu_idx = item[2].to("npu")
            npu_output = self.npu_op_exec(npu_input1, item[1], npu_idx)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
