import random
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIndexPut(TestCase):
    def cpu_op_exec(self, input1, indices, value):
        output = input1.index_put(indices, value, accumulate=True)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, indices, value):
        output = input1.index_put(indices, value, accumulate=True)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inp_exec(self, input1, indices, value):
        input1.index_put_(indices, value, accumulate=True)
        output = input1.numpy()
        return output

    def npu_op_inp_exec(self, input1, indices, value):
        input1.index_put_(indices, value, accumulate=True)
        input1 = input1.to("cpu")
        output = input1.numpy()
        return output

    def case_exec(self, shape):
        cpu_indices2 = []
        npu_indices2 = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices2.append(cpu_indices1)
                npu_indices2.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_indices2, cpu_value)
            npu_output = self.npu_op_exec(npu_input, npu_indices2, npu_value)
            self.assertEqual(cpu_output, npu_output)

    def case_exec_fp16(self, shape):
        cpu_indices3 = []
        npu_indices3 = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_input = cpu_input.to(torch.float32)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices3.append(cpu_indices1)
                npu_indices3.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_value = cpu_value.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_indices3, cpu_value)
            npu_output = self.npu_op_exec(npu_input, npu_indices3, npu_value)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def case_inp_exec(self, shape):
        cpu_indices4 = []
        npu_indices4 = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices4.append(cpu_indices1)
                npu_indices4.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_output = self.cpu_op_inp_exec(
                cpu_input, cpu_indices4, cpu_value)
            npu_output = self.npu_op_inp_exec(
                npu_input, npu_indices4, npu_value)
            self.assertEqual(cpu_output, npu_output)

    def case_inp_exec_fp16(self, shape):
        cpu_indices5 = []
        npu_indices5 = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_input = cpu_input.to(torch.float32)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices5.append(cpu_indices1)
                npu_indices5.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_value = cpu_value.to(torch.float32)
            cpu_output = self.cpu_op_inp_exec(
                cpu_input, cpu_indices5, cpu_value)
            npu_output = self.npu_op_inp_exec(
                npu_input, npu_indices5, npu_value)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_index_put_shape_format_fp32(self):
        format_list = [0]
        shape_list = [(5, 6)]
        shape_format = [[[np.float32, i, j], [np.int64, 0, [1, 2]], [
            np.float32, 0, [1, 2]]] for i in format_list for j in shape_list]
        self.case_exec(shape_format)
        self.case_inp_exec(shape_format)

    def test_index_put_shape_format_fp16(self):
        format_list = [0]
        shape_list = [(5, 6)]
        shape_format = [[[np.float16, i, j], [np.int64, 0, [1, 2]], [
            np.float16, 0, [1, 2]]] for i in format_list for j in shape_list]
        self.case_exec_fp16(shape_format)
        self.case_inp_exec_fp16(shape_format)

    def test_index_put_null(self):
        cpu_input1 = torch.rand(2, 2)
        cpu_input2 = torch.rand(2, 2)
        cpu_mask_index = torch.tensor([[False, False], [False, False]])
        npu_mask_index = cpu_mask_index.to("npu")
        npu_input1 = cpu_input1.to("npu")
        npu_input2 = cpu_input2.to("npu")
        cpu_input1[cpu_mask_index] = cpu_input2.detach()[cpu_mask_index]
        npu_input1[npu_mask_index] = npu_input2.detach()[npu_mask_index]
        self.assertEqual(cpu_input1, npu_input1.to("cpu"))

    def test_index_put_undefined_fp32(self):
        cinput = torch.randn(4, 3, 2, 3)
        ninput = cinput.npu()
        cinput[:, [[1, 2, 1], [1, 2, 0]], :, [[1, 0, 2]]] = 1000
        ninput[:, [[1, 2, 1], [1, 2, 0]], :, [[1, 0, 2]]] = 1000
        self.assertRtolEqual(cinput.numpy(), ninput.cpu().numpy())

    def test_index_put_tensor_fp32(self):
        cinput = torch.randn(4, 4, 4, 4)
        ninput = cinput.npu()
        value = torch.tensor([100, 200, 300, 400], dtype=torch.float32)
        cinput[:, :, [0, 1, 2, 3], [0, 1, 2, 3]] = value
        ninput[:, :, [0, 1, 2, 3], [0, 1, 2, 3]] = value.npu()
        self.assertRtolEqual(cinput.numpy(), ninput.cpu().numpy())

    def test_index_put_NCDHW_and_resize(self):
        index_list = [20, 26, 21, 8, 25], [8, 15, 2, 17, 1], [20, 79, 5, 28, 45], [24, 3, 44, 37, 10], [2, 3, 0, 10, 14]
        index = [torch.LongTensor(i) for i in index_list]
        input_data = torch.randn(32, 20, 80, 80, 16)
        input2 = torch.randn(5,)

        index_npu = [i.npu() for i in index]
        input_data_npu = input_data.npu()
        input2_npu = input2.npu()

        input_data.index_put_(index, input2)
        cpu_res = input_data.numpy()
        input_data_npu.index_put_(index_npu, input2_npu)
        npu_res = input_data_npu.cpu().numpy()
        self.assertRtolEqual(cpu_res, npu_res)

    def test_index_put_different_device(self):
        m = random.randint(10, 20)
        elems = random.randint(20000, 30000)
        cpu_values = torch.rand(elems)
        npu_values = cpu_values.npu()
        indices = torch.randint(m, (elems,))
        cpu_input1 = torch.rand(m)
        npu_input1 = cpu_input1.npu()
        cpu_output1 = self.cpu_op_exec(cpu_input1, (indices,), cpu_values)
        npu_output1 = self.npu_op_exec(npu_input1, (indices,), npu_values)
        self.assertRtolEqual(cpu_output1, npu_output1)

        m = random.randint(10, 20)
        elems = random.randint(20000, 30000)
        values = torch.rand(elems)
        cpu_indices = torch.randint(m, (elems,))
        npu_indices = cpu_indices.npu()
        input1 = torch.rand(m)
        cpu_output2 = self.cpu_op_exec(input1, (cpu_indices,), values)
        npu_output2 = self.npu_op_exec(input1, (npu_indices,), values)
        self.assertRtolEqual(cpu_output2, npu_output2)

    def test_index_put_shape_mismatch(self):
        torch.set_default_device('npu')
        data_tensor = torch.zeros(3, 3)
        indices = (torch.tensor([0, 2]), torch.tensor([1, 0]))
        values = torch.tensor([[[[5.0]]], [[[10.0]]]])
        torch.set_default_device('cpu')
        with self.assertRaises(RuntimeError):
            data_tensor.index_put_(indices, values)

    def test_index_put_shape_mismatch_broadcast_together(self):
        torch.set_default_device('npu')
        data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int64)
        indices = (torch.tensor([True, False, True]), torch.tensor([0, 1, 2]))
        values = torch.tensor([10, 20], dtype=torch.int64)
        torch.set_default_device('cpu')
        with self.assertRaises(IndexError):
            result = data.index_put(indices, values, accumulate=False)

    def make_complex_base(self, shape, dtype):
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        numel = 1
        for dim in shape:
            numel *= dim
        real = torch.arange(1, numel + 1, dtype=float_dtype).reshape(shape)
        return torch.complex(real, real + 0.5)

    def case_complex_exec(self, dtype):
        # Data is built on CPU with exactly representable values and then
        # copied to NPU, to keep accumulate results bit-comparable.
        idx_row = torch.tensor([3, 1])
        idx_dup = torch.tensor([2, 0, 0, 3])
        value = self.make_complex_base((2, 5), dtype)
        value_dup = self.make_complex_base((4, 5), dtype)

        # out-of-place unique replace
        cpu_base = self.make_complex_base((4, 5), dtype)
        npu_base = cpu_base.npu()
        cpu_out = cpu_base.index_put((idx_row,), value, accumulate=False)
        npu_out = npu_base.index_put((idx_row.npu(),), value.npu(),
                                     accumulate=False)
        self.assertEqual(cpu_out.numpy(), npu_out.cpu().numpy())

        # out-of-place accumulate with duplicate and out-of-order indices
        cpu_out2 = cpu_base.index_put((idx_dup,), value_dup, accumulate=True)
        npu_out2 = npu_base.index_put((idx_dup.npu(),), value_dup.npu(),
                                      accumulate=True)
        self.assertEqual(cpu_out2.numpy(), npu_out2.cpu().numpy())

        # in-place accumulate with duplicate and out-of-order indices
        cpu_inp = self.make_complex_base((4, 5), dtype)
        npu_inp = cpu_inp.npu()
        cpu_inp.index_put_((idx_dup,), value_dup, accumulate=True)
        npu_inp.index_put_((idx_dup.npu(),), value_dup.npu(), accumulate=True)
        self.assertEqual(cpu_inp.numpy(), npu_inp.cpu().numpy())

        # in-place unique replace
        cpu_inp2 = self.make_complex_base((4, 5), dtype)
        npu_inp2 = cpu_inp2.npu()
        cpu_inp2.index_put_((idx_row,), value, accumulate=False)
        npu_inp2.index_put_((idx_row.npu(),), value.npu(), accumulate=False)
        self.assertEqual(cpu_inp2.numpy(), npu_inp2.cpu().numpy())

        # row value broadcast
        row_value = self.make_complex_base((5,), dtype)
        cpu_inp3 = self.make_complex_base((4, 5), dtype)
        npu_inp3 = cpu_inp3.npu()
        cpu_inp3.index_put_((idx_row,), row_value, accumulate=False)
        npu_inp3.index_put_((idx_row.npu(),), row_value.npu(),
                            accumulate=False)
        self.assertEqual(cpu_inp3.numpy(), npu_inp3.cpu().numpy())

        # complex scalar value; the update decomposes to two per-lane real
        # writes (regression shape: a trailing-undefined index call would
        # silently mis-write this case on the current float32 real kernel)
        scalar_value = torch.tensor(2 + 3j, dtype=dtype)
        cpu_inp4 = self.make_complex_base((4, 5), dtype)
        npu_inp4 = cpu_inp4.npu()
        cpu_inp4.index_put_((idx_row,), scalar_value, accumulate=True)
        npu_inp4.index_put_((idx_row.npu(),), scalar_value.npu(),
                            accumulate=True)
        self.assertEqual(cpu_inp4.numpy(), npu_inp4.cpu().numpy())

        # direct (accumulate=true, unsafe=true) entry: the signature the
        # autograd backward of index reaches this op with
        cpu_inp5 = self.make_complex_base((4, 5), dtype)
        npu_inp5 = cpu_inp5.npu()
        grad = self.make_complex_base((2, 5), dtype)
        torch.ops.aten._index_put_impl_(cpu_inp5, (idx_row,), grad, True, True)
        torch.ops.aten._index_put_impl_(npu_inp5, (idx_row.npu(),), grad.npu(),
                                        True, True)
        self.assertEqual(cpu_inp5.numpy(), npu_inp5.cpu().numpy())

        # bool mask indices on the original rank; value numel > 1 keeps the
        # per-lane write out of the masked_fill fast path
        mask = torch.zeros(4, 5, dtype=torch.bool)
        mask[1, 2] = True
        mask[3, 4] = True
        value_mask = self.make_complex_base((2,), dtype)
        cpu_inp6 = self.make_complex_base((4, 5), dtype)
        npu_inp6 = cpu_inp6.npu()
        cpu_inp6.index_put_((mask,), value_mask, accumulate=False)
        npu_inp6.index_put_((mask.npu(),), value_mask.npu(), accumulate=False)
        self.assertEqual(cpu_inp6.numpy(), npu_inp6.cpu().numpy())

        # non-contiguous target: the write must alias back to the base tensor
        cpu_nc = self.make_complex_base((5, 4), dtype)
        npu_nc = cpu_nc.npu()
        cpu_dst = cpu_nc.t()
        npu_dst = npu_nc.t()
        self.assertEqual(cpu_dst.data_ptr(), cpu_nc.data_ptr())
        self.assertEqual(npu_dst.data_ptr(), npu_nc.data_ptr())
        cpu_dst.index_put_((idx_row,), value, accumulate=False)
        npu_dst.index_put_((idx_row.npu(),), value.npu(), accumulate=False)
        self.assertEqual(npu_dst.data_ptr(), npu_nc.data_ptr())
        self.assertEqual(cpu_nc.numpy(), npu_nc.cpu().numpy())

    def test_index_put_complex64(self):
        self.case_complex_exec(torch.complex64)

    def test_index_put_complex128(self):
        self.case_complex_exec(torch.complex128)

    def test_index_put_complex_dtype_mismatch(self):
        for dtype in (torch.complex64, torch.complex128):
            cpu_base = self.make_complex_base((2, 3), dtype)
            npu_base = cpu_base.npu()
            value = torch.ones(2, 3, dtype=torch.float32)
            with self.assertRaises(RuntimeError):
                cpu_base.index_put_((torch.tensor([0, 1]),), value)
            with self.assertRaises(RuntimeError):
                npu_base.index_put_((torch.tensor([0, 1]).npu(),),
                                    value.npu())

    def test_index_put_complex_too_many_indices(self):
        for dtype in (torch.complex64, torch.complex128):
            cpu_base = self.make_complex_base((2, 3), dtype)
            npu_base = cpu_base.npu()
            value = self.make_complex_base((1,), dtype)
            with self.assertRaises(IndexError):
                cpu_base.index_put_(
                    (torch.tensor([0]), torch.tensor([1]),
                     torch.tensor([0])), value)
            with self.assertRaises(IndexError):
                npu_base.index_put_(
                    (torch.tensor([0]).npu(), torch.tensor([1]).npu(),
                     torch.tensor([0]).npu()), value.npu())


if __name__ == "__main__":
    run_tests()
