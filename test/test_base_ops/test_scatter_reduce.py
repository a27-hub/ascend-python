import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestScatterReduce(TestCase):
    def cpu_op_exec(self, input1, dim, index, src, reduce, include_self):
        if hasattr(torch, "scatter_reduce"):
            output = torch.scatter_reduce(
                input1, dim, index, src, reduce=reduce, include_self=include_self
            )
        else:
            output = input1.scatter_reduce(dim, index, src, reduce=reduce, include_self=include_self)
        return output.numpy()

    def npu_op_exec(self, input1, dim, index, src, reduce, include_self):
        if hasattr(torch, "scatter_reduce"):
            output = torch.scatter_reduce(
                input1, dim, index, src, reduce=reduce, include_self=include_self
            )
        else:
            output = input1.scatter_reduce(dim, index, src, reduce=reduce, include_self=include_self)
        return output.to("cpu").numpy()

    def cpu_op_exec_out(self, input1, dim, index, src, reduce, include_self, output):
        if hasattr(torch, "scatter_reduce"):
            torch.scatter_reduce(
                input1, dim, index, src, reduce=reduce, include_self=include_self, out=output
            )
            return output.numpy()
        try:
            torch.ops.aten.scatter_reduce.out(input1, dim, index, src, reduce, include_self, out=output)
        except TypeError:
            torch.ops.aten.scatter_reduce.out(input1, dim, index, src, reduce, include_self, output)
        return output.numpy()

    def npu_op_exec_out(self, input1, dim, index, src, reduce, include_self, output):
        if hasattr(torch, "scatter_reduce"):
            torch.scatter_reduce(
                input1, dim, index, src, reduce=reduce, include_self=include_self, out=output
            )
            return output.to("cpu").numpy()
        try:
            torch.ops.aten.scatter_reduce.out(input1, dim, index, src, reduce, include_self, out=output)
        except TypeError:
            torch.ops.aten.scatter_reduce.out(input1, dim, index, src, reduce, include_self, output)
        return output.to("cpu").numpy()

    def cpu_op_exec_inp(self, input1, dim, index, src, reduce, include_self):
        if hasattr(input1, "scatter_reduce_"):
            input1.scatter_reduce_(dim, index, src, reduce=reduce, include_self=include_self)
        else:
            torch.ops.aten.scatter_reduce_(input1, dim, index, src, reduce, include_self)
        return input1.numpy()

    def npu_op_exec_inp(self, input1, dim, index, src, reduce, include_self):
        if hasattr(input1, "scatter_reduce_"):
            input1.scatter_reduce_(dim, index, src, reduce=reduce, include_self=include_self)
        else:
            torch.ops.aten.scatter_reduce_(input1, dim, index, src, reduce, include_self)
        return input1.to("cpu").numpy()

    def test_scatter_reduce_float32_shape_format(self):
        shape_format = [
            [0, [np.int64, 0, [10, 20]], [np.float32, 0, [10, 20]], [np.float32, 0, [10, 20]]],
            [1, [np.int64, 0, [10, 20]], [np.float32, 0, [10, 20]], [np.float32, 0, [10, 20]]],
            [0, [np.int64, 0, [2, 6]], [np.float32, 0, [2, 6]], [np.float32, 0, [2, 6]]],
            [1, [np.int64, 0, [2, 6]], [np.float32, 0, [2, 6]], [np.float32, 0, [2, 6]]],
            [0, [np.int64, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]]],
            [1, [np.int64, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]]],
            [2, [np.int64, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]]],
        ]
        reduce_list = ["sum", "amin", "amax", "prod", "mean"]
        include_self_list = [True, False]

        for item in shape_format:
            for reduce in reduce_list:
                for include_self in include_self_list:
                    cpu_src, npu_src = create_common_tensor(item[2], 1, 100)
                    cpu_index, npu_index = create_common_tensor(item[1], 0, (item[1][2][item[0]] - 1))
                    cpu_input, npu_input = create_common_tensor(item[3], 1, 100)

                    cpu_output = self.cpu_op_exec(cpu_input, item[0], cpu_index, cpu_src, reduce, include_self)
                    npu_output = self.npu_op_exec(npu_input, item[0], npu_index, npu_src, reduce, include_self)
                    self.assertRtolEqual(cpu_output, npu_output)

                    cpu_out_buf = torch.empty_like(cpu_input)
                    npu_out_buf = torch.empty_like(npu_input)
                    cpu_output_out = self.cpu_op_exec_out(
                        cpu_input, item[0], cpu_index, cpu_src, reduce, include_self, cpu_out_buf
                    )
                    npu_output_out = self.npu_op_exec_out(
                        npu_input, item[0], npu_index, npu_src, reduce, include_self, npu_out_buf
                    )
                    self.assertRtolEqual(cpu_output_out, npu_output_out)

                    cpu_inp_output = self.cpu_op_exec_inp(
                        cpu_input.clone(), item[0], cpu_index, cpu_src, reduce, include_self
                    )
                    npu_inp_output = self.npu_op_exec_inp(
                        npu_input.clone(), item[0], npu_index, npu_src, reduce, include_self
                    )
                    self.assertRtolEqual(cpu_inp_output, npu_inp_output)

    def test_scatter_reduce_float16_shape_format(self):
        def cpu_op_exec_fp16(input1, dim, index, src, reduce, include_self):
            if hasattr(torch, "scatter_reduce"):
                output = torch.scatter_reduce(
                    input1, dim, index, src, reduce=reduce, include_self=include_self
                )
            else:
                output = input1.scatter_reduce(dim, index, src, reduce=reduce, include_self=include_self)
            return output.float().numpy().astype(np.float16)

        def cpu_op_exec_inp_fp16(input1, dim, index, src, reduce, include_self):
            if hasattr(input1, "scatter_reduce_"):
                input1.scatter_reduce_(dim, index, src, reduce=reduce, include_self=include_self)
            else:
                torch.ops.aten.scatter_reduce_(input1, dim, index, src, reduce, include_self)
            return input1.float().numpy().astype(np.float16)

        shape_format = [
            [0, [np.int64, 0, [10, 20]], [np.float16, 0, [10, 20]], [np.float16, 0, [10, 20]]],
            [1, [np.int64, 0, [10, 20]], [np.float16, 0, [10, 20]], [np.float16, 0, [10, 20]]],
            [0, [np.int64, 0, [2, 6]], [np.float16, 0, [2, 6]], [np.float16, 0, [2, 6]]],
            [1, [np.int64, 0, [2, 6]], [np.float16, 0, [2, 6]], [np.float16, 0, [2, 6]]],
            [0, [np.int64, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]]],
            [1, [np.int64, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]]],
            [2, [np.int64, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]]],
        ]
        reduce_list = ["amin", "amax"]
        include_self_list = [True, False]

        for item in shape_format:
            for reduce in reduce_list:
                for include_self in include_self_list:
                    cpu_src, npu_src = create_common_tensor(item[2], 1, 100)
                    cpu_index, npu_index = create_common_tensor(item[1], 0, (item[1][2][item[0]] - 1))
                    cpu_input, npu_input = create_common_tensor(item[3], 1, 100)

                    cpu_output = cpu_op_exec_fp16(cpu_input, item[0], cpu_index, cpu_src, reduce, include_self)
                    npu_output = self.npu_op_exec(npu_input, item[0], npu_index, npu_src, reduce, include_self)
                    self.assertRtolEqual(cpu_output, npu_output)

                    cpu_inp_output = cpu_op_exec_inp_fp16(
                        cpu_input, item[0], cpu_index, cpu_src, reduce, include_self
                    )
                    npu_inp_output = self.npu_op_exec_inp(
                        npu_input, item[0], npu_index, npu_src, reduce, include_self
                    )
                    self.assertRtolEqual(cpu_inp_output, npu_inp_output)


if __name__ == "__main__":
    run_tests()
