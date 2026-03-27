import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestAddmm(TestCase):
    def cpu_op_exec(self, input1, input2, input3, scalar1=1.0, scalar2=1.0):
        output = torch.addmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3, scalar1=1.0, scalar2=1.0):
        output = torch.addmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def addmm_backward_result(self, shape_format):
        for item in shape_format:
            input1_info, input2_info, input3_info = item
            cpu_input1, npu_input1 = create_common_tensor(input1_info, -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(input2_info, -1, 1)
            cpu_input3, npu_input3 = create_common_tensor(input3_info, -1, 1)
            if input1_info[0] == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
                cpu_input3 = cpu_input3.to(torch.float32)
            if input2_info[-1]:
                cpu_input2 = cpu_input2.t()
                npu_input2 = npu_input2.t()
            if input3_info[-1]:
                cpu_input3 = cpu_input3.t()
                npu_input3 = npu_input3.t()
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            if input1_info[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)

    @SupportedDevices(['Ascend910B'])
    def test_mm_split_k_fp16(self):
        torch.npu.set_compile_mode(jit_compile=True)
        # dtype, format, shape, (transpose)
        shape_format = [
            [
                [np.float16, 2, [640, ]],
                [np.float16, 2, [55296, 640], True],
                [np.float16, 2, [55296, 640], False]
            ],
        ]
        self.addmm_backward_result(shape_format)

    # @SupportedDevices(['Ascend910D'])
    # def test_admm_cubemathtype_four(self):
    #     torch.npu.matmul.cube_math_type = torch.npu.CubeMathType.FORCE_GRP_ACC_FOR_FP32
    #     shape_format = [
    #         [
    #             [np.float32, 2, [4096, ]],
    #             [np.float32, 2, [1, 4096], False],
    #             [np.float32, 2, [4096, 4096], False]
    #         ],
    #     ]
    #     self.addmm_backward_result(shape_format)

if __name__ == "__main__":
    run_tests()
