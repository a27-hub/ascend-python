import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestSwigluBackwardMxQuantWithDualAxis(TestCase):
    def npu_op_exec(self, x, y_grad, group_index=None, activate_left=True, round_mode="rint",
                    scale_alg=1, dst_type=torch_npu.float8_e4m3fn, dst_type_max=0):
        return torch_npu.npu_swiglu_backward_mx_quant_with_dual_axis(x, y_grad,
                                                                      group_index=group_index,
                                                                      activate_left=activate_left,
                                                                      round_mode=round_mode,
                                                                      scale_alg=scale_alg,
                                                                      dst_type=dst_type,
                                                                      dst_type_max=dst_type_max)

    def generate_input(self, shape, value, dtype="bfloat16"):
        if dtype == "float16":
            data_type = torch.float16
        elif dtype == "bfloat16":
            data_type = torch.bfloat16
        return torch.full(shape, value, dtype=data_type)

    @SupportedDevices(['Ascend950'])
    def test_npu_swiglu_backward_mx_quant_with_dual_axis(self, device="npu"):
        x = self.generate_input([4, 8], value=1, dtype="bfloat16").to(device)
        y_grad = self.generate_input([4, 8], value=1, dtype="bfloat16").to(device)
        group_index = torch.tensor([2, 4], dtype=torch.int64).npu()

        npu_output = self.npu_op_exec(x, y_grad, group_index)

        y1_out, mxscale1, y2_out, mxscale2 = npu_output

        # y1_out and y2_out have the same shape as x
        self.assertEqual(y1_out.shape, x.shape)
        self.assertEqual(y2_out.shape, x.shape)
        # mxscale1: [batch, ceil(last_dim/2 / 64), 2]
        self.assertEqual(mxscale1.shape[0], x.shape[0])
        self.assertEqual(mxscale1.shape[-1], 2)
        # mxscale2: [..., quant_size, half_last_dim, 2]
        self.assertEqual(mxscale2.shape[-1], 2)


if __name__ == "__main__":
    run_tests()
