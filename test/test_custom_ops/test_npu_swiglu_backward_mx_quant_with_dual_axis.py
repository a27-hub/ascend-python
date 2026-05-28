import math
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestSwigluBackwardMxQuantWithDualAxis(TestCase):
    def npu_op_exec(self, x, y_grad, group_index=None, activate_left=True, round_mode="rint",
                    scale_alg=1, dst_type=torch_npu.float8_e4m3fn, dst_type_max=0):
        return torch_npu._npu_swiglu_backward_mx_quant_with_dual_axis(x, y_grad,
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

    def _expected_mxscale1_shape(self, x_shape):
        """mxscale1: x_shape 末维替换为 ceil(N/64)，再追加 2"""
        shape = list(x_shape)
        shape[-1] = math.ceil(shape[-1] / 64)
        shape.append(2)
        return shape

    def _expected_mxscale2_shape(self, x_shape, group_index=None):
        """mxscale2: 倒数第二维替换为 quant_size，再追加 2
           有 group_index: quant_size = floor(M/64) + group_index.size(0)
           无 group_index: quant_size = ceil(M/64)
        """
        shape = list(x_shape)
        m = shape[-2]
        if group_index is not None:
            quant_size = math.floor(m / 64) + group_index.shape[0]
        else:
            quant_size = math.ceil(m / 64)
        shape[-2] = quant_size
        shape.append(2)
        return shape

    @SupportedDevices(['Ascend950'])
    def test_npu_swiglu_backward_mx_quant_with_dual_axis_output_shape(self, device="npu"):
        # x=[128,128], y_grad last_dim = x last_dim // 2 = 64
        x = self.generate_input([128, 128], value=1.0, dtype="bfloat16").to(device)
        y_grad = self.generate_input([128, 64], value=1.0, dtype="bfloat16").to(device)
        group_index = torch.tensor([2, 4], dtype=torch.int64).npu()

        y1_out, mxscale1, y2_out, mxscale2 = self.npu_op_exec(x, y_grad, group_index)

        # y1_out / y2_out 与 x 同形
        self.assertEqual(list(y1_out.shape), [128, 128])
        self.assertEqual(list(y2_out.shape), [128, 128])
        # mxscale1: [128, ceil(128/64), 2] = [128, 2, 2]
        self.assertEqual(list(mxscale1.shape), self._expected_mxscale1_shape([128, 128]))
        # mxscale2: floor(128/64)+2=4 → [4, 128, 2]
        self.assertEqual(list(mxscale2.shape), self._expected_mxscale2_shape([128, 128], group_index.cpu()))

    @SupportedDevices(['Ascend950'])
    def test_npu_swiglu_backward_mx_quant_with_dual_axis_without_group_index(self, device="npu"):
        # 无 group_index 时 mxscale2 用 ceil 计算 quant_size
        x = self.generate_input([128, 128], value=1.0, dtype="bfloat16").to(device)
        y_grad = self.generate_input([128, 64], value=1.0, dtype="bfloat16").to(device)

        y1_out, mxscale1, y2_out, mxscale2 = self.npu_op_exec(x, y_grad)

        self.assertEqual(list(y1_out.shape), [128, 128])
        self.assertEqual(list(y2_out.shape), [128, 128])
        self.assertEqual(list(mxscale1.shape), self._expected_mxscale1_shape([128, 128]))
        # mxscale2: ceil(128/64)=2 → [2, 128, 2]
        self.assertEqual(list(mxscale2.shape), self._expected_mxscale2_shape([128, 128]))

    @SupportedDevices(['Ascend950'])
    def test_npu_swiglu_backward_mx_quant_with_dual_axis_golden(self, device="npu"):
        # x=[4,8], y_grad=[4,4], group_index=[2,4]: 验证精确 shape 和 dtype
        x = self.generate_input([4, 8], value=1.0, dtype="bfloat16").to(device)
        y_grad = self.generate_input([4, 4], value=1.0, dtype="bfloat16").to(device)
        group_index = torch.tensor([2, 4], dtype=torch.int64).npu()

        y1_out, mxscale1, y2_out, mxscale2 = self.npu_op_exec(x, y_grad, group_index)

        # y1_out / y2_out: [4,8]; mxscale1: [4,1,2]; mxscale2: [2,8,2]
        self.assertEqual(list(y1_out.shape), [4, 8])
        self.assertEqual(list(y2_out.shape), [4, 8])
        self.assertEqual(list(mxscale1.shape), [4, 1, 2])
        self.assertEqual(list(mxscale2.shape), [2, 8, 2])
        # mxscale 均以 uint8 存储（float8_e8m0 用 Byte 张量承载）
        self.assertEqual(mxscale1.dtype, torch.uint8)
        self.assertEqual(mxscale2.dtype, torch.uint8)


if __name__ == "__main__":
    run_tests()
