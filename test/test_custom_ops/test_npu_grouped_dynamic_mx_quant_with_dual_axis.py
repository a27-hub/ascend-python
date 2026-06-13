import math
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestGroupedDynamicMxQuantWithDualAxis(TestCase):
    def npu_op_exec(self, x, group_index, round_mode="rint", scale_alg=1,
                    dst_type=torch_npu.float8_e4m3fn, dst_type_max=0):
        return torch_npu.npu_grouped_dynamic_mx_quant_with_dual_axis(
            x,
            group_index,
            round_mode=round_mode,
            scale_alg=scale_alg,
            dst_type=dst_type,
            dst_type_max=dst_type_max)

    def generate_input(self, shape, value=0.0, dtype="bfloat16"):
        data_type = torch.float16 if dtype == "float16" else torch.bfloat16
        return torch.full(shape, value, dtype=data_type)

    def _expected_mxscale1_shape(self, x_shape):
        return [x_shape[0], x_shape[1] // 64, 2]

    def _expected_mxscale2_shape(self, x_shape, group_index):
        return [math.floor(x_shape[0] / 64) + group_index.shape[0], x_shape[1], 2]

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_dynamic_mx_quant_with_dual_axis_output_shape(self, device="npu"):
        x = self.generate_input([128, 128], value=1.0, dtype="bfloat16").to(device)
        group_index = torch.tensor([64, 128], dtype=torch.int64).npu()

        y1, mxscale1, y2, mxscale2 = self.npu_op_exec(x, group_index)

        self.assertEqual(list(y1.shape), [128, 128])
        self.assertEqual(list(y2.shape), [128, 128])
        self.assertEqual(list(mxscale1.shape), self._expected_mxscale1_shape([128, 128]))
        self.assertEqual(list(mxscale2.shape), self._expected_mxscale2_shape([128, 128], group_index.cpu()))

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_dynamic_mx_quant_with_dual_axis_matches_split_ops(self, device="npu"):
        x = self.generate_input([64, 64], value=0.0, dtype="bfloat16").to(device)
        group_index = torch.tensor([64], dtype=torch.int64).npu()

        expected_y1, expected_mxscale1 = torch_npu.npu_dynamic_mx_quant(
            x, axis=-1, round_mode="rint", dst_type=torch_npu.float8_e4m3fn,
            block_size=32, scale_alg=1, dst_type_max=0.0)
        expected_y2, expected_mxscale2 = torch_npu.npu_grouped_dynamic_mx_quant(
            x, group_index, round_mode="rint", dst_type=torch_npu.float8_e4m3fn,
            blocksize=32, scale_alg=1)

        y1, mxscale1, y2, mxscale2 = self.npu_op_exec(x, group_index)

        assert torch.all(y1.cpu().view(torch.uint8) == expected_y1.cpu().view(torch.uint8))
        assert torch.all(mxscale1.cpu().view(torch.uint8) == expected_mxscale1.cpu().view(torch.uint8))
        assert torch.all(y2.cpu().view(torch.uint8) == expected_y2.cpu().view(torch.uint8))
        assert torch.all(mxscale2.cpu().view(torch.uint8) == expected_mxscale2.cpu().view(torch.uint8))


if __name__ == "__main__":
    run_tests()
