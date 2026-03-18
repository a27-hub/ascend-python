import math
import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


class TestGroupedDynamicBlockQuant(TestCase):
    def custom_op_exec(self, x, group_list, min_scale=0.0, round_mode="rint",
                       dst_type=291, row_block_size=1, col_block_size=128, group_list_type=0):
        return torch_npu.npu_grouped_dynamic_block_quant(
            x, group_list, min_scale=min_scale, round_mode=round_mode,
            dst_type=dst_type, row_block_size=row_block_size,
            col_block_size=col_block_size, group_list_type=group_list_type)

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_dynamic_block_quant_2d_fp16(self, device="npu"):
        x = torch.randn(64, 256, dtype=torch.float16, device=device)
        group_list = torch.tensor([32, 64], dtype=torch.int32, device=device)
        y, scale = self.custom_op_exec(x, group_list)
        # y shape 应与 x 一致
        self.assertEqual(y.shape, x.shape)
        # scale shape: [64/1 + 2, ceil(256/128)] = [66, 2]
        expected_scale_shape = (64 // 1 + 2, math.ceil(256 / 128))
        self.assertEqual(scale.shape, torch.Size(expected_scale_shape))
        self.assertEqual(scale.dtype, torch.float32)

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_dynamic_block_quant_3d_bf16(self, device="npu"):
        x = torch.randn(2, 64, 256, dtype=torch.bfloat16, device=device)
        group_list = torch.tensor([32, 64], dtype=torch.int32, device=device)
        y, scale = self.custom_op_exec(x, group_list)
        self.assertEqual(y.shape, x.shape)
        # scale shape: [2, 64/1 + 2, ceil(256/128)] = [2, 66, 2]
        expected_scale_shape = (2, 64 // 1 + 2, math.ceil(256 / 128))
        self.assertEqual(scale.shape, torch.Size(expected_scale_shape))
        self.assertEqual(scale.dtype, torch.float32)

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_dynamic_block_quant_row_block(self, device="npu"):
        x = torch.randn(64, 256, dtype=torch.float16, device=device)
        group_list = torch.tensor([16, 32], dtype=torch.int32, device=device)
        row_block_size = 2
        y, scale = self.custom_op_exec(x, group_list, row_block_size=row_block_size)
        self.assertEqual(y.shape, x.shape)
        # scale shape: [64/2 + 2, ceil(256/128)] = [34, 2]
        expected_scale_shape = (64 // row_block_size + 2, math.ceil(256 / 128))
        self.assertEqual(scale.shape, torch.Size(expected_scale_shape))


if __name__ == "__main__":
    run_tests()
