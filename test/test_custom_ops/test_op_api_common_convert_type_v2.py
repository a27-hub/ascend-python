import os

# ConvertTypeV2 is selected only by task queue v2. Set this before importing
# torch_npu so OptionsManager sees it before any op_api macro caches the value.
os.environ["TASK_QUEUE_ENABLE"] = "2"

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestOpApiCommonConvertTypeV2(TestCase):
    @SupportedDevices(["Ascend950"])
    def test_mxfp4_grouped_matmul_weight_nz_task_queue_v2(self):
        torch.manual_seed(0)

        group_list = torch.tensor([16], dtype=torch.int64).npu()
        x = torch.randint(2, 3, size=(16, 128), dtype=torch.int8).view(torch.uint8)
        weight = torch.randint(2, 3, size=(1, 256, 32), dtype=torch.int8).view(torch.uint8)
        scale = torch.randint(124, 130, size=(1, 32), dtype=torch.int8)

        output = torch_npu.npu_grouped_matmul(
            [x.npu()],
            [torch_npu.npu_format_cast(weight.npu(), 29)],
            bias=None,
            scale=[scale.npu()],
            offset=None,
            antiquant_scale=None,
            antiquant_offset=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            output_dtype=torch.float32,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        )

        torch_npu.npu.synchronize()
        self.assertEqual(output[0].shape, torch.Size([16, 64]))


if __name__ == "__main__":
    run_tests()
