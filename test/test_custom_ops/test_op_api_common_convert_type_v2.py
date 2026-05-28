import os

import numpy as np

# ConvertTypeV2 is selected only by task queue v2. Set this before importing
# torch_npu so OptionsManager sees it before any op_api macro caches the value.
os.environ["TASK_QUEUE_ENABLE"] = "2"

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestOpApiCommonConvertTypeV2(TestCase):
    def mx_a8w4_gmm_golden(self, x, weight, antiquant_scale, pertoken_scale, group_list):
        output = []
        offset = 0
        for idx, group_size in enumerate(group_list):
            end = offset + group_size
            x_i = x[offset:end].to(torch.float32)
            scale_i = antiquant_scale[idx].to(torch.float32)
            weight_i = weight[idx].to(torch.float32) / 64
            pertoken_scale_i = pertoken_scale[offset:end].to(torch.float32)
            output.append(torch.matmul(x_i * pertoken_scale_i, weight_i * scale_i))
            offset = end
        return torch.cat(output, dim=0).to(torch.float16)

    @SupportedDevices(["Ascend950"])
    def test_mx_a8w4_grouped_matmul_weight_nz_task_queue_v2(self):
        torch.manual_seed(0)
        np.random.seed(0)

        m = 16
        k = 256
        n = 512
        expert_num = 2
        group_size = 32
        group_list_cpu = torch.tensor([8, 8], dtype=torch.int64)

        x = torch.ones(size=(m, k), dtype=torch.float8_e4m3fn)
        weight_nd = torch.ones(size=(expert_num, n, k), dtype=torch.float32)
        antiquant_scale = torch.randint(
            low=124, high=130, size=(expert_num, n, k // group_size), dtype=torch.uint8)
        pertoken_scale = torch.randint(low=124, high=130, size=(m, k // group_size), dtype=torch.uint8)

        cpu_antiquant_scale = 2 ** (antiquant_scale.to(torch.float64) - 127)
        cpu_antiquant_scale = torch.repeat_interleave(cpu_antiquant_scale, group_size, dim=2).transpose(-1, -2)
        cpu_pertoken_scale = 2 ** (pertoken_scale.to(torch.float64) - 127)
        cpu_pertoken_scale = torch.repeat_interleave(cpu_pertoken_scale, group_size, dim=1)

        expected = self.mx_a8w4_gmm_golden(
            x, weight_nd.transpose(-1, -2), cpu_antiquant_scale, cpu_pertoken_scale, group_list_cpu)

        weight_nz = torch_npu.npu_format_cast(weight_nd.npu(), 29, torch.float8_e4m3fn)
        weight_nz = torch_npu.npu_convert_weight_to_int4pack(weight_nz).transpose(-1, -2)

        output = torch_npu.npu_grouped_matmul(
            [x.npu()],
            [weight_nz],
            bias=None,
            scale=None,
            offset=None,
            antiquant_scale=[antiquant_scale.transpose(-1, -2).npu()],
            antiquant_offset=None,
            per_token_scale=[pertoken_scale.npu()],
            group_list=group_list_cpu.npu(),
            activation_input=None,
            activation_quant_scale=None,
            activation_quant_offset=None,
            split_item=3,
            group_type=0,
            group_list_type=1,
            act_type=0,
            output_dtype=torch.float16,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        )

        torch_npu.npu.synchronize()
        self.assertRtolEqual(expected.to(torch.float32).numpy(), output[0].to(torch.float32).cpu().numpy(), 0.001)


if __name__ == "__main__":
    run_tests()
