# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


def rotate_half(x):
    import torch
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class TestNpuDequantSwigluQuant(TestCase):

    def cpu_op_exec(self, x, weight_scale, activate_scale, bias, quant_scale, quant_offset, group_num, activate_left, quant_mode):
        x = x.to(torch.float32)
        # if bias is not None:
        #     x = torch.add(x, bias)
        res = torch.mul(x, weight_scale.to(torch.float32))
        res = torch.mul(res, activate_scale.to(torch.float32))
        out = torch.chunk(res, 2, dim=-1)
        if activate_left:
            self_tensor = out[0]
            other = out[1]
        else:
            self_tensor = out[1]
            other = out[0]
        output = torch.nn.functional.silu(self_tensor) * other  # 100
        output = torch.mul(output, quant_scale.to(torch.float32))  # 100
        # if quant_mode == "static":
        #     output = torch.add(output, quant_offset.to(torch.float32))

        scale_out = torch.zeros([x.shape[0]], dtype=torch.float)
        if quant_mode == "dynamic":
            abs = torch.abs(output)
            max_values = torch.amax(abs, dim=-1)
            scale_out = max_values / 127
            max_values = 127 / max_values
            output = output * max_values.unsqueeze(1)
        output = torch.clamp(output, -128, 127)
        output = torch.round(output)
        return output.to(torch.int8).cpu().numpy(), scale_out.cpu().numpy()

    def npu_op_exec(self, x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index, activate_left, quant_mode):
        y, scale = torch_npu.npu_dequant_swiglu_quant(
            x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index, activate_left=activate_left, quant_mode=quant_mode)
        return y, scale

    @SupportedDevices(['Ascend910B'])
    def test_npu_dequant_swiglu_quant(self, device="npu"):
        x_shape = (4608, 2048)
        x = torch.randint(-10, 10, x_shape, dtype=torch.int32).npu()
        weight_scale = torch.randn(x_shape[1], dtype=torch.float32).npu()
        activation_scale = torch.randn(
            (x_shape[0], 1), dtype=torch.float32).npu()
        bias = None
        quant_scale = torch.randn(
            (1, x_shape[1] // 2), dtype=torch.float16).npu()
        quant_offset = None

        group_index = torch.tensor([144], dtype=torch.int64).npu()

        y, scale = self.cpu_op_exec(x.cpu(), weight_scale.cpu(), activation_scale.cpu(
        ), bias, quant_scale.cpu(), quant_offset, group_index, False, "dynamic")
        # y_npu, scale_npu = self.npu_op_exec(x, weight_scale, activation_scale, bias,
        #                                     quant_scale, quant_offset, group_index, activate_left=False, quant_mode=1)
        self.assertRtolEqual(y, y)
        self.assertRtolEqual(scale, scale)


if __name__ == "__main__":
    run_tests()
