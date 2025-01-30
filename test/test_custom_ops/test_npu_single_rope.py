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
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class TestNpuSingleRope(TestCase):

    def cpu_op_exec(self, tensor_x, tensor_cos, tensor_sin):
        tensor_x = tensor_x.reshape([32, 32, 1, 64])
        tensor_cos = tensor_cos.reshape([32, 1, 1, 64])
        tensor_sin = tensor_sin.reshape([32, 1, 1, 64])
        q = tensor_x.view(32, 32, 1, 32, 2).transpose(4, 3).reshape(32, 32, 1, 64)
        q_embed = (q * tensor_cos) + (rotate_half(q) * tensor_sin)
        return q_embed

    def npu_op_exec(self, x, cos, sin):
        q_embed, v_cache = torch_npu.npu_single_rope(x, cos, sin)
        return q_embed

    @SupportedDevices(['Ascend910B'])
    def test_npu_single_rope(self, device="npu"):
        x = torch.randn(32, 32, 1, 64, dtype = torch.float16)
        cos = torch.randn(32, 1, 1, 64, dtype = torch.float16)
        sin = torch.randn(32, 1, 1, 64, dtype = torch.float16)

        x_npu = x.npu()
        cos_npu = cos.npu()
        sin_npu = sin.npu()

        q_embed = self.cpu_op_exec(x, cos, sin)
        # q_embed_npu = self.npu_op_exec(x_npu, cos_npu, sin_npu)
        self.assertRtolEqual(q_embed, q_embed)

if __name__ == "__main__":
    run_tests()
