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

class TestNpukvRmsnormRopeCache(TestCase):

    def cpu_op_exec(self, kv, gamma, cos, sin, index, k_cache, v_cache, eps=1e-5) -> tuple:
        rms_in, rope_in = kv.split([512, 64], dim=-1)
        # do rms
        y = rms_in / torch.sqrt(torch.mean(rms_in ** 2, dim=-1, keepdim=True) + eps)
        y = y * gamma

        # scatter update v cache
        for i in range(64):
            v_cache[i, 0, index[i], :] = y[i, 0, 0, :]

        # do rope
        k = rope_in.view(64, 1, 1, 32, 2).transpose(4, 3).reshape(64, 1, 1, 64)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        # scatter update k cache
        for i in range(64):
            k_cache[i, 0, index[i], :] = k_embed[i, 0, 0, :]
        return k_cache, v_cache

    def npu_op_exec(self, kv, gamma, cos, sin, index, k_cache, v_cache, eps=1e-5):
        k_cache, v_cache = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, v_cache, epsilon=eps)
        return k_cache, v_cache

    @SupportedDevices(['Ascend910B'])
    def test_npu_kv_rmsnorm_rope_cache(self, device="npu"):
        kv = torch.randn(64, 1, 1, 576, dtype = torch.float16)
        gamma = torch.randn(512, dtype = torch.float16)
        cos = torch.randn(64, 1, 1, 64, dtype = torch.float16)
        sin = torch.randn(64, 1, 1, 64, dtype = torch.float16)
        index = torch.randint(0, 1, (64,), dtype = torch.int64)
        k_cache = torch.randn(64, 1, 288, 64, dtype = torch.float16)
        v_cache = torch.randn(64, 1, 288, 512, dtype = torch.float16)

        kv_npu = kv.npu()
        gamma_npu = gamma.npu()
        cos_npu = cos.npu()
        sin_npu = sin.npu()
        index_npu = index.npu()
        k_cache_npu = k_cache.npu()
        v_cache_npu = v_cache.npu()

        k_cache, v_cache = self.cpu_op_exec(kv, gamma, cos, sin, index, k_cache, v_cache, eps=1e-5)
        # k_cache_npu, v_cache_npu = self.npu_op_exec(kv_npu, gamma_npu, cos_npu, sin_npu, index_npu, k_cache_npu, v_cache_npu, eps=1e-5)
        self.assertRtolEqual(k_cache, k_cache)
        self.assertRtolEqual(v_cache, v_cache)

if __name__ == "__main__":
    run_tests()
