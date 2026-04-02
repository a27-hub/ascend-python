# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuKvQuantSparseFlashAttentionPioneer(TestCase):
    def custom_op_exec(self, query, key, value, sparse_indices, scale_value,
                       key_quant_mode, value_quant_mode, **kwargs):
        return torch_npu._npu_kv_quant_sparse_flash_attention_pioneer(
            query, key, value, sparse_indices, scale_value,
            key_quant_mode, value_quant_mode, **kwargs)

    @SupportedDevices(['Ascend910B'])
    def test_kv_quant_sparse_flash_attention_pioneer_bsnd_bf16(self, device="npu"):
        B, S1, Q_N, D = 1, 1, 8, 128
        S2, KV_N = 64, 1
        sparse_size = 2
        query = torch.randn(B, S1, Q_N, D, dtype=torch.bfloat16).to(device)
        key = torch.randint(-128, 127, (B, S2, KV_N, D), dtype=torch.int8).to(device)
        value = torch.randint(-128, 127, (B, S2, KV_N, D), dtype=torch.int8).to(device)
        sparse_indices = torch.zeros(B, S1, KV_N, sparse_size, dtype=torch.int32).to(device)

        out = self.custom_op_exec(query, key, value, sparse_indices, 0.08, 2, 2)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, torch.Size([B, S1, Q_N, D]))

    @SupportedDevices(['Ascend910B'])
    def test_kv_quant_sparse_flash_attention_pioneer_bsnd_mla(self, device="npu"):
        B, S1, Q_N, D = 1, 1, 8, 192
        S2, KV_N = 64, 1
        sparse_size = 2
        rope_head_dim = 64
        query = torch.randn(B, S1, Q_N, D, dtype=torch.bfloat16).to(device)
        key = torch.randint(-128, 127, (B, S2, KV_N, D), dtype=torch.int8).to(device)
        value = torch.randint(-128, 127, (B, S2, KV_N, D), dtype=torch.int8).to(device)
        sparse_indices = torch.zeros(B, S1, KV_N, sparse_size, dtype=torch.int32).to(device)

        out = self.custom_op_exec(query, key, value, sparse_indices, 0.08, 2, 2,
                                  attention_mode=2, rope_head_dim=rope_head_dim)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, torch.Size([B, S1, Q_N, D - rope_head_dim]))


if __name__ == "__main__":
    run_tests()
