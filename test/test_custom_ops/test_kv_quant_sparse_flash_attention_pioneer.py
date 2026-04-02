# Copyright (c) 2025 Huawei Technologies Co., Ltd
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


class TestKvQuantSparseFlashAttentionPioneer(TestCase):
    def custom_op_exec(self, query, key, value, sparse_indices, scale_value,
                       key_quant_mode, value_quant_mode, **kwargs):
        return torch_npu._npu_kv_quant_sparse_flash_attention_pioneer(
            query, key, value, sparse_indices, scale_value,
            key_quant_mode, value_quant_mode, **kwargs)

    def generate_input(self, shape, dtype):
        if dtype == "float16":
            return torch.randn(shape, dtype=torch.float16)
        elif dtype == "bfloat16":
            return torch.randn(shape, dtype=torch.bfloat16)
        return torch.randn(shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_kv_quant_sparse_flash_attention_pioneer_bsnd(self, device="npu"):
        # BSND layout test: [B, S1, Q_N, D]
        # Output D = query.D - rope_head_dim
        rope_head_dim = 64
        query_d = 128
        B, S1, Q_N = 2, 8, 4
        S2 = 16

        query = self.generate_input((B, S1, Q_N, query_d), "float16").to(device)
        key = torch.randint(-128, 127, (B, S2, 1, 64)).to(device).to(torch.int8)
        value = torch.randint(-128, 127, (B, S2, 1, 64)).to(device).to(torch.int8)
        sparse_indices = torch.randint(0, S2, (B, S1, 1, 2)).to(device).to(torch.int32)

        output = self.custom_op_exec(
            query, key, value, sparse_indices,
            scale_value=0.125,
            key_quant_mode=2,
            value_quant_mode=2,
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=3,
            rope_head_dim=rope_head_dim
        )

        # Output shape: D dimension excludes rope part
        expected_shape = (B, S1, Q_N, query_d - rope_head_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, torch.bfloat16)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_kv_quant_sparse_flash_attention_pioneer_tnd(self, device="npu"):
        # TND layout test: [Q_T, Q_N, D]
        # Output D = query.D - rope_head_dim
        rope_head_dim = 32
        query_d = 64
        Q_T, Q_N = 8, 4
        T2, KV_N = 16, 1

        query = self.generate_input((Q_T, Q_N, query_d), "float16").to(device)
        key = torch.randint(-128, 127, (T2, KV_N, 64)).to(device).to(torch.int8)
        value = torch.randint(-128, 127, (T2, KV_N, 64)).to(device).to(torch.int8)
        sparse_indices = torch.randint(0, T2, (Q_T, KV_N, 2)).to(device).to(torch.int32)
        actual_seq_lengths_query = torch.tensor([Q_T], dtype=torch.int32).to(device)

        output = self.custom_op_exec(
            query, key, value, sparse_indices,
            scale_value=0.125,
            key_quant_mode=2,
            value_quant_mode=2,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            actual_seq_lengths_query=actual_seq_lengths_query,
            rope_head_dim=rope_head_dim
        )

        # Output shape: D dimension excludes rope part
        expected_shape = (Q_T, Q_N, query_d - rope_head_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, torch.bfloat16)


if __name__ == "__main__":
    run_tests()
