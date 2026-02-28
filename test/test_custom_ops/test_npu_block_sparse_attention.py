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

"""
npu_block_sparse_attention 正向算子单测。
参考 test_npu_sparse_flash_attention、test_fused_attention_ln_qkv：与 CPU 标杆对比，显存 <50G。
"""

import math
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

# 约束：q/kv_input_layout 仅 "TND"/"BNSD"；block_sparse_mask 必传；TND 时 actual_seq_lengths/actual_seq_lengths_kv 必选。
DTYPE = torch.float16
MIN_B, MIN_S, MIN_N, MIN_D = 1, 1, 2, 4
MIN_NUM_KV_HEADS = 2
MIN_S_LOW = 1


def _softmax_np(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x = x - x_max
    y = np.exp(x)
    return y / y.sum(axis=-1, keepdims=True)


def cpu_block_sparse_attention_bnsd(query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads):
    """
    CPU 标杆：BNSD layout，block_sparse_mask[b,n,i,j]=1 表示 q_block i 与 kv_block j 参与计算.
    query (B,N,S,D), key/value (B,N2,S,D), block_sparse_mask (B,N,ceil_q,ceil_kv).
    """
    q = query.cpu().to(torch.float32).numpy()
    k = key.cpu().to(torch.float32).numpy()
    v = value.cpu().to(torch.float32).numpy()
    mask = block_sparse_mask.cpu().numpy()
    B, N, S, D = q.shape
    N2 = k.shape[1]
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (S + block_x - 1) // block_x
    ceil_kv = (S + block_y - 1) // block_y
    out = np.zeros((B, N, S, D), dtype=np.float32)
    for b in range(B):
        for n in range(N):
            n2 = n % N2
            for s1 in range(S):
                i_block = s1 // block_x
                scores = np.full(S, -1e10, dtype=np.float32)
                for s2 in range(S):
                    j_block = s2 // block_y
                    if i_block < ceil_q and j_block < ceil_kv and mask[b, n, i_block, j_block] != 0:
                        scores[s2] = float(scale_value) * np.dot(q[b, n, s1, :], k[b, n2, s2, :])
                probs = _softmax_np(scores)
                out[b, n, s1, :] = np.dot(probs, v[b, n2, :, :])
    return torch.from_numpy(out).to(query.dtype)


def cpu_block_sparse_attention_tnd(query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads):
    """CPU 标杆：TND layout，batch=1，T 为总 token 数."""
    q = query.cpu().to(torch.float32).numpy()
    k = key.cpu().to(torch.float32).numpy()
    v = value.cpu().to(torch.float32).numpy()
    mask = block_sparse_mask.cpu().numpy()
    T, N, D = q.shape
    N2 = k.shape[1]
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (T + block_x - 1) // block_x
    ceil_kv = (T + block_y - 1) // block_y
    out = np.zeros((T, N, D), dtype=np.float32)
    for n in range(N):
        n2 = n % N2
        for s1 in range(T):
            i_block = s1 // block_x
            scores = np.full(T, -1e10, dtype=np.float32)
            for s2 in range(T):
                j_block = s2 // block_y
                if i_block < ceil_q and j_block < ceil_kv and mask[0, n, i_block, j_block] != 0:
                    scores[s2] = float(scale_value) * np.dot(q[s1, n, :], k[s2, n2, :])
            probs = _softmax_np(scores)
            out[s1, n, :] = np.dot(probs, v[:, n2, :])
    return torch.from_numpy(out).to(query.dtype)


class TestNPUBlockSparseAttention(TestCase):
    """Test npu_block_sparse_attention 正向，精度与 CPU 标杆一致，显存 <50G."""

    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_cpu_compare(self, device="npu"):
        """BNSD：NPU 与 CPU 标杆对比."""
        torch.npu.empty_cache()
        B, S, N, D = MIN_B, MIN_S_LOW, MIN_N, MIN_D
        num_kv_heads = MIN_NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = [1, 1]
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int32)

        cpu_out = cpu_block_sparse_attention_bnsd(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, _ = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask,
            "BNSD", "BNSD", num_key_value_heads=num_kv_heads,
            scale_value=scale_value, inner_precise=1,
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.01)
        del query, key, value, block_sparse_mask, npu_out
        torch.npu.empty_cache()

    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_with_optional_args_cpu_compare(self, device="npu"):
        """BNSD 带 block_shape/actual_seq_lengths：NPU 与 CPU 标杆对比."""
        torch.npu.empty_cache()
        B, S, N, D = MIN_B, MIN_S_LOW, MIN_N, MIN_D
        num_kv_heads = MIN_NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = [1, 1]
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int32)

        cpu_out = cpu_block_sparse_attention_bnsd(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, _ = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask,
            "BNSD", "BNSD", num_key_value_heads=num_kv_heads,
            scale_value=scale_value, inner_precise=1,
            block_shape=block_shape,
            actual_seq_lengths=[S], actual_seq_lengths_kv=[S],
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.01)
        del query, key, value, block_sparse_mask, npu_out
        torch.npu.empty_cache()

    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_tnd_cpu_compare(self, device="npu"):
        """TND：NPU 与 CPU 标杆对比，TND 必传 actual_seq_lengths/actual_seq_lengths_kv."""
        torch.npu.empty_cache()
        T, N, D = MIN_S_LOW, MIN_N, MIN_D
        num_kv_heads = MIN_NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = [1, 1]
        ceil_q = (T + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (T + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(T, N, D, dtype=DTYPE)
        key = torch.randn(T, num_kv_heads, D, dtype=DTYPE)
        value = torch.randn(T, num_kv_heads, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(1, N, ceil_q, ceil_kv, dtype=torch.int32)

        cpu_out = cpu_block_sparse_attention_tnd(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, _ = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask,
            "TND", "TND", num_key_value_heads=num_kv_heads,
            scale_value=scale_value, inner_precise=1,
            actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.01)
        del query, key, value, block_sparse_mask, npu_out
        torch.npu.empty_cache()


if __name__ == "__main__":
    run_tests()
