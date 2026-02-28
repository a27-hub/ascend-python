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
与 CPU 标杆对比，形状适中（B=2, S=32, N=8, D=64）便于稳定跑通。

约束符合说明：
- q_input_layout/kv_input_layout 仅 "TND"/"BNSD"；block_sparse_mask 必传。
- query/key/value 统一 float16；N1>=N2 且 N1 % N2==0；block_sparse_mask 与 block_shape 对应；inner_precise 仅 0 或 1；TND 时 actual_seq_lengths/actual_seq_lengths_kv 必传。

输入格式说明：aclnn 要求 query/key/value/block_sparse_mask 为 ND 格式。torch.randn().to(device) 在 NPU 上
可能得到 NCL（3D）或 NCHW（4D）等默认格式；算子 C++ 层已在调用 aclnn 前统一做 npu_format_cast(·, ACL_FORMAT_ND)，
因此本测试无需在 Python 侧显式转 ND，直接 .to(device) 即可。
"""

import unittest
import math
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

DTYPE = torch.float16
# 合理测试形状：B=2, S=32, N=8, D=64, num_kv_heads=8，block_shape=[8,8] -> 4x4 块
B, S, N, D = 2, 32, 8, 64
NUM_KV_HEADS = 8
BLOCK_SHAPE = [8, 8]


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
    """Test npu_block_sparse_attention 正向，与 CPU 标杆对比."""

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_cpu_compare(self, device="npu"):
        """BNSD：NPU 与 CPU 标杆对比，统一使用 BLOCK_SHAPE."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
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
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.01)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_with_optional_args_cpu_compare(self, device="npu"):
        """BNSD 显式传 block_shape/actual_seq_lengths：NPU 与 CPU 标杆对比."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
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
            actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.01)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_tnd_cpu_compare(self, device="npu"):
        """TND：NPU 与 CPU 标杆对比，必传 actual_seq_lengths/actual_seq_lengths_kv."""
        torch.npu.empty_cache()
        T = S  # TND 总 token 数，与 BNSD 的 S 一致便于对比
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
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
            block_shape=BLOCK_SHAPE,
            actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.01)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_autograd_backward(self, device="npu"):
        """验证 derivatives.yaml 自动前反向绑定：对 forward 输出调用 .backward()，query/key/value 能收到梯度且与显式 backward 一致."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE, device=device)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE, device=device)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE, device=device)
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int32, device=device)

        attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask,
            "BNSD", "BNSD", num_key_value_heads=num_kv_heads,
            scale_value=scale_value, inner_precise=1,
            block_shape=block_shape,
        )
        grad_out = torch.ones_like(attention_out, device=device)
        attention_out.backward(gradient=grad_out)

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)

        d_query_autograd = query.grad.clone()
        d_key_autograd = key.grad.clone()
        d_value_autograd = value.grad.clone()

        query.grad = None
        key.grad = None
        value.grad = None
        d_explicit = torch_npu.npu_block_sparse_attention_backward(
            grad_out, query, key, value, attention_out.detach(), softmax_lse.detach(), block_sparse_mask,
            block_shape=block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )
        d_query_explicit, d_key_explicit, d_value_explicit = d_explicit

        self.assertRtolEqual(d_query_autograd.cpu().float(), d_query_explicit.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(d_key_autograd.cpu().float(), d_key_explicit.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(d_value_autograd.cpu().float(), d_value_explicit.cpu().float(), prec=0.01, prec16=0.02)

        del query, key, value, block_sparse_mask, attention_out, softmax_lse, grad_out
        del d_query_autograd, d_key_autograd, d_value_autograd, d_query_explicit, d_key_explicit, d_value_explicit
        torch.npu.empty_cache()


if __name__ == "__main__":
    run_tests()
