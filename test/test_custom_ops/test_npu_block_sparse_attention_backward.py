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
npu_block_sparse_attention_backward 反向算子单测。
与正向解耦：attention_out、softmax_lse 由本文件内 CPU 正向标杆生成，仅反向在 NPU 执行并与 CPU 反向标杆对比。
形状与正向测试一致（B=2, S=32, N=8, D=64），便于稳定跑通。

约束：block_shape 若传则至少两元素且>0；TND 时 actual_seq_lengths/actual_seq_lengths_kv 必传；head_dim<=128；N1>=N2 且 N1 % N2==0。
"""

import unittest
import math
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

DTYPE = torch.float16
# 与正向一致：B=2, S=32, N=8, D=64, num_kv_heads=8, block_shape=[128,128]（blockY 须为 128 的倍数）
B, S, N, D = 2, 32, 8, 64
NUM_KV_HEADS = 8
BLOCK_SHAPE = [128, 128]


def _softmax_np(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x = x - x_max
    y = np.exp(x)
    return y / y.sum(axis=-1, keepdims=True)


def _logsumexp_np(scores):
    """对一行 scores 计算 log(sum(exp(scores)))，masked 位置为 -1e10 时仍稳定."""
    s = scores.astype(np.float32)
    m = np.max(s)
    if m <= -1e9:
        return m
    return m + np.log(np.sum(np.exp(s - m)))


def cpu_block_sparse_attention_bnsd_with_lse(query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads):
    """CPU 正向标杆：BNSD，返回 (attention_out, softmax_lse)，供反向测试生成入参，不依赖 NPU 正向."""
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
    lse = np.zeros((B, N, S, 1), dtype=np.float32)
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
                lse[b, n, s1, 0] = _logsumexp_np(scores)
                probs = _softmax_np(scores)
                out[b, n, s1, :] = np.dot(probs, v[b, n2, :, :])
    attention_out = torch.from_numpy(out).to(query.dtype)
    softmax_lse = torch.from_numpy(lse).to(torch.float32)
    return attention_out, softmax_lse


def cpu_block_sparse_attention_tnd_with_lse(query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads):
    """CPU 正向标杆：TND，返回 (attention_out, softmax_lse)，供反向测试生成入参."""
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
    lse = np.zeros((T, N, 1), dtype=np.float32)
    for n in range(N):
        n2 = n % N2
        for s1 in range(T):
            i_block = s1 // block_x
            scores = np.full(T, -1e10, dtype=np.float32)
            for s2 in range(T):
                j_block = s2 // block_y
                if i_block < ceil_q and j_block < ceil_kv and mask[0, n, i_block, j_block] != 0:
                    scores[s2] = float(scale_value) * np.dot(q[s1, n, :], k[s2, n2, :])
            lse[s1, n, 0] = _logsumexp_np(scores)
            probs = _softmax_np(scores)
            out[s1, n, :] = np.dot(probs, v[:, n2, :])
    attention_out = torch.from_numpy(out).to(query.dtype)
    softmax_lse = torch.from_numpy(lse).to(torch.float32)
    return attention_out, softmax_lse


def _softmax_backward_np(dP, P):
    """dP 与 P 同 shape，返回 softmax 反向的 dS."""
    dP = dP.astype(np.float32)
    P = P.astype(np.float32)
    sum_dp_p = (dP * P).sum(axis=-1, keepdims=True)
    return dP * P - P * sum_dp_p


def cpu_block_sparse_attention_backward_bnsd(
    query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads
):
    """CPU 标杆：BNSD backward，与 forward 相同 mask/scale 重算 P 再算 dQ,dK,dV."""
    q = query.cpu().to(torch.float32).numpy()
    k = key.cpu().to(torch.float32).numpy()
    v = value.cpu().to(torch.float32).numpy()
    dout = d_out.cpu().to(torch.float32).numpy()
    mask = block_sparse_mask.cpu().numpy()
    B, N, S, D = q.shape
    N2 = k.shape[1]
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (S + block_x - 1) // block_x
    ceil_kv = (S + block_y - 1) // block_y
    scale = float(scale_value)

    d_query = np.zeros_like(q, dtype=np.float32)
    d_key = np.zeros_like(k, dtype=np.float32)
    d_value = np.zeros_like(v, dtype=np.float32)

    for b in range(B):
        for n in range(N):
            n2 = n % N2
            P = np.zeros((S, S), dtype=np.float32)
            for s1 in range(S):
                i_block = s1 // block_x
                scores = np.full(S, -1e10, dtype=np.float32)
                for s2 in range(S):
                    j_block = s2 // block_y
                    if i_block < ceil_q and j_block < ceil_kv and mask[b, n, i_block, j_block] != 0:
                        scores[s2] = scale * np.dot(q[b, n, s1, :], k[b, n2, s2, :])
                P[s1, :] = _softmax_np(scores).ravel()

            dP = np.zeros((S, S), dtype=np.float32)
            for s1 in range(S):
                for s2 in range(S):
                    dP[s1, s2] = np.dot(dout[b, n, s1, :], v[b, n2, s2, :])
            dS = _softmax_backward_np(dP, P) * scale

            for s1 in range(S):
                d_query[b, n, s1, :] = np.dot(dS[s1, :], k[b, n2, :, :])
            for s2 in range(S):
                d_key[b, n2, s2, :] += np.dot(dS[:, s2], q[b, n, :, :])
            for s2 in range(S):
                d_value[b, n2, s2, :] = np.dot(P[:, s2], dout[b, n, :, :])

    return (
        torch.from_numpy(d_query).to(query.dtype),
        torch.from_numpy(d_key).to(key.dtype),
        torch.from_numpy(d_value).to(value.dtype),
    )


def cpu_block_sparse_attention_backward_tnd(
    query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads
):
    """CPU 标杆：TND backward，batch=1."""
    q = query.cpu().to(torch.float32).numpy()
    k = key.cpu().to(torch.float32).numpy()
    v = value.cpu().to(torch.float32).numpy()
    dout = d_out.cpu().to(torch.float32).numpy()
    mask = block_sparse_mask.cpu().numpy()
    T, N, D = q.shape
    N2 = k.shape[1]
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (T + block_x - 1) // block_x
    ceil_kv = (T + block_y - 1) // block_y
    scale = float(scale_value)

    d_query = np.zeros_like(q, dtype=np.float32)
    d_key = np.zeros_like(k, dtype=np.float32)
    d_value = np.zeros_like(v, dtype=np.float32)

    for n in range(N):
        n2 = n % N2
        P = np.zeros((T, T), dtype=np.float32)
        for s1 in range(T):
            i_block = s1 // block_x
            scores = np.full(T, -1e10, dtype=np.float32)
            for s2 in range(T):
                j_block = s2 // block_y
                if i_block < ceil_q and j_block < ceil_kv and mask[0, n, i_block, j_block] != 0:
                    scores[s2] = scale * np.dot(q[s1, n, :], k[s2, n2, :])
            P[s1, :] = _softmax_np(scores).ravel()

        dP = np.zeros((T, T), dtype=np.float32)
        for s1 in range(T):
            for s2 in range(T):
                dP[s1, s2] = np.dot(dout[s1, n, :], v[s2, n2, :])
        dS = _softmax_backward_np(dP, P) * scale

        for s1 in range(T):
            d_query[s1, n, :] = np.dot(dS[s1, :], k[:, n2, :])
        for s2 in range(T):
            d_key[s2, n2, :] += np.dot(dS[:, s2], q[:, n, :])
        for s2 in range(T):
            d_value[s2, n2, :] = np.dot(P[:, s2], dout[:, n, :])

    return (
        torch.from_numpy(d_query).to(query.dtype),
        torch.from_numpy(d_key).to(key.dtype),
        torch.from_numpy(d_value).to(value.dtype),
    )


class TestNPUBlockSparseAttentionBackward(TestCase):
    """Test npu_block_sparse_attention_backward，与 CPU 反向标杆对比."""

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_backward_bnsd_cpu_compare(self, device="npu"):
        """Backward BNSD：与 CPU 反向标杆对比，统一使用 BLOCK_SHAPE；attention_out/softmax_lse 由 CPU 正向标杆生成."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        d_out = torch.randn(B, N, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int32)

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_bnsd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)
        attention_out_cpu, softmax_lse_cpu = cpu_block_sparse_attention_bnsd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)
        attention_out = attention_out_cpu.to(device)
        softmax_lse_out = softmax_lse_cpu.to(device)

        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value, attention_out, softmax_lse_out, block_sparse_mask,
            block_shape=block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )

        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.02)
        del query, key, value, d_out, block_sparse_mask, attention_out, softmax_lse_out
        del d_query, d_key, d_value
        torch.npu.empty_cache()

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_backward_bnsd_with_optional_args_cpu_compare(self, device="npu"):
        """Backward BNSD 显式传 block_shape/actual_seq_lengths；attention_out/softmax_lse 由 CPU 正向标杆生成."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        d_out = torch.randn(B, N, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int32)

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_bnsd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)
        attention_out_cpu, softmax_lse_cpu = cpu_block_sparse_attention_bnsd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)
        attention_out = attention_out_cpu.to(device)
        softmax_lse_out = softmax_lse_cpu.to(device)

        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value, attention_out, softmax_lse_out, block_sparse_mask,
            block_shape=block_shape, actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )

        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.02)
        del query, key, value, d_out, block_sparse_mask, attention_out, softmax_lse_out
        del d_query, d_key, d_value
        torch.npu.empty_cache()

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_backward_tnd_cpu_compare(self, device="npu"):
        """Backward TND：T=S，必传 actual_seq_lengths/actual_seq_lengths_kv；attention_out/softmax_lse 由 CPU 正向标杆生成."""
        torch.npu.empty_cache()
        T = S
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (T + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (T + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(T, N, D, dtype=DTYPE)
        key = torch.randn(T, num_kv_heads, D, dtype=DTYPE)
        value = torch.randn(T, num_kv_heads, D, dtype=DTYPE)
        d_out = torch.randn(T, N, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(1, N, ceil_q, ceil_kv, dtype=torch.int32)

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_tnd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)
        attention_out_cpu, softmax_lse_cpu = cpu_block_sparse_attention_tnd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)
        attention_out = attention_out_cpu.to(device)
        softmax_lse_out = softmax_lse_cpu.to(device)

        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value, attention_out, softmax_lse_out, block_sparse_mask,
            block_shape=block_shape,
            actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
            q_input_layout="TND", kv_input_layout="TND",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )

        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.02)
        del query, key, value, d_out, block_sparse_mask, attention_out, softmax_lse_out
        del d_query, d_key, d_value
        torch.npu.empty_cache()


if __name__ == "__main__":
    run_tests()
