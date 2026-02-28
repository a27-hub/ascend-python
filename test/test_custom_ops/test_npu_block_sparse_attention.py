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

import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

# 接口约束：q/kv_input_layout 仅支持 "TND"/"BNSD"；query/key/value 同 dtype；
# block_sparse_mask 必传，shape 为 [batch, headNum, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKVS, blockShapeY)]；
# TND 时 actual_seq_lengths/actual_seq_lengths_kv 必选；inner_precise 仅支持 0 或 1。
# 显存控制在 55G 内：Backward 用例使用较小规模。
DTYPE = torch.float16
MIN_B, MIN_S, MIN_N, MIN_D = 1, 2, 2, 4
MIN_NUM_KV_HEADS = 2
MIN_S_LOW = 1
# Backward 用例规模（控制显存 ~55G）
BWD_B, BWD_S, BWD_N, BWD_D = 1, 2, 2, 4
BWD_NUM_KV_HEADS = 2
BWD_T = 2


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
    G = N // N2
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
    # TND: (T, N, D) -> 视为 batch=1, S=T
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


def _softmax_backward_np(dP, P):
    """dP 与 P 同 shape，返回 softmax 反向的 dS."""
    dP = dP.astype(np.float32)
    P = P.astype(np.float32)
    sum_dp_p = (dP * P).sum(axis=-1, keepdims=True)
    return dP * P - P * sum_dp_p


def cpu_block_sparse_attention_backward_bnsd(
    query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads
):
    """
    CPU 标杆：BNSD backward. 用与 forward 相同的 mask 与 scale 重算 P，再算 dQ,dK,dV.
    """
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
            # 重算 P (S, S)
            P = np.zeros((S, S), dtype=np.float32)
            for s1 in range(S):
                i_block = s1 // block_x
                scores = np.full(S, -1e10, dtype=np.float32)
                for s2 in range(S):
                    j_block = s2 // block_y
                    if i_block < ceil_q and j_block < ceil_kv and mask[b, n, i_block, j_block] != 0:
                        scores[s2] = scale * np.dot(q[b, n, s1, :], k[b, n2, s2, :])
                P[s1, :] = _softmax_np(scores).ravel()

            # dP[s1,s2] = d_out[b,n,s1,:] . v[b,n2,s2,:]
            dP = np.zeros((S, S), dtype=np.float32)
            for s1 in range(S):
                for s2 in range(S):
                    dP[s1, s2] = np.dot(dout[b, n, s1, :], v[b, n2, s2, :])
            dS = _softmax_backward_np(dP, P) * scale

            # dQ[b,n,s1,:] = sum_s2 dS[s1,s2] * K[b,n2,s2,:]
            for s1 in range(S):
                d_query[b, n, s1, :] = np.dot(dS[s1, :], k[b, n2, :, :])
            # dK[b,n2,s2,:] += sum_s1 dS[s1,s2] * Q[b,n,s1,:]
            for s2 in range(S):
                d_key[b, n2, s2, :] += np.dot(dS[:, s2], q[b, n, :, :])
            # dV[b,n2,s2,:] = sum_s1 P[s1,s2] * d_out[b,n,s1,:]
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


class TestNPUBlockSparseAttention(TestCase):
    """Test npu_block_sparse_attention forward，精度与 CPU 标杆一致."""

    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_cpu_compare(self, device="npu"):
        """BNSD 4D：NPU 与 CPU 标杆对比."""
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

        npu_out = npu_out.cpu().to(torch.float32)
        cpu_out = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out, npu_out, prec=0.005, prec16=0.01)

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

        npu_out = npu_out.cpu().to(torch.float32)
        cpu_out = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out, npu_out, prec=0.005, prec16=0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_tnd_cpu_compare(self, device="npu"):
        """TND 3D：NPU 与 CPU 标杆对比，TND 必传 actual_seq_lengths/actual_seq_lengths_kv."""
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

        npu_out = npu_out.cpu().to(torch.float32)
        cpu_out = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out, npu_out, prec=0.005, prec16=0.01)


class TestNPUBlockSparseAttentionBackward(TestCase):
    """Test npu_block_sparse_attention_backward，精度与 CPU 标杆一致，显存控制在 55G 内."""

    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_backward_bnsd_cpu_compare(self, device="npu"):
        """Backward BNSD：NPU 与 CPU 标杆对比，符合约束，显存 ~55G 内."""
        torch.npu.empty_cache()
        B, S, N, D = BWD_B, BWD_S, BWD_N, BWD_D
        num_kv_heads = BWD_NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = [1, 1]
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        d_out = torch.randn(B, N, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int32)

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_bnsd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        attention_out, softmax_lse_out = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask,
            "BNSD", "BNSD", num_key_value_heads=num_kv_heads,
            scale_value=scale_value, inner_precise=1,
        )
        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value, attention_out, softmax_lse_out, block_sparse_mask,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )

        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.02)

    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_backward_bnsd_with_optional_args_cpu_compare(self, device="npu"):
        """Backward BNSD 带 block_shape/actual_seq：NPU 与 CPU 标杆对比，显存 ~55G 内."""
        torch.npu.empty_cache()
        B, S, N, D = BWD_B, BWD_S, BWD_N, BWD_D
        num_kv_heads = BWD_NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = [1, 1]
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        d_out = torch.randn(B, N, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int32)

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_bnsd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        attention_out, softmax_lse_out = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask,
            "BNSD", "BNSD", num_key_value_heads=num_kv_heads,
            scale_value=scale_value, inner_precise=1,
            block_shape=block_shape, actual_seq_lengths=[S], actual_seq_lengths_kv=[S],
        )
        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value, attention_out, softmax_lse_out, block_sparse_mask,
            block_shape=block_shape, actual_seq_lengths=[S], actual_seq_lengths_kv=[S],
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )

        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.02)

    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_backward_tnd_cpu_compare(self, device="npu"):
        """Backward TND：TND 必传 actual_seq_lengths/actual_seq_lengths_kv，NPU 与 CPU 标杆对比，显存 ~55G 内."""
        torch.npu.empty_cache()
        T, N, D = BWD_T, BWD_N, BWD_D
        num_kv_heads = BWD_NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = [1, 1]
        ceil_q = (T + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (T + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(T, N, D, dtype=DTYPE)
        key = torch.randn(T, num_kv_heads, D, dtype=DTYPE)
        value = torch.randn(T, num_kv_heads, D, dtype=DTYPE)
        d_out = torch.randn(T, N, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(1, N, ceil_q, ceil_kv, dtype=torch.int32)

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_tnd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        attention_out, softmax_lse_out = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask,
            "TND", "TND", num_key_value_heads=num_kv_heads,
            scale_value=scale_value, inner_precise=1,
            actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
        )
        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value, attention_out, softmax_lse_out, block_sparse_mask,
            actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
            q_input_layout="TND", kv_input_layout="TND",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )

        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.02)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.02)


if __name__ == "__main__":
    run_tests()
