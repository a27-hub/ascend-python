"""
attention_pioneer 算子精度验证脚本
场景: fp16, TND_NTD, PA, sparseMode=4 (band), Sink Token, RoPE (MLA decode)
      非连续 KV cache (as_strided, k_nope 与 k_rope 共享底层内存)
      多 batch 不同 S_kv
      支持三种 PA cache 布局: BBH / BNBD / PANZ

三种布局对比 (底层 k_base 内存布局完全相同, 只是 view 不同):
  BBH  (3D): (total_blocks, block_size, H)                    H = N_kv * D
  BNBD (4D): (total_blocks, N_kv, block_size, D)
  PANZ (5D): (total_blocks, N_kv, D//d0, block_size, d0)      d0 = 16

非连续内存布局 (k_base 中每个 block):
  ┌──────────────────────────────┬─────────────────────┐
  │  k_nope (block_size × D)     │  k_rope (bs × D_r)  │
  │  128 × 512 = 65536 元素      │  128 × 64 = 8192    │
  └──────────────────────────────┴─────────────────────┘
  page_head_dim = D + D_rope = 576  (k_rope 恰好填满, 无额外 padding)
  stride[0] = block_size * N_kv * page_head_dim = 73728 (非连续)

关键: sink 与 actualSeqLengths 的关系
  - actual_seq_lengths_kv: 传给 NPU 的值, 不含 sink
  - block_table 计算: 用原始 S_kv (不含 sink), 因为 PA cache 只存正常 KV
  - golden 计算: 每个 batch 的 KV 序列 = sink_len + S_kv[b]

用法:
  python test_attention_pioneer_sink_pa_rope.py --layout BBH
  python test_attention_pioneer_sink_pa_rope.py --layout BNBD
  python test_attention_pioneer_sink_pa_rope.py --layout PANZ
"""

import argparse
import math
import torch
import numpy as np
import torch_npu


# ===================== 命令行参数 =====================
parser = argparse.ArgumentParser(description="attention_pioneer 非连续 KV cache 精度验证")
parser.add_argument("--layout", type=str, default="BBH", choices=["BBH", "BNBD", "PANZ"],
                    help="PA cache 布局: BBH(3D) / BNBD(4D) / PANZ(5D)")
args = parser.parse_args()
CACHE_LAYOUT = args.layout


# ===================== 可配置参数 =====================
B = 4                   # batch size
N_q = 32                # query heads (num_heads)
N_kv = 1                # kv heads (num_key_value_heads), GQA
D = 512                 # head dim (nope 部分), TND_NTD 要求 D=512
D_rope = 64             # rope dim, MLA 要求 D_rope=64
SINK_LEN = 128          # sink token 数量
BLOCK_SIZE = 128        # PA block 大小
PRE_TOKENS = 511        # band 窗口
NEXT_TOKENS = 0
SCALE = 1.0 / math.sqrt(D + D_rope)  # = 1/24
DTYPE = torch.float16
SPARSE_MODE = 4

# 每个 batch 不同的 Q 长度和 KV 长度
S_q_list = [15, 22, 23, 25]        # 每个 batch 的 query seq len
S_kv_list = [512, 1024, 1536, 2048] # 每个 batch 的 KV 长度 (不含 sink)
MAX_S_Q = max(S_q_list)             # 用于 BNSD 预分配
MAX_S_KV = max(S_kv_list)           # 用于 PA cache 预分配

# TND 参数
T = sum(S_q_list)       # total query tokens = 85

# 非连续参数
PADDING_PER_HEAD = 64
PAGE_HEAD_DIM = D + PADDING_PER_HEAD    # 576
H = N_kv * D                            # 512
H_rope = N_kv * D_rope                  # 64

# PANZ 专用
D0SIZE = 16                             # NZ fractal 基本单位
D1_K = D // D0SIZE                      # 32
D1_KR = D_rope // D0SIZE                # 4


# ===================== 辅助函数 =====================
def make_block_table_varied(B, S_kv_list, block_size):
    """为每个 batch 生成 block_table (不同 S_kv)"""
    max_blocks_per_seq = max((s + block_size - 1) // block_size for s in S_kv_list)
    block_table = torch.full((B, max_blocks_per_seq), -1, dtype=torch.int32)
    blocks_per_batch = []
    phys_offset = 0
    for b in range(B):
        n_blocks = (S_kv_list[b] + block_size - 1) // block_size
        blocks_per_batch.append(n_blocks)
        for i in range(n_blocks):
            block_table[b, i] = phys_offset + i
        phys_offset += max_blocks_per_seq
    total_blocks = B * max_blocks_per_seq
    return block_table, max_blocks_per_seq, total_blocks, blocks_per_batch


def create_noncontiguous_kv_cache(total_blocks, block_size, N_kv, D, D_rope,
                                   page_head_dim, dtype, layout):
    """
    创建非连续 KV cache: k_nope 和 k_rope 共享 k_base
    根据 layout 返回不同 shape 的 view, 底层内存布局相同
    """
    base_size = total_blocks * block_size * N_kv * page_head_dim
    k_base = torch.zeros(base_size, dtype=dtype)
    v_base = torch.zeros(base_size, dtype=dtype)

    stride_0 = block_size * N_kv * page_head_dim  # 73728
    kr_offset = N_kv * block_size * D              # 65536

    if layout == "BBH":
        # (total_blocks, block_size, N_kv*D)
        k_cache = torch.as_strided(k_base,
            size=[total_blocks, block_size, N_kv * D],
            stride=[stride_0, D * N_kv, 1], storage_offset=0)
        k_rope_cache = torch.as_strided(k_base,
            size=[total_blocks, block_size, N_kv * D_rope],
            stride=[stride_0, D_rope * N_kv, 1], storage_offset=kr_offset)
        v_cache = torch.as_strided(v_base,
            size=[total_blocks, block_size, N_kv * D],
            stride=[stride_0, D * N_kv, 1], storage_offset=0)

    elif layout == "BNBD":
        # (total_blocks, N_kv, block_size, D)
        k_cache = torch.as_strided(k_base,
            size=[total_blocks, N_kv, block_size, D],
            stride=[stride_0, block_size * D, D, 1], storage_offset=0)
        k_rope_cache = torch.as_strided(k_base,
            size=[total_blocks, N_kv, block_size, D_rope],
            stride=[stride_0, block_size * D_rope, D_rope, 1], storage_offset=kr_offset)
        v_cache = torch.as_strided(v_base,
            size=[total_blocks, N_kv, block_size, D],
            stride=[stride_0, block_size * D, D, 1], storage_offset=0)

    elif layout == "PANZ":
        # (total_blocks, N_kv, D//d0, block_size, d0)
        d0 = D0SIZE
        d1_k = D // d0
        d1_kr = D_rope // d0
        k_cache = torch.as_strided(k_base,
            size=[total_blocks, N_kv, d1_k, block_size, d0],
            stride=[stride_0, block_size * D, block_size * d0, d0, 1], storage_offset=0)
        k_rope_cache = torch.as_strided(k_base,
            size=[total_blocks, N_kv, d1_kr, block_size, d0],
            stride=[stride_0, block_size * D_rope, block_size * d0, d0, 1], storage_offset=kr_offset)
        v_cache = torch.as_strided(v_base,
            size=[total_blocks, N_kv, d1_k, block_size, d0],
            stride=[stride_0, block_size * D, block_size * d0, d0, 1], storage_offset=0)

    return k_base, k_cache, k_rope_cache, v_base, v_cache


def _bnsd_to_cache(data, layout, d0size=D0SIZE):
    """
    将 (N_kv, length, dim) 数据转换为 cache 写入格式
    BBH:  -> (length, N_kv*dim)
    BNBD: -> (N_kv, length, dim)   (不变)
    PANZ: -> (N_kv, dim//d0, length, d0)
    """
    if layout == "BBH":
        N, L, dim = data.shape
        return data.permute(1, 0, 2).reshape(L, N * dim)
    elif layout == "BNBD":
        return data
    elif layout == "PANZ":
        N, L, dim = data.shape
        return data.reshape(N, L, dim // d0size, d0size).permute(0, 2, 1, 3)


def _cache_to_bnsd(data, layout, N_kv, dim, d0size=D0SIZE):
    """
    将 cache 读取的数据转换回 (N_kv, length, dim) 格式
    BBH:  (length, N_kv*dim) -> (N_kv, length, dim)
    BNBD: (N_kv, length, dim) -> 不变
    PANZ: (N_kv, dim//d0, length, d0) -> (N_kv, length, dim)
    """
    if layout == "BBH":
        L = data.shape[0]
        return data.reshape(L, N_kv, dim).permute(1, 0, 2)
    elif layout == "BNBD":
        return data
    elif layout == "PANZ":
        N, d1, L, d0 = data.shape
        return data.permute(0, 2, 1, 3).reshape(N, L, dim)


def _write_block(cache, phys_blk, length, data, layout):
    """将数据写入指定物理 block"""
    if layout == "BBH":
        cache[phys_blk, :length, :] = data
    elif layout == "BNBD":
        cache[phys_blk, :, :length, :] = data
    elif layout == "PANZ":
        cache[phys_blk, :, :, :length, :] = data


def _read_block(cache, phys_blk, length, layout):
    """从指定物理 block 读取数据"""
    if layout == "BBH":
        return cache[phys_blk, :length, :]
    elif layout == "BNBD":
        return cache[phys_blk, :, :length, :]
    elif layout == "PANZ":
        return cache[phys_blk, :, :, :length, :]


def fill_noncontiguous_cache_varied(k_cache, k_rope_cache, v_cache,
                                    key_data, key_rope_data, value_data,
                                    block_table, block_size, N_kv, D, D_rope,
                                    S_kv_list, layout):
    """将 BNSD 数据填入非连续 cache, 支持三种布局"""
    B_sz = key_data.shape[0]
    for b in range(B_sz):
        S_kv_b = S_kv_list[b]
        n_blocks_b = (S_kv_b + block_size - 1) // block_size
        for blk_i in range(n_blocks_b):
            phys_blk = block_table[b, blk_i].item()
            seq_start = blk_i * block_size
            seq_end = min(seq_start + block_size, S_kv_b)
            length = seq_end - seq_start
            if length <= 0:
                continue
            # 取 BNSD 切片: (N_kv, length, dim)
            k_slice = key_data[b, :, seq_start:seq_end, :]
            kr_slice = key_rope_data[b, :, seq_start:seq_end, :]
            v_slice = value_data[b, :, seq_start:seq_end, :]
            # 转换并写入
            _write_block(k_cache, phys_blk, length, _bnsd_to_cache(k_slice, layout), layout)
            _write_block(k_rope_cache, phys_blk, length, _bnsd_to_cache(kr_slice, layout), layout)
            _write_block(v_cache, phys_blk, length, _bnsd_to_cache(v_slice, layout), layout)


def verify_noncontiguous_cache_varied(k_cache, k_rope_cache, v_cache,
                                      key_data, key_rope_data, value_data,
                                      block_table, block_size, N_kv, D, D_rope,
                                      S_kv_list, layout):
    """CPU 验证: 从非连续 cache 读回数据并与原始 BNSD 比较"""
    B_sz = key_data.shape[0]
    errors = {"k": 0, "k_rope": 0, "v": 0}
    total_checks = 0
    for b in range(B_sz):
        S_kv_b = S_kv_list[b]
        n_blocks_b = (S_kv_b + block_size - 1) // block_size
        for blk_i in range(n_blocks_b):
            phys_blk = block_table[b, blk_i].item()
            seq_start = blk_i * block_size
            seq_end = min(seq_start + block_size, S_kv_b)
            length = seq_end - seq_start
            if length <= 0:
                continue
            total_checks += 1
            k_read = _cache_to_bnsd(_read_block(k_cache, phys_blk, length, layout), layout, N_kv, D)
            kr_read = _cache_to_bnsd(_read_block(k_rope_cache, phys_blk, length, layout), layout, N_kv, D_rope)
            v_read = _cache_to_bnsd(_read_block(v_cache, phys_blk, length, layout), layout, N_kv, D)
            if not torch.equal(k_read, key_data[b, :, seq_start:seq_end, :]):
                errors["k"] += 1
            if not torch.equal(kr_read, key_rope_data[b, :, seq_start:seq_end, :]):
                errors["k_rope"] += 1
            if not torch.equal(v_read, value_data[b, :, seq_start:seq_end, :]):
                errors["v"] += 1
    return errors, total_checks


def repeat_kv(x, n_rep):
    """GQA: (B, N_kv, S, D) -> (B, N_q, S, D)"""
    if n_rep == 1:
        return x
    B, N_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, N_kv, n_rep, S, D).reshape(B, N_kv * n_rep, S, D)


def generate_band_mask(S_q, S_kv_total, pre_tokens, next_tokens):
    """band mask, True = 被 mask"""
    mask = torch.ones(S_q, S_kv_total, dtype=torch.bool)
    for q_pos in range(S_q):
        q_global = S_kv_total - S_q + q_pos
        low = max(q_global - pre_tokens, 0)
        high = min(q_global + next_tokens, S_kv_total - 1)
        mask[q_pos, low:high + 1] = False
    return mask


def golden_attention_mla(
    query_nope, query_rope, key_nope, key_rope, value,
    key_sink, key_rope_sink, value_sink,
    S_q_list, S_kv_list, pre_tokens, next_tokens, scale, N_q, N_kv, sink_len,
):
    """Golden reference: MLA Attention, per-batch 不同 S_q/S_kv + Sink + Band Mask"""
    B_l = query_nope.shape[0]
    D_v = value.shape[3]
    D_total = query_nope.shape[3] + query_rope.shape[3]
    n_rep = N_q // N_kv

    query_nope = query_nope.float()
    query_rope = query_rope.float()
    key_nope = key_nope.float()
    key_rope = key_rope.float()
    value = value.float()
    key_sink_f = key_sink.float()
    key_rope_sink_f = key_rope_sink.float()
    value_sink_f = value_sink.float()

    k_sink_combined = torch.cat([key_sink_f, key_rope_sink_f], dim=-1)
    k_sink_bnsd = k_sink_combined.unsqueeze(0).permute(0, 2, 1, 3)
    k_sink_bnsd_rep = repeat_kv(k_sink_bnsd.expand(1, -1, -1, -1), n_rep)

    v_sink_bnsd = value_sink_f[:, :, :D_v].unsqueeze(0).permute(0, 2, 1, 3)
    v_sink_bnsd_rep = repeat_kv(v_sink_bnsd.expand(1, -1, -1, -1), n_rep)

    # 每个 batch 输出 shape 不同, 收集后拼接
    output_list = []

    for b in range(B_l):
        S_q_b = S_q_list[b]
        S_kv_b = S_kv_list[b]
        S_kv_total = sink_len + S_kv_b

        # Q: 只取该 batch 的有效 S_q 个 token
        Q_b = torch.cat([query_nope[b:b+1, :, :S_q_b, :],
                         query_rope[b:b+1, :, :S_q_b, :]], dim=-1)  # (1, N_q, S_q_b, D+D_rope)

        K_b = torch.cat([key_nope[b:b+1, :, :S_kv_b, :],
                         key_rope[b:b+1, :, :S_kv_b, :]], dim=-1)
        V_b = value[b:b+1, :, :S_kv_b, :]

        K_b = repeat_kv(K_b, n_rep)
        V_b = repeat_kv(V_b, n_rep)

        K_full = torch.cat([k_sink_bnsd_rep, K_b], dim=2)
        V_full = torch.cat([v_sink_bnsd_rep, V_b], dim=2)

        score = torch.matmul(Q_b, K_full.transpose(2, 3)) * scale

        band_mask = generate_band_mask(S_q_b, S_kv_total, pre_tokens, next_tokens)
        mask_4d = band_mask.unsqueeze(0).unsqueeze(0).expand(1, N_q, S_q_b, S_kv_total)

        if sink_len > 0:
            is_invalid_row = torch.all(mask_4d[:, :, :, sink_len:] == True, dim=-1, keepdim=True)
            sink_mask = torch.zeros(mask_4d[:, :, :, :sink_len].shape, dtype=torch.bool)
            sink_mask[is_invalid_row.expand_as(sink_mask)] = True
            mask_4d = mask_4d.clone()
            mask_4d[:, :, :, :sink_len] = sink_mask

        score = score.masked_fill(mask_4d, -1.7e38)

        x_max = score.max(dim=-1, keepdim=True)[0]
        exp_vals = torch.exp(score - x_max)
        exp_sum = exp_vals.sum(dim=-1, keepdim=True)
        softmax_res = (exp_vals / exp_sum).to(torch.float16).float()

        out_b = torch.matmul(softmax_res, V_full)  # (1, N_q, S_q_b, D)
        output_list.append(out_b)

    return output_list


def generate_atten_mask_2048():
    """sparseMode=4: (2048, 2048) 上三角 mask"""
    return torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1)


def create_npu_cache(k_base_npu, v_base_npu, total_blocks, block_size,
                     N_kv, D, D_rope, page_head_dim, layout):
    """在 NPU 上创建与 CPU 相同布局的非连续 cache view"""
    stride_0 = block_size * N_kv * page_head_dim
    kr_offset = N_kv * block_size * D

    if layout == "BBH":
        k_cache = torch.as_strided(k_base_npu,
            size=[total_blocks, block_size, N_kv * D],
            stride=[stride_0, D * N_kv, 1], storage_offset=0)
        k_rope_cache = torch.as_strided(k_base_npu,
            size=[total_blocks, block_size, N_kv * D_rope],
            stride=[stride_0, D_rope * N_kv, 1], storage_offset=kr_offset)
        v_cache = torch.as_strided(v_base_npu,
            size=[total_blocks, block_size, N_kv * D],
            stride=[stride_0, D * N_kv, 1], storage_offset=0)

    elif layout == "BNBD":
        k_cache = torch.as_strided(k_base_npu,
            size=[total_blocks, N_kv, block_size, D],
            stride=[stride_0, block_size * D, D, 1], storage_offset=0)
        k_rope_cache = torch.as_strided(k_base_npu,
            size=[total_blocks, N_kv, block_size, D_rope],
            stride=[stride_0, block_size * D_rope, D_rope, 1], storage_offset=kr_offset)
        v_cache = torch.as_strided(v_base_npu,
            size=[total_blocks, N_kv, block_size, D],
            stride=[stride_0, block_size * D, D, 1], storage_offset=0)

    elif layout == "PANZ":
        d0 = D0SIZE
        d1_k = D // d0
        d1_kr = D_rope // d0
        k_cache = torch.as_strided(k_base_npu,
            size=[total_blocks, N_kv, d1_k, block_size, d0],
            stride=[stride_0, block_size * D, block_size * d0, d0, 1], storage_offset=0)
        k_rope_cache = torch.as_strided(k_base_npu,
            size=[total_blocks, N_kv, d1_kr, block_size, d0],
            stride=[stride_0, block_size * D_rope, block_size * d0, d0, 1], storage_offset=kr_offset)
        v_cache = torch.as_strided(v_base_npu,
            size=[total_blocks, N_kv, d1_k, block_size, d0],
            stride=[stride_0, block_size * D, block_size * d0, d0, 1], storage_offset=0)

    return k_cache, k_rope_cache, v_cache


# ===================== 生成输入数据 =====================
print("=" * 60)
print(f"attention_pioneer 精度验证 [{CACHE_LAYOUT}]")
print(f"  场景: fp16 / TND_NTD / PA({CACHE_LAYOUT}) / sparseMode=4 / Sink / RoPE (MLA)")
print(f"  特性: 非连续 KV cache + 每 batch 不同 S_q/S_kv")
print(f"  B={B}, N_q={N_q}, N_kv={N_kv}")
print(f"  S_q_list={S_q_list}  (每个 batch 的 Q 长度)")
print(f"  S_kv_list={S_kv_list}  (每个 batch 的 KV 长度, 不含 sink)")
print(f"  D={D}, D_rope={D_rope}, SINK_LEN={SINK_LEN}")
print(f"  BLOCK_SIZE={BLOCK_SIZE}, T={T}")
print(f"  PRE_TOKENS={PRE_TOKENS}, NEXT_TOKENS={NEXT_TOKENS}")
print(f"  SCALE={SCALE:.6f}")
print(f"  PAGE_HEAD_DIM={PAGE_HEAD_DIM} (D+padding={D}+{PADDING_PER_HEAD})")
if CACHE_LAYOUT == "PANZ":
    print(f"  D0SIZE={D0SIZE}, D1_K={D1_K}, D1_KR={D1_KR}")
print("=" * 60)

torch.manual_seed(42)

# ---- 生成原始 BNSD 数据 ----
# Q: 用 MAX_S_Q 预分配, 每个 batch 只用前 S_q_list[b] 个 token
query_bnsd = torch.randn(B, N_q, MAX_S_Q, D, dtype=DTYPE)
query_rope_bnsd = torch.randn(B, N_q, MAX_S_Q, D_rope, dtype=DTYPE)
key_data = torch.randn(B, N_kv, MAX_S_KV, D, dtype=DTYPE)
value_data = torch.randn(B, N_kv, MAX_S_KV, D, dtype=DTYPE)
key_rope_data = torch.randn(B, N_kv, MAX_S_KV, D_rope, dtype=DTYPE)

# Sink tensors: 3D (SINK_LEN, N_kv, D)
key_sink = torch.randn(SINK_LEN, N_kv, D, dtype=DTYPE)
key_rope_sink = torch.randn(SINK_LEN, N_kv, D_rope, dtype=DTYPE)
value_sink = torch.randn(SINK_LEN, N_kv, D, dtype=DTYPE)

# ---- 转 TND 格式 (拼接每个 batch 的有效 token) ----
query_tnd_parts = []
query_rope_tnd_parts = []
for b in range(B):
    sq = S_q_list[b]
    query_tnd_parts.append(
        query_bnsd[b, :, :sq, :].permute(1, 0, 2).reshape(sq, N_q, D))
    query_rope_tnd_parts.append(
        query_rope_bnsd[b, :, :sq, :].permute(1, 0, 2).reshape(sq, N_q, D_rope))
query_tnd = torch.cat(query_tnd_parts, dim=0).contiguous()
query_rope_tnd = torch.cat(query_rope_tnd_parts, dim=0).contiguous()

# ---- Block table (不含 sink, 用原始 S_kv 计算) ----
block_table, max_blocks_per_seq, total_blocks, blocks_per_batch = \
    make_block_table_varied(B, S_kv_list, BLOCK_SIZE)

print(f"\n  Block table (用原始 S_kv, 不含 sink):")
print(f"    max_blocks_per_seq={max_blocks_per_seq}, total_blocks={total_blocks}")
print(f"    blocks_per_batch={blocks_per_batch}")
for b in range(B):
    valid = block_table[b, :blocks_per_batch[b]].tolist()
    print(f"    batch {b}: S_kv={S_kv_list[b]}, blocks={blocks_per_batch[b]}, "
          f"phys_ids={valid}")

# ===================== 创建非连续 KV cache =====================
print(f"\n创建非连续 KV cache (layout={CACHE_LAYOUT})...")
k_base, k_cache, k_rope_cache, v_base, v_cache = create_noncontiguous_kv_cache(
    total_blocks, BLOCK_SIZE, N_kv, D, D_rope, PAGE_HEAD_DIM, DTYPE, CACHE_LAYOUT)

print(f"  k_cache  shape={k_cache.shape}, stride={k_cache.stride()}, contiguous={k_cache.is_contiguous()}")
print(f"  k_rope   shape={k_rope_cache.shape}, stride={k_rope_cache.stride()}, contiguous={k_rope_cache.is_contiguous()}")
print(f"  v_cache  shape={v_cache.shape}, stride={v_cache.stride()}, contiguous={v_cache.is_contiguous()}")

# ---- 填入数据 ----
fill_noncontiguous_cache_varied(
    k_cache, k_rope_cache, v_cache,
    key_data, key_rope_data, value_data,
    block_table, BLOCK_SIZE, N_kv, D, D_rope, S_kv_list, CACHE_LAYOUT)

# ===================== CPU 验证: 非连续 cache 数据完整性 =====================
print(f"\nCPU 验证: 非连续 cache 数据完整性...")
errors, total_checks = verify_noncontiguous_cache_varied(
    k_cache, k_rope_cache, v_cache,
    key_data, key_rope_data, value_data,
    block_table, BLOCK_SIZE, N_kv, D, D_rope, S_kv_list, CACHE_LAYOUT)

print(f"  k_cache  读回: {total_checks - errors['k']}/{total_checks} PASS")
print(f"  k_rope   读回: {total_checks - errors['k_rope']}/{total_checks} PASS")
print(f"  v_cache  读回: {total_checks - errors['v']}/{total_checks} PASS")
all_pass = all(v == 0 for v in errors.values())
print(f"  {'ALL PASS' if all_pass else 'FAIL'}")
if not all_pass:
    raise RuntimeError("非连续 cache 验证失败!")

# ---- 验证 k_nope/k_rope 内存不互相覆盖 (batch 0, block 0) ----
phys_0 = block_table[0, 0].item()
offset_0 = phys_0 * BLOCK_SIZE * N_kv * PAGE_HEAD_DIM
blk0_k = k_base[offset_0 : offset_0 + BLOCK_SIZE * D]
blk0_kr = k_base[offset_0 + BLOCK_SIZE * D : offset_0 + BLOCK_SIZE * (D + D_rope)]
if CACHE_LAYOUT == "PANZ":
    # PANZ (NZ fractal): D 拆为 (d1, d0), 内存布局为 (N_kv, d1, block_size, d0)
    k_orig = key_data[0, :, :BLOCK_SIZE, :].reshape(N_kv, BLOCK_SIZE, D1_K, D0SIZE) \
             .permute(0, 2, 1, 3).contiguous().reshape(-1)
    kr_orig = key_rope_data[0, :, :BLOCK_SIZE, :].reshape(N_kv, BLOCK_SIZE, D1_KR, D0SIZE) \
              .permute(0, 2, 1, 3).contiguous().reshape(-1)
else:
    # BBH/BNBD: 内存布局为 (N_kv, block_size, D), 即 BSH 顺序
    k_orig = key_data[0, :, :BLOCK_SIZE, :].permute(1, 0, 2).reshape(-1)
    kr_orig = key_rope_data[0, :, :BLOCK_SIZE, :].permute(1, 0, 2).reshape(-1)
print(f"  k_base 内存: k_nope 正确={torch.equal(blk0_k, k_orig)}, "
      f"k_rope 正确={torch.equal(blk0_kr, kr_orig)}")

# ===================== 其他输入 =====================
# actual_seq_lengths (Q): 累加格式
import itertools
actual_seq_lengths = list(itertools.accumulate(S_q_list))

# actual_seq_lengths_kv: per-batch, 不含 sink (NPU 自己处理 sink)
actual_seq_lengths_kv = list(S_kv_list)
atten_mask = generate_atten_mask_2048()

print(f"\n  actual_seq_lengths (Q, 累加): {actual_seq_lengths}")
print(f"  actual_seq_lengths_kv (不含 sink): {actual_seq_lengths_kv}")
print(f"  block_table shape: {block_table.shape}")
print(f"  atten_mask shape: {atten_mask.shape}")

# ===================== Golden Reference =====================
print("\n计算 golden reference (CPU float32)...")
print(f"  每个 batch:")
for b in range(B):
    print(f"    batch {b}: S_q={S_q_list[b]}, S_kv={S_kv_list[b]} + sink={SINK_LEN} = {S_kv_list[b] + SINK_LEN}")

golden_list = golden_attention_mla(
    query_bnsd.cpu(), query_rope_bnsd.cpu(),
    key_data.cpu(), key_rope_data.cpu(), value_data.cpu(),
    key_sink.cpu(), key_rope_sink.cpu(), value_sink.cpu(),
    S_q_list, S_kv_list,
    PRE_TOKENS, NEXT_TOKENS, SCALE, N_q, N_kv, SINK_LEN,
)

# golden_list: [(1, N_q, S_q_b, D), ...] -> NTD (N_q, T, D)
golden_ntd_parts = []
for b in range(B):
    # (1, N_q, S_q_b, D) -> (N_q, S_q_b, D)
    golden_ntd_parts.append(golden_list[b].squeeze(0))
golden_ntd = torch.cat(golden_ntd_parts, dim=1).contiguous().to(DTYPE)  # (N_q, T, D)

print(f"  Golden output shape (NTD): {golden_ntd.shape}")
print(f"  Golden output[0, 0, :5]: {golden_ntd[0, 0, :5]}")

# ===================== NPU 算子调用 =====================
print(f"\n将非连续 cache 传输到 NPU (layout={CACHE_LAYOUT})...")

k_base_npu = k_base.npu()
v_base_npu = v_base.npu()

k_cache_npu, k_rope_cache_npu, v_cache_npu = create_npu_cache(
    k_base_npu, v_base_npu, total_blocks, BLOCK_SIZE,
    N_kv, D, D_rope, PAGE_HEAD_DIM, CACHE_LAYOUT)

print(f"  k_cache_npu  contiguous={k_cache_npu.is_contiguous()}, stride={k_cache_npu.stride()}")
print(f"  k_rope_npu   contiguous={k_rope_cache_npu.is_contiguous()}, stride={k_rope_cache_npu.stride()}")
print(f"  v_cache_npu  contiguous={v_cache_npu.is_contiguous()}, stride={v_cache_npu.stride()}")

print(f"\n调用 NPU attention_pioneer 算子 (非连续 {CACHE_LAYOUT} cache)...")

npu_output, softmax_lse = torch_npu._npu_attention_pioneer(
    query_tnd.npu(),
    k_cache_npu,
    v_cache_npu,
    atten_mask=atten_mask.npu(),
    actual_seq_lengths=actual_seq_lengths,
    actual_seq_lengths_kv=actual_seq_lengths_kv,
    block_table=block_table.npu(),
    query_rope=query_rope_tnd.npu(),
    key_rope=k_rope_cache_npu,
    key_sink=key_sink.npu(),
    key_rope_sink=key_rope_sink.npu(),
    value_sink=value_sink.npu(),
    num_heads=N_q,
    scale=SCALE,
    pre_tokens=PRE_TOKENS,
    next_tokens=NEXT_TOKENS,
    input_layout="TND_NTD",
    num_key_value_heads=N_kv,
    sparse_mode=SPARSE_MODE,
    block_size=BLOCK_SIZE,
    softmax_lse_flag=False,
)

npu_result = npu_output.cpu()
print(f"  NPU output shape (NTD): {npu_result.shape}")
print(f"  NPU output[0, 0, :5]: {npu_result[0, 0, :5]}")

# ===================== 精度比较 =====================
print("\n" + "=" * 60)
print(f"精度比较 [{CACHE_LAYOUT}]:")
print("=" * 60)

golden_fp32 = golden_ntd.float()
npu_fp32 = npu_result.float()

abs_diff = (golden_fp32 - npu_fp32).abs()
max_abs_diff = abs_diff.max().item()
mean_abs_diff = abs_diff.mean().item()

rel_diff = abs_diff / (golden_fp32.abs() + 1e-8)
max_rel_diff = rel_diff.max().item()
mean_rel_diff = rel_diff.mean().item()

print(f"  最大绝对误差: {max_abs_diff:.6e}")
print(f"  平均绝对误差: {mean_abs_diff:.6e}")
print(f"  最大相对误差: {max_rel_diff:.6e}")
print(f"  平均相对误差: {mean_rel_diff:.6e}")

atol = 1e-2
rtol = 1e-2
is_close = torch.allclose(golden_fp32, npu_fp32, atol=atol, rtol=rtol)
print(f"\n  torch.allclose(atol={atol}, rtol={rtol}): {'PASS' if is_close else 'FAIL'}")

close_mask = (abs_diff <= atol + rtol * golden_fp32.abs())
num_close = close_mask.sum().item()
total_elem = close_mask.numel()
pass_rate = num_close / total_elem * 100
print(f"  逐元素通过率: {num_close}/{total_elem} ({pass_rate:.2f}%)")

atol_loose = 5e-2
rtol_loose = 5e-2
is_close_loose = torch.allclose(golden_fp32, npu_fp32, atol=atol_loose, rtol=rtol_loose)
print(f"  torch.allclose(atol={atol_loose}, rtol={rtol_loose}): {'PASS' if is_close_loose else 'FAIL'}")

if not is_close:
    flat_idx = abs_diff.argmax().item()
    idx = np.unravel_index(flat_idx, abs_diff.shape)
    print(f"\n  误差最大位置: {idx}")
    print(f"    Golden: {golden_fp32[idx].item():.6f}")
    print(f"    NPU:    {npu_fp32[idx].item():.6f}")
    print(f"    Diff:   {abs_diff[idx].item():.6e}")

# ---- Per-batch 精度 ----
print(f"\n  Per-batch 精度:")
t_offset = 0
for b in range(B):
    sq = S_q_list[b]
    g_b = golden_fp32[:, t_offset:t_offset + sq, :]
    n_b = npu_fp32[:, t_offset:t_offset + sq, :]
    diff_b = (g_b - n_b).abs()
    print(f"    batch {b} (S_q={sq}, S_kv={S_kv_list[b]}+sink={SINK_LEN}={S_kv_list[b]+SINK_LEN}): "
          f"max_diff={diff_b.max().item():.6e}, "
          f"allclose={torch.allclose(g_b, n_b, atol=atol, rtol=rtol)}")
    t_offset += sq

print("\n" + "=" * 60)
if is_close:
    print(f"结论: [{CACHE_LAYOUT}] PASS")
else:
    print(f"结论: [{CACHE_LAYOUT}] FAIL")
    if is_close_loose:
        print("  (宽松精度下通过, 可能是 fp16 累积误差)")
print("=" * 60)
