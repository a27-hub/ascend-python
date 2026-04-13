import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
import unittest
from typing import Optional
import torch.nn.functional as f
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.set_device(2)

torch.set_printoptions(precision=16)
# torch.set_printoptions(threshold=float('inf'))  # 显示所有元素

relative_tol = 2 ** (-8)
# relative_tol = 5e-3
# relative_tol = 1e-5
absolute_tol = 2 ** (-8)
error_tol = 1e-4

def print_tensor_info(tensor, tensor_name):
    """封装打印函数，统一格式"""
    print(f"\n=== {tensor_name} 张量信息 ===")
    print(f"张量形状: {tensor.shape}")  # 打印形状
    # 打印前10个值（展平后）
    print(f"前10个值: {tensor.flatten()[:10].cpu().numpy()}")
    # 打印后10个值（展平后）
    print(f"后10个值: {tensor.flatten()[-10:].cpu().numpy()}")

def verify_result2(output, golden):
    output=output.reshape(-1).npu()
    # output=output.reshape(-1).to(torch.float64).cpu()
    golden=golden.reshape(-1).npu()
    # golden=golden.reshape(-1).to(torch.float64).cpu()
    different_element_results = torch.isclose(output,
                                           golden,
                                           rtol=relative_tol,
                                           atol=absolute_tol,
                                           equal_nan=True).npu()
    different_element_indexes = torch.where(different_element_results == False)[0]

    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data)
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data))

        if index == 10:
            block_sparse_mask
    # print(golden.size())
    # print(golden.size())
    error_ratio = float(different_element_indexes.shape[0]) / (golden.size()[0])
    print("错误率: %.6f, 精度: %.6f" % (error_ratio, error_tol))
    print("错误个数: ", different_element_indexes.shape[0], '/', golden.reshape(-1).size()[0])
    # return error_ratio <= error_tol
    return error_ratio <= error_tol, 0 if different_element_indexes.shape[0] == 0 else 1

def verify_result(output, golden):
    output=output.reshape(-1).to(torch.float32).cpu()
    # output=output.reshape(-1).to(torch.float64).cpu()
    golden=golden.reshape(-1).to(torch.float32).cpu()
    # golden=golden.reshape(-1).to(torch.float64).cpu()
    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=relative_tol,
                                           atol=absolute_tol,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]

    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data)
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data))

        if index == 10:
            block_sparse_mask
    # print(golden.size())
    # print(golden.size())
    error_ratio = float(different_element_indexes.size) / (golden.size()[0])
    print("错误率: %.6f, 精度: %.6f" % (error_ratio, error_tol))
    print("错误个数: ", different_element_indexes.size, '/', golden.reshape(-1).size()[0])
    return error_ratio <= error_tol, different_element_indexes.size == 0

def get_chunk(
    input,  # tensor of shape (S, ...)
    C,      # chunk size
    start   # chunk start position
):
    S = input.shape[0]
    end = start + c
    if end <= S:
        return input[start:end]
    else:
        pad_size = end - S
        if len(input.shape) > 1:
            # for q, k, v
            return F.pad(input[start:], (0, 0, 0, pad_size))
        else:
            return F.pad(input[start:], (0, pad_size))

def stage1_chunk(
    query,  # (C, Dk)
    key,    # (C, Dk)
    value,  # (C, Dv)
    g,      # (C,)
    beta,   # (C,)
    scale
):
    device = query.device
    C = query.shape[0]

    # bf16 @ bf16 -> bf16
    kkt = (key @ key.transpose(-1, -2)) 
    kkt = kkt.float()  # (C, C)
    kkt = kkt * beta.float().unsqueeze(-1)    # (C, Dk)
    # kkt = k_beta @ key.transpose(-1, -2)  # (C, C)

    qkt = (query @ key.transpose(-1, -2))    # (C, C)

    g_cum = g.cumsum(dim=-1)  # (C,)
    g_cum_exp = g_cum.exp()   # (C,)

    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=-1) # fp32
    attn = (g_cum_exp[:, None] / g_cum_exp[None, :]) * lower   # (C, C)

    attn_1 = kkt * attn   # (C, C)
    attn_1 *= -1.0
    for i in range(1, C):
        row = attn_1[i, :i].clone()
        sub = attn_1[:i, :i].clone()
        attn_1[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn_1 = attn_1 + torch.eye(C, dtype=attn_1.dtype, device=attn_1.device)    # now finished inverse
    attn_1 = attn_1.to(torch.bfloat16)

    kg = key.float() * (g_cum_exp[-1, None] / g_cum_exp)[..., None]  # (C, Dk)
    k_cumdecay = (beta.unsqueeze(-1).float() * g_cum_exp[:, None]) * (-1) * key.float()      # (C, Dk)

    # fp32 * fp32 -> bf16
    v_beta = (value.float() * beta.unsqueeze(-1).float()).to(torch.bfloat16)  # (C, Dv)

    # k_cumdecay = -1.0 * k_beta * g_cum_exp[:, None]      # (C, Dk)
    q_prime = query.float() * scale * g_cum_exp[:, None]       # (C, Dk)

    v_inner = attn_1.to(torch.bfloat16) @ v_beta.to(torch.bfloat16)    # (C, Dv)
    k_cumdecay = attn_1.to(torch.bfloat16) @ k_cumdecay.to(torch.bfloat16)        # (C, Dk)

    return g_cum_exp, k_cumdecay, v_inner, q_prime.to(torch.bfloat16), kg.to(torch.bfloat16), qkt

def stage1(
    query,  # (S, Nk, Dk)
    key,    # (S, Nk, Dk)
    value,  # (S, Nv, Dv)
    g,      # (S, Nv)
    beta,   # (S, Nv)
    scale,
    C       # chunk size
):
    S, Nk, Dk = query.shape
    _, Nv, Dv = value.shape
    device=query.device

    # query = query.to(torch.float32)
    # key = key.to(torch.float32)
    # value = value.to(torch.float32)
    # beta = beta.to(torch.float32)

    if Nv // Nk > 1:
        query = query.repeat_interleave(Nv // Nk, dim=1)
        key = key.repeat_interleave(Nv // Nk, dim=1)

    padded_seq_len = (S + C - 1) // C * C

    g_cum_exp = torch.zeros((Nv, padded_seq_len), dtype=torch.float32, device=device)
    # k_cumdecay = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.float32, device=device)
    # v_inner = torch.zeros((Nv, padded_seq_len, Dv), dtype=torch.float32, device=device)
    # q_prime = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.float32, device=device)
    # kg = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.float32, device=device)
    # qkt = torch.zeros((Nv, padded_seq_len, C), dtype=torch.float32, device=device)
    k_cumdecay = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    v_inner = torch.zeros((Nv, padded_seq_len, Dv), dtype=torch.bfloat16, device=device)
    q_prime = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    kg = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    qkt = torch.zeros((Nv, padded_seq_len, C), dtype=torch.bfloat16, device=device)

    loop_range = range(0, padded_seq_len, C)
    for nid in range(Nv):
        for idx in reversed(loop_range):     # use reverse loop to simulate parallel
            q_chunk = get_chunk(query[:, nid, :], C, idx)
            k_chunk = get_chunk(key[:, nid, :], C, idx)
            v_chunk = get_chunk(value[:, nid, :], C, idx)
            g_chunk = get_chunk(g[:, nid], C, idx)
            beta_chunk = get_chunk(beta[:, nid], C, idx)

            g_cum_exp_chunk, k_cumdecay_chunk, v_inner_chunk, qg_chunk, kg_chunk, qkt_chunk = stage1_chunk(
                q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, scale)

            g_cum_exp[nid, idx:idx + C] = g_cum_exp_chunk
            k_cumdecay[nid, idx:idx + C, :] = k_cumdecay_chunk
            v_inner[nid, idx:idx + C, :] = v_inner_chunk
            q_prime[nid, idx:idx + C, :] = qg_chunk
            kg[nid, idx:idx + C, :] = kg_chunk
            qkt[nid, idx:idx + C, :] = qkt_chunk
    return g_cum_exp, k_cumdecay, v_inner, q_prime, kg, qkt

def stage2_chunk(
    q_prime,    # (C, Dk)
    v_inner,    # (C, Dv)
    g_cum_exp,  # (C)
    k_cumdecay, # (C, Dk)
    state,      # (Dv, Dk)
    kg,         # (C, Dk)
):
    # C1
    # bf16 @ bf16 -> fp32
    attn_inter = (q_prime @ state.to(torch.bfloat16).transpose(0, 1))            # (C, Dv)
    # attn_inter = q_prime.float() @ state.to(torch.bfloat16).transpose(0, 1).float()            # (C, Dv)
    # bf16 @ bf16 -> fp32
    v_prime = (k_cumdecay @ state.to(torch.bfloat16).transpose(0, 1))           # (C, Dv)

    # v_prime = k_cumdecay.float() @ state.to(torch.bfloat16).transpose(0, 1).float()           # (C, Dv)
    # print_tensor_info(v_inner, 'v_inner')
    v_new = v_inner + v_prime                                # (C, Dv), use mm atomic add
    # print(attn_inter.reshape(-1)[:10])
    # print(attn_inter.reshape(-1)[-10:])
    # print(v_new.reshape(-1)[:10])
    # print(v_new.reshape(-1)[-10:])

    state_out = (v_new.transpose(0, 1).to(torch.bfloat16) @ kg)                   # (Dv, C) @ (C, Dk)
    # print(state.reshape(-1)[:10])
    # print(state.reshape(-1)[-10:])
    # V1
    state_old = (state.float() * g_cum_exp[-1]).to(torch.bfloat16)
    # print(g_cum_exp[:10])
    # print(g_cum_exp[-10:])
    state_out = state_old + state_out                        # (Dv, Dk)

    return state_out, attn_inter, v_new

def stage2(
    q_prime,    # (Nv, Sp, Dk)
    v_inner,    # (Nv, Sp, Dv)
    g_cum_exp,  # (Nv, Sp)
    k_cumdecay, # (Nv, Sp, Dk)
    state,      # (Nv, Dv, Dk)
    kg,         # (Nv, Sp, Dk)
    C,           # chunk size
    T, 
    bid

):
    Nv, Sp, Dv = v_inner.shape
    _, _, Dk = q_prime.shape
    attn_inter = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=q_prime.device)
    v_new = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=q_prime.device)
    final_state = torch.empty_like(state).to(torch.bfloat16)

    # model = Stage23().npu()
    for nid in range(Nv):
        cur_state = state[nid]
        for idx in range(0, Sp, C):
            qg_chunk = q_prime[nid, idx:idx + C, :]
            v_inner_chunk = v_inner[nid, idx:idx + C, :]
            g_cum_exp_chunk = g_cum_exp[nid, idx:idx + C]
            k_cumdecay_chunk = k_cumdecay[nid, idx:idx + C, :]
            kg_chunk = kg[nid, idx:idx + C, :]
            cur_state, attn_inter_chunk, v_new_chunk = stage2_chunk(
                qg_chunk, v_inner_chunk, g_cum_exp_chunk, k_cumdecay_chunk, cur_state, kg_chunk)
            # dummy = torch.empty(0, dtype=qg_chunk.dtype).npu()
            # s2_1, s2_2, s2_3, s2_4 = model(
            #     q_prime, kg, v_inner, g_cum_exp, k_cumdecay, state.to(torch.bfloat16).contiguous(),
            #     dummy, dummy, dummy, T, bid, nid, idx, C, Dk, Dv)

            attn_inter[nid, idx:idx + C, :] = attn_inter_chunk
            v_new[nid, idx:idx + C, :] = v_new_chunk
        final_state[nid, ...] = cur_state
    # print_tensor_info(v_new, "k_cumdecay")
    return final_state, attn_inter, v_new

def stage3_chunk(
    qkt,       # (C, C)
    value,       # (C, Dv)
    scale,       # float
    g_cum_exp,   # (C,)
    attn_inter,  # (C, Dv)
    v_new        # (C, Dv)
):
    # print(qkt.shape, value.shape, g_cum_exp.shape, attn_inter.shape, v_new.shape)
    device = value.device
    C = value.shape[0]
    core_attn_out = torch.zeros_like(value).to(torch.bfloat16)  # (C, Dv)
    # V1
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=0)
    masked_qkt = qkt.float() * scale * (g_cum_exp[:, None] / g_cum_exp[None, :]) * lower.float()
    # print(qkt.shape)
    # print(masked_qkt.shape)
    # print(g_cum_exp.shape)
    # print(attn_inter.shape)
    # print(v_new.shape)
    # print(masked_qkt[:7, :7].reshape(-1)[:10])
    # print(masked_qkt[:7, :7].reshape(-1)[-10:])
    # C2
    attn_inner = (masked_qkt.to(torch.bfloat16) @ v_new)           # (C, Dv)
    core_attn_out = (attn_inter + attn_inner).to(torch.bfloat16)   # (C, Dv)

    return core_attn_out

def stage3(
    qkt,         # (Nv, Sp, C)
    value,       # (S, Nv, Dv)
    scale,       # float
    g_cum_exp,   # (Nv, Sp)
    attn_inter,  # (Nv, Sp, Dv)
    v_new,       # (Nv, Sp, Dv)
    C,            # chunk size
    T,
    bid
):
    Nv, Sp, Dv = attn_inter.shape
    S, _, _ = value.shape
    assert Sp == (S + C - 1) // C * C

    attn_out =  torch.empty((Sp, Nv, Dv), dtype=torch.bfloat16, device=value.device)   # (Sp, Nv, Dv)

    # model = Stage23().npu()

    for nid in range(Nv):
        for idx in range(0, Sp, C):
            v_chunk = get_chunk(value[:, nid, :], C, idx)
            g_cum_exp_chunk = g_cum_exp[nid, idx:idx + C]
            attn_inter_chunk = attn_inter[nid, idx:idx + C, :]
            v_new_chunk = v_new[nid, idx:idx + C, :]
            qkt_chunk = qkt[nid, idx:idx + C, :]
            attn_out_chunk = stage3_chunk(qkt_chunk, v_chunk, scale, g_cum_exp_chunk, attn_inter_chunk, v_new_chunk)

            attn_out[idx:idx + C, nid, ...] = attn_out_chunk
    
    return attn_out


def chunk_gdn_golden(
    query,              # (T, Nk, Dk)
    key,                # (T, Nk, Dk)
    value,              # (T, Nv, Dv)
    beta,               # (T, Nv)
    scale,              # float
    initial_state,      # (B, Nv, Dv, Dk)
    actual_seq_lengths, # (B,)
    g = None            # (T, Nv)
):
    T, Nk, Dk = query.shape
    B, Nv, Dv, _ = initial_state.shape
    device=query.device

    if g is None:
        g = torch.zeros((T, Nv), dtype=torch.float32, device=device)
    attn_out = torch.empty((T, Nv, Dv), dtype=query.dtype, device=device)
    attn_out = (attn_out).to(torch.bfloat16)
    final_state = torch.empty_like(initial_state).to(torch.bfloat16)

    start = 0
    C = 64
    for bid in range(B):
        cur_state = initial_state[bid].clone().to(torch.float32)
        S = actual_seq_lengths[bid]
        end = start + S

        g_cum_exp, k_cum_decay, v_inner, q_prime, kg, qkt = stage1(
            query[start:end], key[start:end], value[start:end], g[start:end], beta[start:end], scale, C)

        cur_state, attn_inter, v_new = stage2(
            q_prime, v_inner, g_cum_exp, k_cum_decay, cur_state, kg, C, T, bid)
        final_state[bid] = cur_state.to(torch.bfloat16)
        attn_out_paddend = stage3(
            qkt, value[start:end], scale, g_cum_exp, attn_inter, v_new, C, T, bid)

        attn_out[start:end, ...] = attn_out_paddend[:S]
        start = end

    import time_step
    torch_npu.npu.synchronize()
    start = time.perf_counter()
    o2, state2 = torch_npu.npu_chunk_gated_delta_rule(
        query.npu(), key.npu(), value.npu(), 
        beta = beta.npu(),
        initial_state = initial_state.clone().to(torch.bfloat16).npu(),
        actual_seq_lengths = actual_seq_lengths.npu(),
        scale = scale,
        g = g)
    torch_npu.npu.synchronize()
    end = time.perf_counter()
    print(f'==========   Time: {(end - start) * 1_000_000}us   ==============')
    # import triton
    # import time   
    # start = time.perf_counter_ns()
    # torch_npu.npu_chunk_gated_delta_rule(
    #     query.npu(), key.npu(), value.npu(), 
    #     beta = beta.npu(),
    #     initial_state = initial_state.to(torch.bfloat16).npu(),
    #     actual_seq_lengths = actual_seq_lengths.npu(),
    #     scale = scale,
    #     g = g)
    # end = time.perf_counter_ns()

    # t_torch = triton.testing.do_bench(lambda: 
    # )
    # print(f"runtime of torch = {(end - start) / 1000} us")
    # verify_result(state2, final_state)
    # verify_result(o2, attn_out)
    _, res1 = verify_result2(state2, final_state)
    _, res2 = verify_result2(o2, attn_out)
    res = 0
    res = 1 if res1 == 1 or res2 == 1 else 0
    # return attn_out, final_state
    # return attn_out, final_state, 0
    return attn_out, final_state, res

def generate_inputs(bs=64, seqlen=2, nk=32, nv=32, dk=128, dv=128, dtype=torch.float32):
    actual_seq_lengths = (torch.ones(bs) * seqlen).npu().to(torch.int32)
    T = torch.sum(actual_seq_lengths)
    print(f"\n{bs=}, {seqlen=}, {nk=}, {nv=}, {dk=}, {dv=}")
    initial_state = torch.rand((bs, nv, dv, dk), dtype=dtype, device='npu')

    q = torch.rand((T, nk, dk), dtype=dtype, device='npu')
    k = torch.rand((T, nk, dk), dtype=dtype, device='npu')
    v = torch.rand((T, nv, dv), dtype=dtype, device='npu')
    g = torch.rand((T, nv), dtype=torch.float32, device='npu') * -1.0
    beta = torch.rand((T, nv), dtype=dtype, device='npu')


    ssm_state_indices = (torch.arange(bs, device='npu')).to(torch.int32)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    scale = 1 / (dk**-0.5)
    num_accepted_tokens = actual_seq_lengths

    return {
        "query": q,
        "key": k,
        "value": v,
        "initial_state": initial_state,
        "g": g,
        "beta": beta,
        "scale": scale,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "actual_seq_lengths": actual_seq_lengths
    }

def show_err(pred, target, epsilon):
    abs_error = torch.abs(pred - target)
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    print(f"\ttarget max abs {abs(target).max():.2e}")
    print(f"\tmax absolute error {abs_error.max():.2e}")
    print(f"\tmax relative error {rel_error.max():.2e}")

def chunk_gated_delta_rule_npu_hs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    num_heads = q.shape[-2]
    num_value_heads = v.shape[-2]

    if num_value_heads // num_heads > 1:
        q = q.repeat_interleave(num_value_heads // num_heads, dim=2)
        k = k.repeat_interleave(num_value_heads // num_heads, dim=2)

    batch_size = initial_state.shape[0]
    core_attn_out = []
    last_recurrent_state = torch.empty_like(initial_state)

    for b_idx in range(batch_size):

        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        cur_q = q[:, start:end, ...]
        cur_k = k[:, start:end, ...]
        cur_v = v[:, start:end, ...]
        cur_g = g[:, start:end, ...]
        cur_beta = beta[:, start:end, ...]
        cur_state = initial_state[b_idx].unsqueeze(0)

        cur_core_attn_out, cur_last_recurrent_state = chunk_gated_delta_rule_native(
            query=cur_q,
            key=cur_k,
            value=cur_v,
            g=cur_g,
            beta=cur_beta,
            initial_state=cur_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        core_attn_out.append(cur_core_attn_out)
        last_recurrent_state[b_idx] = cur_last_recurrent_state

    tar_dtype = core_attn_out[0].dtype
    tar_device = core_attn_out[0].device
    tar_shape = list(core_attn_out[0].shape)
    tar_shape[1] = cu_seqlens[-1]
    final_cor_attn_out = torch.empty(tar_shape, dtype=tar_dtype, device=tar_device)

    for b_idx in range(batch_size):

        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        final_cor_attn_out[:, start:end, ...] = core_attn_out[b_idx]

    return final_cor_attn_out, last_recurrent_state

class TestChunkGatedDeltaRule(TestCase):
    @unittest.skip('Skip test temporarily: CANN version-related operators not supported yet')
    @SupportedDevices(["Ascend910b"])
    def test_chunk_gated_delta_rule_1(self, device="npu"):
        (bs, seqlen, nk, nv, dk, dv) = (64, 2, 32, 32, 128, 128)

        inputs = generate_inputs(bs=bs, seqlen=seqlen, nk=nk, nv=nv, dk=dk, dv=dv, dtype=torch.bfloat16)
        actual_seq_lengths = inputs['actual_seq_lengths']  # (B,)
        cu_seqlens=F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)
        initial_state = inputs['initial_state']  # (B, Nv, Dv, Dk)
        q = inputs['query']  # (T, Nk, Dk)
        k = inputs['key']    # (T, Nk, Dk)
        v = inputs['value']  # (T, Nv, Dv)
        g = inputs['g']      # (T, Nv)
        beta = inputs['beta']  # (T, Nv)
        scale = 1 / (k.shape[-1] ** 0.5)

        o_hs, state_hs = chunk_gated_delta_rule_npu_hs(
            q.unsqueeze(0).to(torch.bfloat16),
            k.unsqueeze(0).to(torch.bfloat16),
            v.unsqueeze(0).to(torch.bfloat16),
            g.unsqueeze(0),
            beta.unsqueeze(0).to(torch.bfloat16),
            scale=scale,
            initial_state=initial_state.transpose(-1, -2).clone().to(torch.bfloat16),
            cu_seqlens=cu_seqlens
        )
        o_hs = o_hs[0]
        state_hs = state_hs.transpose(-1, -2)

        # print(initial_state.shape)
        o_golden, state_golden, res = chunk_gdn_golden(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), beta.to(torch.bfloat16),
            scale, initial_state.clone(), actual_seq_lengths, g=g.to(torch.float32)
        )
        # return res
        # print("o_golden vs o_hs:")
        # show_err(o_golden, o_hs, 1e-6)
        # # show_err(o_golden, o_hs, 1e-6)
        # print("state_golden vs state_hs:")
        # show_err(state_golden, state_hs, 1e-6)
        # if torch.allclose(o_golden, o_hs, 1e-5, 1e-5) and torch.allclose(state_golden, state_hs, 1e-5, 1e-5):
        #     print("PASS")
        # else:
        #     print("FAILED")
        self.assertRtolEqual(o_golden, o_hs, 1e-5)
        self.assertRtolEqual(state_golden, state_hs, 1e-5)

if __name__ == "__main__":

    run_tests()