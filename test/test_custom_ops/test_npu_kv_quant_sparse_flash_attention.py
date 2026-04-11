import math
import random
import unittest
import torch
import numpy as np
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

def _fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == -23:
        return True, 0
    hif8_value_tmp = fraction32_int >> (23 - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    elif hif8_value_tmp == 0:
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1

def _fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = 16383
    if exponent == -23:
        f14_values = (fraction32_int >> 10) + 8192
        t14_values = fraction32_int & t14_mask
        hif8_value = 0
    else:
        hif8_value = fraction32_int >> (23 - hif8_bits_num)
        f14_t14 = fraction32_int - (hif8_value << (23 - hif8_bits_num))
        f14_values = f14_t14 >> (23 - hif8_bits_num - 14)
        t14_values = f14_t14 & t14_mask
    if f14_values >= t14_values:
        if hif8_value == pow(2, hif8_bits_num) - 1:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value

def _get_hif8_fraction_bits_number(exponent):
    if exponent < -22:
        return -1, 3, 0
    if -22 <= exponent < -15:
        return 0, 3, 0
    if exponent == 0:
        return 1, 0, 3
    if abs(exponent) == 1:
        return 2, 1, 3
    if 2 <= abs(exponent) <= 3:
        return 4, 2, 3
    if 4 <= abs(exponent) <= 7:
        return 8, 3, 2
    if 8 <= abs(exponent) <= 15:
        return 12, 4, 1
    if exponent > 15:
        return 12, 4, -1

def cvt_float32_to_hifuint8(x, round_mode="round", over_mode=True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    Ec = 0
    over_value = 1.25 * pow(2.0, 15 + Ec)
    if x < 0.0:
        sign = True
        sign_int_value = 128
    if np.isinf(x) or x_abs >= over_value:
        if sign:
            if over_mode:
                return 239
            else:
                return 238
        else:
            if over_mode:
                return 111
            else:
                return 110
    if np.isnan(x):
        if over_mode:
            return 128
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < 4:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    fraction_int = int(x_abs * pow(2, 23) * pow(2, -exponent) - pow(2, 23))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = _get_hif8_fraction_bits_number(exponent)
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = _fp32_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = _fp32_ssr_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    else:
        print(f"unknow round type")
        return 0
    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = _get_hif8_fraction_bits_number(exponent)
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < -23:
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        if exponent <= -23:
            return 0
        else:
            return sign_int_value + exponent + 23
    elif dot_hif8_value == 1:
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + sig_exp + exponent_int_value + hif8_frac_value
    return hif8_int_value

def cvt_hifuint8_to_float(x, over_mode=True):
    x = int(x)
    if x == 0:
        return float(0)
    elif x == 128:
        if over_mode:
            return np.nan
        else:
            return float(0)
    elif x == 239:
        if over_mode:
            return -np.inf
        else:
            return -32768
    elif x == 111:
        if over_mode:
            return np.inf
        else:
            return 32768
    else:
        if x >= 128:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & 120
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= 12:
            exponent = x & 30
            exponent_int = exponent >> 1
            if exponent_int >= 8:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 8

            fra_int = x & 1
            m_value = 1.0 + fra_int * 0.5
        elif dot_4_value >= 8:
            exponent = x & 28
            exponent_int = exponent >> 2
            if exponent_int >= 4:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 4
            fra_int = x & 3
            m_value = 1.0 + fra_int * 0.25
        elif dot_4_value >= 4:
            exponent = x & 24
            exponent_int = exponent >> 3
            if exponent_int >= 2:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 2
            fra_int = x & 7
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= 2:
            exponent = x & 8
            exponent_sign = exponent >> 3
            if exponent_sign >= 1:
                exponent_value = -1
            else:
                exponent_value = 1
            fra_int = x & 7
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 1:
            exponent_value = 0
            fra_int = x & 7
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 0:
            m_value = 1
            exponent_value = (x & 7) - 23
        else:
            print("error, dot error")
            m_value = 0.0
            exponent_value = 0
        return sign * pow(2.0, exponent_value) * m_value

def trans_np_float_tensor_to_hifuint8(in_tensor, round_mode="round", over_mode=True):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    if multi_shape == 1.0:
        multi_shape = int(multi_shape)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_float32_to_hifuint8(in_tensor[i], round_mode, over_mode)
    out_tensor = out_tensor.astype(np.uint8)
    out_tensor = out_tensor.reshape(shape_tensor)
    return out_tensor

def trans_np_hifuint8_tensor_to_float32(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.float32)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_hifuint8_to_float(in_tensor[i])
    out_tensor = out_tensor.reshape(shape_tensor).astype(np.float32)
    return out_tensor

class TestKvQuantSparseFlashAttention(TestCase):
    def pa_to_bsnd(self, pa_in, block_table, actual_seq_lengths):
        block_num, block_size, n, d = pa_in.shape
        b = len(actual_seq_lengths)
        out = torch.zeros((b, block_num * block_size // b, 1, d)).to(pa_in.dtype)
        for i in range(b):
            for j in range(actual_seq_lengths[i] // block_size):
                out[i, j * block_size: (j + 1) * block_size, 0, :] = \
                    pa_in[block_table[i][j], :, 0, :].reshape(block_size, d)
        return out


    def gather_kv(self, k_tensor, v_tensor, sparse_indices, sparse_block_size, sparse_count,
                  batch, n2_idx, s1_idx, cur_actual_seq_lengths_kv):
        s2_sparse = list()
        for sparse_id in sparse_indices:
            if sparse_id == -1: 
                break
            begin_idx = sparse_id * sparse_block_size
            end_idx = begin_idx + sparse_block_size \
                    if begin_idx + sparse_block_size <= cur_actual_seq_lengths_kv else cur_actual_seq_lengths_kv
            s2_sparse.extend(np.arange(begin_idx, end_idx))

        k_sparse, v_sparse = k_tensor[batch, n2_idx, s2_sparse, :], v_tensor[batch, n2_idx, s2_sparse, :]

        return k_sparse, v_sparse, torch.tensor(s2_sparse)

    def softmax(self, x):
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y / x_sum
        return ans


    def cpu_kv_quant_sparse_flash_attention(self,
        query, key, value, sparse_indices, key_dequant_scale, value_dequant_scale,
        scale_value, sparse_block_size,
        actual_seq_lengths_query, actual_seq_lengths_kv,
        layout_query='BSND', layout_kv='PA_BSND', sparse_mode=3, block_table=None,
        attention_mode=0, quant_scale_repo_mode=0, tile_size=0, key_quant_mode=0,
        value_quant_mode=0, rope_head_dim=0):
        query_type = query.dtype
        query_rope = query[..., 512:]
        query = query[..., :512]
        key = self.pa_to_bsnd(key, block_table, actual_seq_lengths_kv)
        key_rope = key[..., 512: 512 + 64 * 2].view(query_type)
        key_quant_scale = key[..., 512 + 64 * 2:].view(torch.float32)

        key_slice = key[..., :512]
        if key.dtype == torch.uint8:
            key = torch.tensor(trans_np_hifuint8_tensor_to_float32(key_slice.numpy()))
        else:
            key = key_slice.to(torch.float32)

        key_quant_scale = np.repeat(key_quant_scale, repeats=tile_size, axis=-1)
        key = (key * key_quant_scale).to(query_type)
        value = key

        batch_size = actual_seq_lengths_query.shape[0]
        num_heads = query.shape[2]
        num_kv_heads = key.shape[2]
        sparse_count = sparse_indices.shape[-1]
        g = num_heads // num_kv_heads

        q_bnsd_tensor = torch.transpose(torch.cat((query, query_rope), axis=-1), 1, 2)
        k_bnsd_tensor = torch.transpose(torch.cat((key, key_rope), axis=-1), 1, 2)
        v_bnsd_tensor = torch.transpose(value, 1, 2)
        sparse_indices_tensor = torch.transpose(sparse_indices, 1, 2)
        out_shape_bnsd = list(q_bnsd_tensor.shape)
        out_shape_bnsd[-1] = out_shape_bnsd[-1] - query_rope.shape[-1]
        y = torch.zeros(out_shape_bnsd, dtype=query_type)

        for batch in range(batch_size):
            cur_acutal_seq_lengths_q = actual_seq_lengths_query[batch]
            cur_actual_seq_lengths_kv = actual_seq_lengths_kv[batch]
            for n2_idx in range(num_kv_heads):
                for s1_idx in range(cur_acutal_seq_lengths_q):
                    q_curr = q_bnsd_tensor[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :]
                    cur_sparse_indices = sparse_indices_tensor[batch, n2_idx, s1_idx, :]
                    k_sparse, v_sparse, s2_index = self.gather_kv(k_bnsd_tensor, v_bnsd_tensor, cur_sparse_indices, sparse_block_size,
                                                  sparse_count, batch, n2_idx, s1_idx, cur_actual_seq_lengths_kv)
                    mm1_res = torch.matmul(q_curr.to(torch.float32), k_sparse.to(torch.float32).T)
                    scale_res = mm1_res * scale_value
                    if sparse_mode == 3:
                        threshold = cur_actual_seq_lengths_kv - cur_acutal_seq_lengths_q + s1_idx + 1
                        mask_index = s2_index >= threshold
                        scale_res[:, mask_index] = -1e12
                    softmax_res = self.softmax(scale_res.numpy())
                    softmax_res = torch.tensor(softmax_res).to(query_type)
                    mm2_res = torch.matmul(softmax_res.to(torch.float32), v_sparse.to(torch.float32))
                    y[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :] = mm2_res
        return torch.transpose(y, 1, 2)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_sfa_eager(self, device="npu"):
        query_type = torch.bfloat16
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        sparse_block_count = 2048
        b = 4
        s1 = 1
        s2 = 8192
        n1 = 128
        n2 = 1
        dn = 512
        dr = 64
        tile_size = 128
        block_size = 256
        layout_query = 'BSND'
        s2_act = 4096

        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
        key = torch.tensor(np.random.uniform(-5, 10, (b * (s2 // block_size), block_size, n2, dn))).to(torch.int8)
        value = key.clone()
        idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, dr))).to(query_type)
        act_seq_q = torch.tensor([s1] * b).to(torch.int32)
        act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32)
        antiquant_scale = torch.tensor(np.random.uniform(-100, 100, (b * (s2 // block_size), block_size, n2,
            dn // tile_size))).to(torch.float32)
        key = torch.cat((key, key_rope.view(torch.int8), antiquant_scale.view(torch.int8)), axis=3)
        query = torch.cat((query, query_rope), axis=3)
        block_table = torch.tensor([range(b * s2 // block_size)], dtype=torch.int32).reshape(b, -1)

        # compare result
        cpu_out = self.cpu_kv_quant_sparse_flash_attention(
            query, key, value, sparse_indices,
            key_dequant_scale=antiquant_scale, value_dequant_scale=antiquant_scale,
            scale_value=scale_value, sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            layout_query='BSND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table,
            attention_mode=2, quant_scale_repo_mode=1, tile_size=tile_size, key_quant_mode=2,
            value_quant_mode=2, rope_head_dim=64)

        query = query.npu()
        key = key.npu()
        value = value.npu()
        sparse_indices = sparse_indices.npu()
        query_rope = query_rope.npu()
        key_rope = key_rope.npu()
        act_seq_q = act_seq_q.npu()
        act_seq_kv = act_seq_kv.npu()
        block_table = block_table.npu()
        antiquant_scale = antiquant_scale.npu()

        npu_out = torch_npu.npu_kv_quant_sparse_flash_attention(
            query, key, value, sparse_indices, 
            scale_value=scale_value, sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            layout_query='BSND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table,
            attention_mode=2, quant_scale_repo_mode=1, tile_size=tile_size, key_quant_mode=2,
            value_quant_mode=2, rope_head_dim=64)

        npu_out = npu_out.cpu().to(torch.float32).numpy()
        cpu_out = cpu_out.to(torch.float32).numpy()

        res = np.isclose(npu_out, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        true_ratio = np.mean(res)
        if true_ratio < 0.99:
            print("npu output:\n", npu_out, npu_out.shape)
            print("cpu output:\n", cpu_out, cpu_out.shape)
            print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        self.assertTrue(true_ratio > 0.99, "precision compare fail")

    @SupportedDevices(['Ascend950'])
    def test_npu_kv_quant_sparse_flash_attention_hif8(self, device="npu"):
        query_type = torch.bfloat16
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        sparse_block_count = 2048
        b = 4
        s1 = 1
        s2 = 8192
        n1 = 48
        n2 = 1
        dn = 512
        dr = 64
        tile_size = 128
        block_size = 256
        layout_query = 'BSND'
        s2_act = 4096

        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
        key_float = np.random.uniform(-5, 10, (b * (s2 // block_size), block_size, n2, dn)).astype(np.float32)
        key = torch.tensor(trans_np_float_tensor_to_hifuint8(key_float, round_mode="hybrid", over_mode=True))
        value = key.clone()
        idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, dr))).to(query_type)
        act_seq_q = torch.tensor([s1] * b).to(torch.int32)
        act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32)
        antiquant_scale = torch.tensor(np.random.uniform(-100, 100, (b * (s2 // block_size), block_size, n2,
            dn // tile_size))).to(torch.float32)
        key = torch.cat((key, key_rope.view(torch.uint8), antiquant_scale.view(torch.uint8)), axis=3)
        query = torch.cat((query, query_rope), axis=3)
        block_table = torch.tensor([range(b * s2 // block_size)], dtype=torch.int32).reshape(b, -1)

        # compare result
        cpu_out = self.cpu_kv_quant_sparse_flash_attention(
            query, key, value, sparse_indices,
            key_dequant_scale=antiquant_scale, value_dequant_scale=antiquant_scale,
            scale_value=scale_value, sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            layout_query='BSND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table,
            attention_mode=2, quant_scale_repo_mode=1, tile_size=tile_size, key_quant_mode=2,
            value_quant_mode=2, rope_head_dim=64)

        query = query.npu()
        key = key.npu()
        value = value.npu()
        sparse_indices = sparse_indices.npu()
        query_rope = query_rope.npu()
        key_rope = key_rope.npu()
        act_seq_q = act_seq_q.npu()
        act_seq_kv = act_seq_kv.npu()
        block_table = block_table.npu()
        antiquant_scale = antiquant_scale.npu()

        npu_out = torch_npu.npu_kv_quant_sparse_flash_attention(
            query, key, value, sparse_indices, 
            scale_value=scale_value, sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            layout_query='BSND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table,
            attention_mode=2, quant_scale_repo_mode=1, tile_size=tile_size, key_quant_mode=2,
            value_quant_mode=2, rope_head_dim=64,
            key_dtype=torch_npu.hifloat8, value_dtype=torch_npu.hifloat8)

        npu_out = npu_out.cpu().to(torch.float32).numpy()
        cpu_out = cpu_out.to(torch.float32).numpy()

        res = np.isclose(npu_out, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        true_ratio = np.mean(res)
        if true_ratio < 0.99:
            print("npu output:\n", npu_out, npu_out.shape)
            print("cpu output:\n", cpu_out, cpu_out.shape)
            print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        self.assertTrue(true_ratio > 0.99, "precision compare fail")

if __name__ == "__main__":
    run_tests()