import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFusedInferAttentionScore(TestCase):
    def softmax(self, x):
        x = x.cpu().numpy().astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y
        return ans, x_sum, x_max

    def supported_op_exec(self, query_states1, past_key, past_value, head_dim, B, N, S, softmax_lse_flag):
        attn_weights1 = torch.matmul(query_states1, past_key.transpose(2, 3)) / 0.0078125
        if (softmax_lse_flag == True):
            softmax_res, softmax_sum, softmax_max = self.softmax(attn_weights1)
            lse = np.log(softmax_sum) + softmax_max
        else:
            lse = np.zeros([B, N, S, 1], np.float32)
            attn_weights1 = torch.max(attn_weights1, torch.full(
                (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device))
            attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)

        attn_output1 = torch.matmul(attn_weights1, past_value)
        return attn_output1, lse
    
    def supported_op_exec_ntd(self, query_states1, past_key, past_value, head_dim, T, N):
        attn_weights1 = torch.matmul(query_states1, past_key.transpose(1, 2)) / 0.0078125

        attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)

        attn_output1 = torch.matmul(attn_weights1, past_value)
        attn_output1 = attn_output1.transpose(0, 1)
        return attn_output1

    def custom_op_exec(self, query, key, value, head_dim, softmax_lse_flag):
        scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

    def custom_op_exec_tnd(self, query, key, value, head_dim, softmax_lse_flag):
        scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="TND", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

    def custom_op_exec_ntd_tnd(self, query, key, value, head_dim, actseqlen, actseqlenkv, softmax_lse_flag):
        scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=2, input_layout="NTD_TND", scale=scale, pre_tokens=65535, next_tokens=65535, actual_seq_lengths=actseqlen, 
            actual_seq_lengths_kv=actseqlenkv, softmax_lse_flag=softmax_lse_flag)
    
    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        golden = golden.flatten().to(torch.float32)
        out = out.flatten().to(torch.float32)
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        calc_times = self.head_size_qk * self.max_context_len + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else:
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            return res
        else:
            if calc_times < 2048:
                error = 2**(-8)
            else:
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            return res

    def group_matmul(self, head, kv_head, A, B):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = torch.matmul(A[i * group_num: (i + 1) * group_num, :, :].to(torch.float32),
                                    B[i:(i + 1), :, :].to(torch.float32))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score

    def softmax_numpy(self, sim):
        sim = sim.cpu().numpy()
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask,    # (q_seqlen, k_seqlen)
    ):
        # Q * K.T
        query = query
        query = torch.permute(query, (1, 0, 2))
        key = torch.permute(key, (1, 2, 0))

        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)

        sim_high = sim_high * scale
        if mask is not None:
            sim_high = sim_high + (mask[:sim_high.shape[-2], :sim_high.shape[-1]] * self.post_mask_factor).to(torch.float32)

        # softmax
        p_high = self.softmax_numpy(sim_high)
        p = torch.from_numpy(p_high).to(query.dtype)
        p_high = torch.from_numpy(p_high).to(torch.float32)
        value = torch.permute(value, (1, 0, 2))

        # P * V
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_high = torch.permute(out_high, (1, 0, 2))
        out = torch.permute(out, (1, 0, 2))
        out = out.to(query.dtype)
        return out, out_high

    def ref_single_query_cached_kv_attention(self, params: dict) -> None:
        output = params['output']
        true_out = params['true_out']
        query = params['query']
        key_cache = params['key_cache']
        value_cache = params['value_cache']
        block_tables = params['block_tables']
        q_seqlen_list = params['q_seqlen_list']
        k_seqlen_list = params['k_seqlen_list']
        global_mask = params['global_mask']
        mask_type = params['mask_type']
        window_size = params['window_size']

        num_heads = query.shape[2]
        kv_heads = value_cache.shape[2]
        head_size_qk = key_cache.shape[3]
        head_size_vo = value_cache.shape[3]
        block_size = value_cache.shape[1]

        batch = len(q_seqlen_list)
        scale = 1.0 / (head_size_qk ** 0.5)
        cu_seqlen = 0
        for i in range(batch):
            q_seqlen = int(q_seqlen_list[i])
            k_seqlen = int(k_seqlen_list[i])
            q = query[i]
            block_table = block_tables[i]
            keys = []
            values = []
            cur_window_size = window_size
            if k_seqlen < window_size:
                cur_window_size = k_seqlen
            for j in range(k_seqlen - cur_window_size, k_seqlen):
                block_number = int(block_table[j // block_size][0])
                block_offset = j % block_size

                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_qk)
                k = torch.tensor(k)
                keys.append(k)

                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                v = torch.tensor(v)
                values.append(v)
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)

            if mask_type == 1:
                mask = global_mask[k_seqlen - q_seqlen:k_seqlen, :k_seqlen]  # prefill, decoder, prefill + decoder
            elif mask_type == 3:
                mask = global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]  # lookahead: cur_q: cur_q + q
            else:
                mask = None
            out, out_high = self.ref_masked_attention(q, keys, values, scale, mask)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            output[i] = out
            true_out[i] = out_high
            cu_seqlen += q_seqlen

    def calc_data(self, params):
        q = params['q']
        k = params['k']
        v = params['v']
        num_heads = params['num_heads']
        kv_heads = params['kv_heads']
        num_blocks = params['num_blocks']
        block_size = params['block_size']
        head_size_qk = params['head_size_qk']
        head_size_vo = params['head_size_vo']
        q_seqlen_list = params['q_seqlen_list']
        k_seqlen_list = params['k_seqlen_list']
        mask_type = params['mask_type']
        window_size = params['window_size']
        blk_tab = params['blk_tab']
        dtype = params.get('dtype', torch.float16)
        self.data_type = dtype
        self.head_size_qk = head_size_qk

        num_tokens = np.array(q_seqlen_list).sum()
        batch_size = len(q_seqlen_list)
        self.max_context_len = max(k_seqlen_list)

        q_seqlen = 1

        query = q.reshape(batch_size, q_seqlen, num_heads, head_size_qk)
        key_cache = k.reshape(num_blocks, block_size, kv_heads, head_size_qk)
        value_cache = v.reshape(num_blocks, block_size, kv_heads, head_size_vo)

        blktab_data = np.array(blk_tab, dtype=np.int32).reshape(-1)

        query = query
        key_cache = key_cache
        value_cache = value_cache

        max_k_seqlen = max(k_seqlen_list)
        max_num_blocks_per_seq = (max_k_seqlen + block_size - 1) // block_size
        block_tables = []   # (num_tokens, max_num_blocks_per_seq）
        for i in range(batch_size):
            block_table = []
            for j in range(max_num_blocks_per_seq):
                block_table.append([blktab_data[i * max_num_blocks_per_seq + j]])
            block_tables.append(block_table)
        self.pre_mask_factor = -10000.0
        self.post_mask_factor = 1.0
        if self.data_type == torch.bfloat16:
            self.pre_mask_factor = 1.0
            self.post_mask_factor = -10000.0

        if mask_type == 1:
            mask = np.ones(shape=(max_k_seqlen, max_k_seqlen)).astype(np.float16)
            mask = np.triu(mask, 1)
            mask *= -10000.0
            mask = torch.from_numpy(mask).to(dtype)
        elif mask_type == 3:
            mask = np.zeros(shape=(batch_size, max_k_seqlen)).astype(np.float16)
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = q_seqlen_list[i]
                kseqlen = k_seqlen_list[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                tri *= self.pre_mask_factor
                mask[pre_qseqlen:(pre_qseqlen + qseqlen), kseqlen - qseqlen:kseqlen] = tri
                pre_qseqlen += qseqlen

            mask = torch.from_numpy(mask).to(dtype)
        elif mask_type == 0:
            mask = None

        shape_out = (batch_size, q_seqlen, num_heads, head_size_vo)
        ref_output = torch.zeros(shape_out, dtype=dtype)
        true_out = torch.zeros(shape_out, dtype=torch.float32)
        ref_params = {
            'output': ref_output,
            'true_out': true_out,
            'query': query,
            'key_cache': key_cache,
            'value_cache': value_cache,
            'block_tables': block_tables,
            'q_seqlen_list': q_seqlen_list,
            'k_seqlen_list': k_seqlen_list,
            'global_mask': mask,
            'mask_type': mask_type,
            'window_size': window_size
        }
        self.ref_single_query_cached_kv_attention(ref_params)

        self.q = query
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = np.array(block_tables).astype(np.int32)
        self.q_seqlen_list = np.array(q_seqlen_list).astype(np.int32)
        self.k_seqlen_list = np.array(k_seqlen_list).astype(np.int32)
        self.mask = mask
        self.golden_out = ref_output
        self.true_out = true_out

    def golden_compare(self, out_tensors, golden_tensors):
        result_old = self.compare_output_data(out_tensors[0], golden_tensors[0], [0.001, 0.001, 0.005, 0.005])
        return result_old

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_sliding(self, device="npu"):
        b = 2
        n = 64
        kv_n = 16
        d_qk = 128
        d_vo = 192

        block_num = 128
        block_size = 64

        max_s_kv = 16 * 1024

        # 生成随机数据，并发送到npu
        q = torch.randn(b, 1, n * d_qk, dtype=torch.bfloat16) # （b, 1, N * D）
        k = torch.randn(block_num, block_size, kv_n * d_qk, dtype=torch.bfloat16) # （blocknum, block_size, KV_N * D）
        v = torch.randn(block_num, block_size, kv_n * d_vo, dtype=torch.bfloat16) # （blocknum, block_size, KV_N * D）

        block_table = torch.randint(0, 128, (b, ((max_s_kv + block_size - 1) // block_size)), dtype=torch.int32)

        scale = 1 / math.sqrt(d_qk)
        actseqlen = [1] * b
        actseqlenkv = [max_s_kv] * b
        window_size = 512
        out, _ = torch.ops.npu.npu_fused_infer_attention_score(q.npu(), k.npu(), v.npu(), block_table=block_table.npu(), actual_seq_lengths_kv=actseqlenkv,
                        num_heads=n, num_key_value_heads=kv_n, input_layout="BSH", scale=scale, pre_tokens=window_size, block_size=block_size)

        calc_params = {
            'q': q,
            'k': k,
            'v': v,
            'num_heads': n,
            'kv_heads': kv_n,
            'num_blocks': block_num,
            'block_size': block_size,
            'head_size_qk': d_qk,
            'head_size_vo': d_vo,
            'q_seqlen_list': actseqlen,
            'k_seqlen_list': actseqlenkv,
            'mask_type': 0,
            'window_size': window_size,
            'blk_tab': block_table,
            'dtype': torch.bfloat16
        }
        self.calc_data(calc_params)
        out_r = out.reshape(self.golden_out.shape)
        res = self.golden_compare([self.golden_out.cpu()], [out_r.cpu()])       
        self.assertRtolEqual(res, True)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score(self, device="npu"):
        query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        softmax_lse_flag = False

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 2048, softmax_lse_flag)
        custom_output = self.custom_op_exec(query, key, value, head_dim, False)
        attention_output = custom_output[0]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_pfa_return_lse(self, device="npu"):
        query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        softmax_lse_flag = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 2048, softmax_lse_flag)
        custom_output = self.custom_op_exec(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]
    
    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_ifa_return_lse(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        softmax_lse_flag = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, softmax_lse_flag)
        custom_output = self.custom_op_exec(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_tnd(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        softmax_lse_flag = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, softmax_lse_flag)
        custom_output = self.custom_op_exec_tnd(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v3_param(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        query_rope = torch.randn(1, 32, 1, 16, dtype=torch.float16).npu()
        key_rope = torch.randn(1, 32, 2048, 16, dtype=torch.float16).npu()
        key_rope_antiquant_scale = torch.randn(1, 1, 1, 16, dtype=torch.float16).npu()
        head_dim = 128
        softmax_lse_flag = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, softmax_lse_flag)
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, query_rope=query_rope, key_rope=key_rope, key_rope_antiquant_scale=key_rope_antiquant_scale,
            num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_tnd_ntd(self, device="npu"):
        query = torch.randn(2, 32, 512, dtype=torch.float16).npu()
        key = torch.randn(1, 1, 2048, 512, dtype=torch.float16).npu()
        value = torch.randn(1, 1, 2048, 512, dtype=torch.float16).npu()

        head_dim = 512
        softmax_lse_flag = True
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="TND_NTD", scale=scale, pre_tokens=65535,
            next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_ntd_tnd(self, device="npu"):
        query = torch.full((2, 32, 192), 1, dtype=torch.bfloat16).npu()
        key = torch.full((2, 32, 192), 1, dtype=torch.bfloat16).npu()
        value = torch.full((2, 32, 128), 1, dtype=torch.bfloat16).npu()

        head_dim = 128
        softmax_lse_flag = False

        actseqlen = [32]
        actseqlenkv = [32]

        golden_output = self.supported_op_exec_ntd(query, key, value, head_dim, 32, 2)
        custom_output = self.custom_op_exec_ntd_tnd(query, key, value, head_dim, actseqlen, actseqlenkv, softmax_lse_flag)
        attention_out = custom_output[0]
        self.assertRtolEqual(golden_output, attention_out, prec=0.000001, prec16=0.000001)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v3_fullquant(self, device="npu"):
        query = torch.randint(1, 2, (1, 32, 2, 128), dtype=torch.int8).npu()
        key = torch.randint(1, 2, (1, 32, 2048, 128), dtype=torch.int8).npu()
        value = torch.randint(1, 2, (1, 32, 2048, 128), dtype=torch.int8).npu()
        dequant_scale1 = torch.ones(1).npu()
        dequant_scale2 = torch.ones(1).npu()
        quant_scale1 = torch.ones(1).npu()
        quant_scale2 = torch.ones(1).npu()

        head_dim = 128
        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, dequant_scale1=dequant_scale1, dequant_scale2=dequant_scale2, quant_scale1=quant_scale1,
            quant_scale2=quant_scale2, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535,
            softmax_lse_flag=softmax_lse_flag)

        query_t = torch.ones(1, 32, 2, 128).npu()
        key_t = torch.ones(1, 32, 2048, 128).npu()
        value_t = torch.ones(1, 32, 2048, 128).npu()
        
        golden_output = self.supported_op_exec(query_t, key_t, value_t, head_dim, 1, 32, 1, softmax_lse_flag)
        res = custom_output[0].equal(golden_output[0])
        self.assertRtolEqual(res, True)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v3_antiquant(self, device="npu"):
        query = torch.ones(32, 8, 1, 128, dtype=torch.float16).npu()
        key = torch.full((32, 8, 2048, 16), 286331353, dtype=torch.int32).npu()
        value = torch.full((32, 8, 2048, 16), 286331353, dtype=torch.int32).npu()
        key_antiquant_scale = torch.ones(1, dtype=torch.float16).npu()
        value_antiquant_scale = torch.ones(1, dtype=torch.float16).npu()
        key_antiquant_mode = torch.ones(1).npu()
        value_antiquant_mode = torch.ones(1).npu()

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, key_antiquant_scale=key_antiquant_scale, value_antiquant_scale=value_antiquant_scale,
            key_antiquant_mode=key_antiquant_mode, value_antiquant_mode=value_antiquant_mode, num_heads=8, input_layout="BNSD", 
            scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(32, 8, 1, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

if __name__ == "__main__":
    run_tests()
