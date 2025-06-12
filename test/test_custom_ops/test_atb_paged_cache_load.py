import random
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


MAX_SEQ_LEN = 1024


class TestPagedCacheLoad(TestCase):
    num_tokens = 16
    num_head = 16
    head_size = 16
    block_size = 16
    num_blocks = 8

    def genarate_data(
            self,
            batch=4,
            seq_len=1,
            num_heads=2,
            head_size_k=32,
            head_size_v=32,
            block_size=128,
            num_blocks=4,
    ):
        num_tokens = batch * seq_len
        key_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(
            np.float16)
        value_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_v // 16, block_size, 16)).astype(
            np.float16)
        
        context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
        max_context_len = max(context_lens)

        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []
        for _ in range(num_tokens):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        
        context_lens = np.array(context_lens).astype(np.int32)
        block_tables = np.array(block_tables).astype(np.int32)
        sum_context_lens = sum(context_lens)
        key_expect = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
        value_expect = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)

        elenum_aligned = 16
        kv_rslt_id = 0

        for i in range(num_tokens):
            block_table = block_tables[i]
            context_lens = int(context_lens[i])

            for j in range(context_lens):
                block_id = int(block_table[j // block_size])
                block_offset = j % block_size

                if block_id < 0:
                    continue

                temp_k = np.zeros((num_heads * head_size_k))
                temp_v = np.zeros((num_heads * head_size_v))

                for k in range(num_heads * head_size_k // elenum_aligned):
                    temp_k[k * elenum_aligned: k * elenum_aligned + elenum_aligned] = value_cache[block_id][k][block_offset][:]
                

                for k in range(num_heads * head_size_v // elenum_aligned):
                    temp_v[k * elenum_aligned: k * elenum_aligned + elenum_aligned] = value_cache[block_id][k][block_offset][:]
                

                key_expect[kv_rslt_id] = temp_k
                value_expect[kv_rslt_id] = temp_v
                kv_rslt_id += 1

        return key_cache, value_cache, block_tables, context_lens, key_expect, value_expect, key, value
    
    @SupportedDevices(['Ascend910B'])
    def test_reshape_and_cache(self):
        in_tensors = [torch.from_numpy(tensor) for tensor in self.genarate_data()]
        in_tensors = [tensor.npu() for tensor in in_tensors]
        _ = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

        key_cache, value_cache, block_tables, context_lens, key_expect, value_expect, key, value = in_tensors
        key_cache = torch_npu.npu_format_cast(key_cache, 29)
        value_cache = torch_npu.npu_format_cast(value_cache, 29)
        torch_npu._npu_paged_cache_load(key_cache, value_cache, block_tables, context_lens, key, value)

        self.assertRtolEqual(key_expect, key)
        self.assertRtolEqual(value_cache, value)




if __name__ == '__main__':
    run_tests()