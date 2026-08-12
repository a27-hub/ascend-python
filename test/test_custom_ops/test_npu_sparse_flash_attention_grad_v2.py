import random
import torch
import torch_npu
import unittest
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestSparseFlashAttentionGradV2(TestCase):
    @SupportedDevices(['Ascend950'])
    def test_sfa_grad_v2(self, device="npu"):
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        query_type = torch.float16
        sparse_block_count = 2048
        b = 4
        s1 = 1
        s2 = 8192
        n1 = 128
        n2 = 1
        dn = 512
        dr = 64
        s2_act = 4096
        attention_mode = 2
        return_softmax_lse = False

        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
        key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
        value = key.clone()
        idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
        act_seq_q = [s1] * b
        act_seq_kv = [s2_act] * b

        query = query.npu()
        key = key.npu()
        value = value.npu()
        sparse_indices = sparse_indices.npu()
        query_rope = query_rope.npu()
        key_rope = key_rope.npu()
        act_seq_q = torch.tensor(act_seq_q).to(torch.int32).npu()
        act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).npu()

        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True
        query_rope.requires_grad = True
        key_rope.requires_grad = True

        npu_out, npu_softmax_max, npu_softmax_sum = torch_npu.npu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, block_table=None,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
            layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1, attention_mode=attention_mode,
            return_softmax_lse=return_softmax_lse)

        npu_out.sum().backward()

        self.assertEqual(query.grad.dtype, query.dtype)
        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.dtype, key.dtype)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.dtype, value.dtype)
        self.assertEqual(value.grad.shape, value.shape)
        self.assertEqual(query_rope.grad.dtype, query_rope.dtype)
        self.assertEqual(query_rope.grad.shape, query_rope.shape)
        self.assertEqual(key_rope.grad.dtype, key_rope.dtype)
        self.assertEqual(key_rope.grad.shape, key_rope.shape)

    @SupportedDevices(['Ascend950'])
    def test_sfa_grad_v2_with_sinks(self, device="npu"):
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        query_type = torch.float16
        sparse_block_count = 2048
        b = 4
        s1 = 1
        s2 = 8192
        n1 = 128
        n2 = 1
        dn = 512
        dr = 64
        s2_act = 4096
        attention_mode = 2
        return_softmax_lse = False

        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
        key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
        value = key.clone()
        idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
        sinks = torch.tensor(np.random.uniform(-1, 1, (n1,))).to(torch.float32)
        act_seq_q = [s1] * b
        act_seq_kv = [s2_act] * b

        query = query.npu()
        key = key.npu()
        value = value.npu()
        sparse_indices = sparse_indices.npu()
        query_rope = query_rope.npu()
        key_rope = key_rope.npu()
        sinks = sinks.npu()
        act_seq_q = torch.tensor(act_seq_q).to(torch.int32).npu()
        act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).npu()

        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True
        query_rope.requires_grad = True
        key_rope.requires_grad = True
        sinks.requires_grad = True

        npu_out, npu_softmax_max, npu_softmax_sum = torch_npu.npu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, block_table=None,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
            layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1, attention_mode=attention_mode,
            return_softmax_lse=return_softmax_lse, sinks=sinks)

        npu_out.sum().backward()

        self.assertEqual(query.grad.dtype, query.dtype)
        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.dtype, key.dtype)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.dtype, value.dtype)
        self.assertEqual(value.grad.shape, value.shape)
        self.assertEqual(query_rope.grad.dtype, query_rope.dtype)
        self.assertEqual(query_rope.grad.shape, query_rope.shape)
        self.assertEqual(key_rope.grad.dtype, key_rope.dtype)
        self.assertEqual(key_rope.grad.shape, key_rope.shape)
        self.assertEqual(sinks.grad.dtype, torch.float32)
        self.assertEqual(sinks.grad.shape, sinks.shape)


if __name__ == "__main__":
    run_tests()
