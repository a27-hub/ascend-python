import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAtbAttention(TestCase):
    @SupportedDevices(['Ascend910B'])
    def test_atb_attention(self):
        torch.manual_seed(42)
        num_tokens = 5
        batch = num_tokens
        q_head_num = 128
        kv_lora_rank = 512
        qk_rope_rank = 64
        num_blocks = 64
        block_size = 128
        kv_head = 1

        go_output = torch.empty(
            [num_tokens, q_head_num, kv_lora_rank],
            dtype=torch.float16,
        ).npu()

        lse_output = torch.empty(
            [num_tokens, q_head_num, 1],
            dtype=torch.float16,
        ).npu()

        q_nope = torch.randn(
            [num_tokens, q_head_num, kv_lora_rank],
            dtype=torch.float16
        ).npu()

        q_rope = torch.randn(
            [num_tokens, q_head_num, qk_rope_rank],
            dtype=torch.float16
        ).npu()

        ckv = torch.randn(
            [num_blocks, block_size, kv_head, kv_lora_rank],
            dtype=torch.float16
        ).npu()

        k_rope = torch.randn(
            [num_blocks, block_size, kv_head, qk_rope_rank],
            dtype=torch.float16
        ).npu()

        block_tables = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.int32).npu()
        contextLens = [10] * batch

        torch_npu.atb.npu_multi_head_latent_attention_with_lse(
            q_nope,
            q_rope,
            ckv,
            k_rope,
            block_tables,
            contextLens,
            q_head_num,
            0.14,
            1,
            calc_type="calc_type_ring",
            output=go_output,
            lse=lse_output
        )

        # 验证输出
        self.assertEqual(go_output.shape, (num_tokens, q_head_num, kv_lora_rank))
        self.assertEqual(lse_output.shape, (num_tokens, q_head_num, 1))
        self.assertEqual(go_output.sum(), 836.0)
        self.assertEqual(lse_output.sum(), 3594.0)

if __name__ == "__main__":
    run_tests()