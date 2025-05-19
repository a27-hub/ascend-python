import random
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from typing import Optional, Tuple
from torch.testing._internal.common_utils import parametrize


class TestRingAttention(TestCase):

    def flash_mla_with_kvcache_torch(
        self,
        q: torch.Tensor,  # [b, s_q=1, h_q, d]
        k_cache: torch.Tensor,  # [num_blocks, block_size, h_kv=1, d]
        block_table: torch.Tensor,  # [b, max_seqlen//block_size]
        cache_seqlens: torch.Tensor,  # [b]
        head_dim_v: int,  # value dimension (kv_lora_rank)
        softmax_scale: float = 1.0,  # attention scale factor
        causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch reference implementation for flash_mla_with_kvcache.

        Args:
            q: Query tensor with shape [batch, 1, num_heads_q, head_dim]
            k_cache: Key cache tensor with shape [num_blocks, block_size, 1, head_dim]
            block_table: Block mapping table [batch, max_seqlen//block_size]
            cache_seqlens: Sequence lengths [batch]
            head_dim_v: Value head dimension (kv_lora_rank)
            softmax_scale: Scale factor for attention scores
            causal: Whether to use causal attention

        Returns:
            output: Output tensor [batch, 1, num_heads_q, head_dim_v]
            lse: Log sum exp [batch, num_heads_q, 1]
        """
        b, s_q, h_q, d = q.shape
        assert s_q == 1, "Decoding only supports sequence length 1"

        num_blocks, block_size, h_kv, _ = k_cache.shape
        assert h_kv == 1, "Key cache should have 1 head dimension"

        device = q.device
        dtype = q.dtype

        # Initialize output tensors
        out = torch.empty(b, 1, h_q, head_dim_v, dtype=dtype, device=device)
        lse = torch.empty(b, h_q, 1, dtype=torch.float32, device=device)

        # Scale query
        q = q * softmax_scale

        for i in range(b):
            # Get sequence length for this batch
            seq_len = cache_seqlens[i].item()

            # Get query for this batch
            q_i = q[i]  # [s_q=1, h_q, d]

            # Get block indices for this batch
            num_blocks_needed = (seq_len + block_size - 1) // block_size
            block_indices = block_table[i, :num_blocks_needed]

            # Gather keys for this batch using block indices
            k_i = k_cache[block_indices]  # [num_blocks_needed, block_size, 1, d]
            k_i = k_i.view(-1, h_kv, d)[:seq_len]  # [seq_len, 1, d]

            # Repeat k for multi-query attention
            k_i = k_i.repeat_interleave(h_q, dim=1)  # [seq_len, h_q, d]

            # Compute attention scores
            scores = torch.einsum("nhd,khd->nhk", q_i, k_i)  # [1, h_q, seq_len]

            # Apply causal mask if needed
            if causal:
                scores.masked_fill_(
                    torch.arange(seq_len, device=device) > seq_len - 1, float("-inf")
                )

            # Compute log sum exp
            lse[i] = scores.logsumexp(dim=-1, keepdim=True)  # [1, h_q, 1]

            # Compute attention weights
            attn_weights = torch.softmax(scores, dim=-1)  # [1, h_q, seq_len]

            # Compute output
            # Use head_dim_v instead of full dimension for values
            v_i = k_i[..., :head_dim_v]  # [seq_len, h_q, head_dim_v]
            out[i] = torch.einsum(
                "nhk,khd->nhd", attn_weights, v_i
            )  # [1, h_q, head_dim_v]

        # out: [b, 1, h_q, head_dim_v]
        # lse: [b, h_q, 1]
        return out, lse

    def get_key_value_from_cache(
        self,
        key_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        head_num_q: int,
        head_dim_qk: int,
        head_dim_v: int,
        batch_size: int,
        block_size: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = seq_lens.sum().item()
        key = torch.empty((num_tokens, head_num_q, head_dim_qk), dtype=key_cache.dtype)
        value = torch.empty((num_tokens, head_num_q, head_dim_v), dtype=key_cache.dtype)

        token_acc = 0
        masks = torch.empty((batch_size, max_seq_len, max_seq_len), dtype=torch.float16)

        for i in range(batch_size):
            seq_len = seq_lens[i].item()

            num_blocks_needed = (seq_len + block_size - 1) // block_size
            block_indices = block_table[i, :num_blocks_needed]

            key_i = key_cache[block_indices]  # [num_blocks_needed, block_size, 1, d]
            key_i = key_i.view(-1, 1, head_dim_qk)[
                :seq_len
            ]  # [seq_len, 1, head_dim_qk=576]

            key_i = key_i.repeat_interleave(
                head_num_q, dim=1
            )  # [seq_len, head_num_q, head_dim_qk]

            value_i = key_i[..., :head_dim_v]  # [seq_len, head_num_q, head_dim_qk]

            key[token_acc : token_acc + seq_len] = key_i
            value[token_acc : token_acc + seq_len] = value_i
            masks[i, :seq_len, :seq_len] = 1.0
            token_acc += seq_len

        # key: [num_tokens, head_num_q, head_dim_qk]
        # value: [num_tokens, head_num_q, head_dim_v]
        return key, value, masks

    # https://github.com/vllm-project/vllm/blob/79a1d25bbd09420ad7d8631671a70cbcaf496cfb/tests/kernels/attention/test_merge_attn_states.py#L15
    def merge_attn_states_torch(
        self,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Reference implementation.
        dtype = prefix_output.dtype
        max_lse = torch.maximum(prefix_lse, suffix_lse)
        p_lse = torch.exp(prefix_lse - max_lse)
        s_lse = torch.exp(suffix_lse - max_lse)
        p_scale = p_lse / (p_lse + s_lse)
        s_scale = s_lse / (p_lse + s_lse)
        p_scale = p_scale.transpose(0, 1).unsqueeze(2)
        s_scale = s_scale.transpose(0, 1).unsqueeze(2)
        output = p_scale * prefix_output + s_scale * suffix_output
        output = output.to(dtype)
        return output

    @SupportedDevices(["Ascend910B"])
    # @pytest.mark.parametrize("batch_size", [128])
    # @pytest.mark.parametrize("seq_len_q", [1])  # MTP = 1, 2
    # @pytest.mark.parametrize("mean_seq_len_k", [4096, 8192])
    # @pytest.mark.parametrize("num_heads_q", [16, 32, 64, 128])  # TP = 8, 4, 2, 1
    # @pytest.mark.parametrize("head_dim", [576])
    # @pytest.mark.parametrize("head_dim_v", [512])
    # @pytest.mark.parametrize("block_size", [64])
    # @pytest.mark.parametrize("varlen", [False, True])
    # @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_ring_attention(self):
        causal = True
        varlen = True
        num_heads_kv = 1

        batch_size = 128
        seq_len_q = 1
        mean_seq_len_k = 4096
        num_heads_q = 16
        head_dim = 576
        head_dim_v = 512
        block_size = 64
        device = torch.device("npu")
        dtype = torch.float16

        # Generate sequence lengths
        cache_seqlens = torch.full(
            (batch_size,), mean_seq_len_k, dtype=torch.int32, device=device
        )

        if varlen:
            for i in range(batch_size):
                cache_seqlens[i] = max(
                    int(random.normalvariate(mean_seq_len_k, mean_seq_len_k / 2)),
                    seq_len_q,
                )

        max_seqlen = cache_seqlens.max().item()
        max_seqlen_pad = ((max_seqlen + 255) // 256) * 256

        # Generate query tensor
        q = torch.randn(
            batch_size, seq_len_q, num_heads_q, head_dim, device=device, dtype=dtype
        )

        # Generate block table
        num_blocks = batch_size * max_seqlen_pad // block_size
        block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)
        block_table = block_table.view(batch_size, max_seqlen_pad // block_size)

        # Generate key cache
        k_cache = torch.randn(
            num_blocks, block_size, num_heads_kv, head_dim, device=device, dtype=dtype
        )

        # set padding values to NaN
        for i in range(batch_size):
            k_cache.view(batch_size, max_seqlen_pad, num_heads_kv, head_dim)[
                i, cache_seqlens[i].item() :
            ] = float("nan")

        # calculate scale
        scale = head_dim**-0.5

        out_torch, lse_torch = self.flash_mla_with_kvcache_torch(
            q=q,
            k_cache=k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            softmax_scale=scale,
            causal=causal,
        )

        key, value, masks = self.get_key_value_from_cache(
            k_cache,
            cache_seqlens,
            block_table,
            num_heads_q,
            head_dim,
            head_dim_v,
            batch_size,
            block_size,
            max_seqlen_pad,
        )

        num_tokens = cache_seqlens.sum().item()
        out_npu = torch.empty(
            num_tokens, num_heads_q, head_dim_v, dtype=dtype, device=device
        )
        lse_npu = torch.empty(num_tokens, num_heads_q, dtype=dtype, device=device)

        query = q.view(batch_size, num_heads_q, head_dim)
        seq_lens_query = torch.ones(batch_size, dtype=torch.int32, device=device)
        seq_lens = torch.stack([seq_lens_query, cache_seqlens], dim=0).cpu()
        torch_npu._npu_ring_attention(
            query,
            key,
            value,
            masks,
            seq_lens,
            num_heads_q,
            num_heads_kv,
            1.0,
            out_npu,
            lse_npu,
        )

        print(out_torch.shape, lse_torch.shape)


if __name__ == "__main__":
    run_tests()
