import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from typing import Optional


class TestPagedCacheLoadSeqStarts(TestCase):

    def _create_mla_cache(
        self,
        num_blocks: int,
        block_size: int,
        entry_size: int,
        device: str,
    ) -> torch.Tensor:
        return torch.randn(
            num_blocks, block_size, entry_size, dtype=torch.float16, device=device
        )

    def _gather_cache_torch(
        src_cache: torch.Tensor,  # [NUM_BLOCKS, BLOCK_SIZE, HEAD, ENTRIES]
        dst: torch.Tensor,  # [TOT_TOKENS, ENTRIES]
        block_table: torch.Tensor,  # [BATCH, BLOCK_INDICES]
        cu_seq_lens: torch.Tensor,  # [BATCH+1]
        batch_size: int,
        seq_starts: Optional[torch.Tensor] = None,  # Optional: [BATCH]
    ) -> None:
        """
        Gather sequence data from source cache to destination tensor
        Args:
            src_cache: Source cache tensor [NUM_BLOCKS, BLOCK_SIZE, HEAD, ENTRIES]
            dst: Destination tensor [TOT_TOKENS, ENTRIES]
            block_table: Block table mapping [BATCH, BLOCK_INDICES]
            cu_seq_lens: Cumulative sequence lengths [BATCH+1]
            batch_size: Batch size
            seq_starts: Optional, starting offsets for each batch [BATCH]
        """
        # Basic parameter checks
        assert src_cache.dtype == dst.dtype, "src_cache and dst must have same dtype"
        assert block_table.dtype == torch.int32, "block_table must be int32"
        assert cu_seq_lens.dtype == torch.int32, "cu_seq_lens must be int32"

        if seq_starts is not None:
            assert seq_starts.dtype == torch.int32, "seq_starts must be int32"

        block_size = src_cache.size(1)
        # Process each batch
        for bid in range(batch_size):
            # Get sequence start and end positions for current batch
            seq_start = cu_seq_lens[bid].item()
            seq_end = cu_seq_lens[bid + 1].item()
            seq_len = seq_end - seq_start

            if seq_len == 0:
                continue

            # Calculate required number of blocks
            tot_blocks = (seq_len + block_size - 1) // block_size

            # Calculate block offset if seq_starts is provided
            offset = 0
            if seq_starts is not None:
                offset = seq_starts[bid].item() // block_size

            # Get block table for current batch
            batch_block_table = block_table[bid, offset : offset + tot_blocks]
            # Calculate complete blocks and last partial block
            full_blocks = tot_blocks - 1 if seq_len % block_size else tot_blocks
            partial_block_size = seq_len % block_size if seq_len % block_size else 0
            # Copy complete blocks
            dst_start = seq_start
            for i in range(full_blocks):
                block_id = batch_block_table[i].item()
                # Copy entire block, remove HEAD dimension
                dst[dst_start : dst_start + block_size] = src_cache[block_id].squeeze(1)
                dst_start += block_size

            # Handle last incomplete block
            if partial_block_size > 0:
                block_id = batch_block_table[full_blocks].item()
                dst[dst_start : dst_start + partial_block_size] = src_cache[
                    block_id, :partial_block_size
                ].squeeze(1)

    @SupportedDevices(["Ascend910B"])
    def test_atb_paged_cache_load(
        self,
        kv_lora_rank,
        qk_rope_head_dim,
        block_size,
        num_blocks,
        max_seq_len,
        batch_size,
    ):
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        block_size = 16
        num_blocks = 1024
        max_seq_len = 512
        batch_size = 8
        device = "npu"
        entry_size = kv_lora_rank + qk_rope_head_dim

        src_cache = self._create_mla_cache(num_blocks, block_size, entry_size, device)
        seq_len_tensor = torch.randint(0, max_seq_len + 1, (batch_size,), device=device)

        total_tokens = seq_len_tensor.sum()
        cu_seq_lens = torch.empty((batch_size + 1), dtype=torch.int32, device=device)
        cu_seq_lens[0] = 0
        cu_seq_lens[1:] = seq_len_tensor.cumsum(dim=0).to(dtype=torch.int32)

        # tot_blocks_tensor = (seq_len_tensor + block_size - 1) // block_size
        block_table = torch.empty(
            (batch_size, num_blocks), dtype=torch.int32, device=device
        )

        for b in range(batch_size):
            perm = torch.randperm(num_blocks, device=device)
            block_table[b, :] = perm

        expected = torch.zeros(
            (total_tokens, entry_size), dtype=src_cache.dtype, device=device
        )
        # generate seq_starts
        max_start = max_seq_len // 2
        seq_starts = torch.randint(
            0, max_start + 1, (batch_size,), dtype=torch.int32, device=device
        )

        # gather_cache_torch(src_cache, expected, block_table, cu_seq_lens, batch_size, seq_starts)
        self.gather_cache_torch(
            src_cache, expected, block_table, cu_seq_lens, batch_size, seq_starts
        )

        key_nope = torch.empty(
            (total_tokens, kv_lora_rank), dtype=torch.float16, device=device
        )
        key_rope = torch.empty(
            (total_tokens, qk_rope_head_dim), dtype=torch.float16, device=device
        )
        key_nope_cache, key_rope_cache = src_cache.split(
            [kv_lora_rank, qk_rope_head_dim], dim=2
        )
        key_nope_cache = key_nope_cache.view(
            num_blocks, block_size, kv_lora_rank // 16, 16
        )
        key_rope_cache = key_rope_cache.view(
            num_blocks, block_size, qk_rope_head_dim // 16, 16
        )
        key_nope_cache = key_nope_cache.permute(0, 2, 1, 3).contiguous()
        key_rope_cache = key_rope_cache.permute(0, 2, 1, 3).contiguous()

        torch_npu._npu_paged_cache_load(
            key_nope_cache,
            key_rope_cache,
            torch.tensor(block_table).int(),
            seq_len_tensor.int(),
            key_nope,
            key_rope,
        )

        key = torch.cat([key_nope, key_rope], dim=1)
        self.assertRtolEqual(expected, key)


if __name__ == "__main__":
    run_tests()
