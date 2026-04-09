import unittest
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAttentionPioneerMetadata(TestCase):

    def test_npu_attention_pioneer_metadata_basic(self, device="npu"):
        """Basic functional test: verify output shape and dtype."""
        actual_seq_q = torch.tensor([32, 64], dtype=torch.int64).npu()
        actual_seq_kv = torch.tensor([128, 256], dtype=torch.int64).npu()

        meta_data = torch_npu._npu_attention_pioneer_metadata(
            num_heads_q=32,
            num_heads_kv=8,
            head_dim_qk=128,
            head_dim_v=128,
            actual_seq_lengths=actual_seq_q,
            actual_seq_lengths_kv=actual_seq_kv,
            batch_size=2,
            sparse_mode=0,
            input_layout="TND",
        )
        self.assertEqual(meta_data.shape, torch.Size([1024]))
        self.assertEqual(meta_data.dtype, torch.int32)

    def test_npu_attention_pioneer_metadata_no_actual_seq(self, device="npu"):
        """Test without actual_seq_lengths (non-TND layout scenario)."""
        meta_data = torch_npu._npu_attention_pioneer_metadata(
            num_heads_q=16,
            num_heads_kv=16,
            head_dim_qk=64,
            head_dim_v=64,
            batch_size=4,
            sparse_mode=3,
            input_layout="BSH",
        )
        self.assertEqual(meta_data.shape, torch.Size([1024]))
        self.assertEqual(meta_data.dtype, torch.int32)

    def test_npu_attention_pioneer_metadata_invalid_heads(self, device="npu"):
        """Test that invalid num_heads raises error."""
        with self.assertRaises(RuntimeError):
            torch_npu._npu_attention_pioneer_metadata(
                num_heads_q=0,
                num_heads_kv=8,
                head_dim_qk=128,
                head_dim_v=128,
            )

    def test_npu_attention_pioneer_metadata_heads_not_divisible(self, device="npu"):
        """Test that num_heads_q not divisible by num_heads_kv raises error."""
        with self.assertRaises(RuntimeError):
            torch_npu._npu_attention_pioneer_metadata(
                num_heads_q=7,
                num_heads_kv=3,
                head_dim_qk=128,
                head_dim_v=128,
            )

    def test_npu_attention_pioneer_metadata_invalid_sparse_mode(self, device="npu"):
        """Test that invalid sparse_mode raises error."""
        with self.assertRaises(RuntimeError):
            torch_npu._npu_attention_pioneer_metadata(
                num_heads_q=32,
                num_heads_kv=8,
                head_dim_qk=128,
                head_dim_v=128,
                sparse_mode=5,
            )

    def test_npu_attention_pioneer_metadata_invalid_block_size(self, device="npu"):
        """Test that non-16-aligned block_size raises error."""
        with self.assertRaises(RuntimeError):
            torch_npu._npu_attention_pioneer_metadata(
                num_heads_q=32,
                num_heads_kv=8,
                head_dim_qk=128,
                head_dim_v=128,
                block_size=17,
            )

    def test_npu_attention_pioneer_metadata_sparse_modes(self, device="npu"):
        """Test all valid sparse modes (0-4)."""
        for mode in range(5):
            meta_data = torch_npu._npu_attention_pioneer_metadata(
                num_heads_q=32,
                num_heads_kv=8,
                head_dim_qk=128,
                head_dim_v=128,
                batch_size=1,
                sparse_mode=mode,
                input_layout="TND",
            )
            self.assertEqual(meta_data.shape, torch.Size([1024]))


if __name__ == "__main__":
    run_tests()
