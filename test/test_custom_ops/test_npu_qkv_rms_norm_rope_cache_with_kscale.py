import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUQkvRmsNormRopeCacheWithKScale(TestCase):

    def _generate_inputs(self, T=4, Nq=8, Nk=2, Nv=2, D=128, BlockNum=10, BlockSize=16, MaxSeqLen=32, Batch=2):
        qkv = torch.randn(T, Nq + Nk + Nv, D, dtype=torch.bfloat16, device='npu')
        q_gamma = torch.randn(D, dtype=torch.float32, device='npu')
        k_gamma = torch.randn(D, dtype=torch.float32, device='npu')
        cos_sin = torch.randn(MaxSeqLen, D, dtype=torch.float32, device='npu')
        slot_mapping = torch.arange(T, dtype=torch.int32, device='npu')
        k_cache = torch.zeros(BlockNum, Nk, BlockSize, D, dtype=torch.float8_e4m3fn, device='npu')
        v_cache = torch.zeros(BlockNum, Nv, BlockSize, D, dtype=torch.float8_e4m3fn, device='npu')
        k_scale_cache = torch.ones(BlockNum, Nk, BlockSize, 1, dtype=torch.float32, device='npu')
        cu_seq_len = torch.zeros(Batch + 1, dtype=torch.int32, device='npu')
        tokens_per_batch = T // Batch
        for i in range(1, Batch + 1):
            cu_seq_len[i] = min(i * tokens_per_batch, T)
        actual_seq_lens = cu_seq_len[1:] - cu_seq_len[:-1]
        head_nums = [Nq, Nk, Nv]
        q_rotation = torch.randn(D, D, dtype=torch.bfloat16, device='npu')
        k_rotation = torch.randn(D, D, dtype=torch.bfloat16, device='npu')
        v_scale = torch.ones(Nv, dtype=torch.float32, device='npu')
        return (qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
                k_scale_cache, cu_seq_len, actual_seq_lens, head_nums, q_rotation, k_rotation, v_scale)

    @SupportedDevices(['Ascend950'])
    def test_normal_inplace(self, device="npu"):
        (qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
         k_scale_cache, cu_seq_len, actual_seq_lens, head_nums, q_rotation, k_rotation, v_scale) = self._generate_inputs()

        q_out, q_scale = torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
            qkv, q_gamma, k_gamma, cos_sin, slot_mapping,
            k_cache, v_cache, k_scale_cache, cu_seq_len, actual_seq_lens, head_nums,
            q_rotation=q_rotation, k_rotation=k_rotation, v_scale=v_scale)

        T, Nq, Nk, Nv, D = qkv.shape[0], head_nums[0], head_nums[1], head_nums[2], qkv.shape[2]
        self.assertEqual(q_out.shape, (T, Nq, D))
        self.assertEqual(q_out.dtype, torch.float8_e4m3fn)
        self.assertEqual(q_scale.shape, (T, Nq))
        self.assertEqual(q_scale.dtype, torch.float32)

    @SupportedDevices(['Ascend950'])
    def test_normal_functional(self, device="npu"):
        (qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
         k_scale_cache, cu_seq_len, actual_seq_lens, head_nums, q_rotation, k_rotation, v_scale) = self._generate_inputs()

        q_out, q_scale, k_cache_out, v_cache_out, k_scale_cache_out = \
            torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale_functional(
                qkv, q_gamma, k_gamma, cos_sin, slot_mapping,
                k_cache, v_cache, k_scale_cache, cu_seq_len, actual_seq_lens, head_nums,
                q_rotation=q_rotation, k_rotation=k_rotation, v_scale=v_scale)

        T, Nq, Nk, Nv, D = qkv.shape[0], head_nums[0], head_nums[1], head_nums[2], qkv.shape[2]
        BlockNum, BlockSize = k_cache.shape[0], k_cache.shape[2]

        self.assertEqual(q_out.shape, (T, Nq, D))
        self.assertEqual(q_out.dtype, torch.float8_e4m3fn)
        self.assertEqual(q_scale.shape, (T, Nq))
        self.assertEqual(q_scale.dtype, torch.float32)
        self.assertEqual(k_cache_out.shape, (BlockNum, Nk, BlockSize, D))
        self.assertEqual(k_cache_out.dtype, torch.float8_e4m3fn)
        self.assertEqual(v_cache_out.shape, (BlockNum, Nv, BlockSize, D))
        self.assertEqual(v_cache_out.dtype, torch.float8_e4m3fn)
        self.assertEqual(k_scale_cache_out.shape, (BlockNum, Nk, BlockSize, 1))
        self.assertEqual(k_scale_cache_out.dtype, torch.float32)

    @SupportedDevices(['Ascend950'])
    def test_missing_q_rotation(self, device="npu"):
        inputs = self._generate_inputs()
        with self.assertRaisesRegex(RuntimeError, "q_rotation"):
            torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
                *inputs[:11], q_rotation=None, k_rotation=inputs[12], v_scale=inputs[13])

    @SupportedDevices(['Ascend950'])
    def test_missing_k_rotation(self, device="npu"):
        inputs = self._generate_inputs()
        with self.assertRaisesRegex(RuntimeError, "k_rotation"):
            torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
                *inputs[:11], q_rotation=inputs[11], k_rotation=None, v_scale=inputs[13])

    @SupportedDevices(['Ascend950'])
    def test_missing_v_scale(self, device="npu"):
        inputs = self._generate_inputs()
        with self.assertRaisesRegex(RuntimeError, "v_scale"):
            torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
                *inputs[:11], q_rotation=inputs[11], k_rotation=inputs[12], v_scale=None)

    @SupportedDevices(['Ascend950'])
    def test_invalid_head_nums_length(self, device="npu"):
        inputs = self._generate_inputs()
        with self.assertRaisesRegex(RuntimeError, "head_nums"):
            torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
                *inputs[:10], head_nums=[1, 2],
                q_rotation=inputs[11], k_rotation=inputs[12], v_scale=inputs[13])

    @SupportedDevices(['Ascend950'])
    def test_nv_not_equal_nk(self, device="npu"):
        inputs = self._generate_inputs()
        with self.assertRaisesRegex(RuntimeError, "Nv"):
            torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
                *inputs[:10], head_nums=[4, 2, 3],
                q_rotation=inputs[11], k_rotation=inputs[12], v_scale=inputs[13])

    @SupportedDevices(['Ascend950'])
    def test_actual_seq_lens_shape_mismatch(self, device="npu"):
        inputs = self._generate_inputs()
        actual_seq_lens_wrong = torch.ones(inputs[8].numel(), dtype=torch.int32, device='npu')
        with self.assertRaisesRegex(RuntimeError, "actual_seq_lens"):
            torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
                *inputs[:9], actual_seq_lens_wrong, inputs[10],
                q_rotation=inputs[11], k_rotation=inputs[12], v_scale=inputs[13])

    @SupportedDevices(['Ascend950'])
    def test_qkv_shape_mismatch(self, device="npu"):
        inputs = self._generate_inputs()
        qkv_wrong = torch.randn(4, 5, 128, dtype=torch.bfloat16, device='npu')
        with self.assertRaisesRegex(RuntimeError, "qkv"):
            torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
                qkv_wrong, inputs[1], inputs[2], inputs[3], inputs[4],
                inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], inputs[10],
                q_rotation=inputs[11], k_rotation=inputs[12], v_scale=inputs[13])


if __name__ == "__main__":
    run_tests()
