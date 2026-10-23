import numpy as np
import torch

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNativeMultiHeadAttention(TestCase):
    # ATen 算子 _native_multi_head_attention：自注意力路径。
    # 权重约定（对齐 native_multi_head_attention_cpu）：
    #   query/key/value: [B, S, D]
    #   qkv_weight: [3D, D], qkv_bias: [3D]
    #   proj_weight: [D, D], proj_bias: [D]
    # mask 为 bool，True == 屏蔽。mask_type: 1 == padding mask [B, S]；2 == generic mask [B, N, S, S]。

    def _mha(self, query, embed_dim, num_head, qkv_weight, qkv_bias,
             proj_weight, proj_bias, mask, mask_type):
        return torch.ops.aten._native_multi_head_attention(
            query, query, query, embed_dim, num_head,
            qkv_weight, qkv_bias, proj_weight, proj_bias,
            mask, False, True, mask_type)[0]

    def _gen_inputs(self, batch, seq_len, embed_dim, num_head, dtype):
        inputs = {}
        for name, shape, low, high in [
            ("query", [batch, seq_len, embed_dim], -1.0, 1.0),
            ("qkv_weight", [3 * embed_dim, embed_dim], -0.1, 0.1),
            ("qkv_bias", [3 * embed_dim], -0.1, 0.1),
            ("proj_weight", [embed_dim, embed_dim], -0.1, 0.1),
            ("proj_bias", [embed_dim], -0.1, 0.1),
        ]:
            cpu, npu = create_common_tensor([dtype, -1, shape], low, high)
            inputs[name] = (cpu, npu)
        return inputs

    def _padding_mask(self, batch, seq_len, pad_len):
        mask = np.zeros((batch, seq_len), dtype=np.bool_)
        if pad_len > 0:
            mask[:, -pad_len:] = True
        return mask

    def _generic_mask(self, batch, num_head, seq_len, pad_len):
        mask = np.zeros((batch, num_head, seq_len, seq_len), dtype=np.bool_)
        if pad_len > 0:
            mask[:, :, :, -pad_len:] = True
        return mask

    def test_native_multi_head_attention_no_mask_float32(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float32)

        cpu_out = self._mha(inputs["query"][0], embed_dim, num_head,
                            inputs["qkv_weight"][0], inputs["qkv_bias"][0],
                            inputs["proj_weight"][0], inputs["proj_bias"][0],
                            None, None)
        npu_out = self._mha(inputs["query"][1], embed_dim, num_head,
                            inputs["qkv_weight"][1], inputs["qkv_bias"][1],
                            inputs["proj_weight"][1], inputs["proj_bias"][1],
                            None, None)
        self.assertRtolEqual(cpu_out, npu_out.cpu(), prec=1.e-3)

    def test_native_multi_head_attention_padding_mask_float32(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float32)

        mask = self._padding_mask(batch, seq_len, pad_len=2)
        cpu_mask = torch.from_numpy(mask)
        npu_mask = torch.from_numpy(mask).to("npu")

        cpu_out = self._mha(inputs["query"][0], embed_dim, num_head,
                            inputs["qkv_weight"][0], inputs["qkv_bias"][0],
                            inputs["proj_weight"][0], inputs["proj_bias"][0],
                            cpu_mask, 1)
        npu_out = self._mha(inputs["query"][1], embed_dim, num_head,
                            inputs["qkv_weight"][1], inputs["qkv_bias"][1],
                            inputs["proj_weight"][1], inputs["proj_bias"][1],
                            npu_mask, 1)
        self.assertRtolEqual(cpu_out, npu_out.cpu(), prec=1.e-3)

    def test_native_multi_head_attention_generic_mask_float32(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float32)

        mask = self._generic_mask(batch, num_head, seq_len, pad_len=2)
        cpu_mask = torch.from_numpy(mask)
        npu_mask = torch.from_numpy(mask).to("npu")

        cpu_out = self._mha(inputs["query"][0], embed_dim, num_head,
                            inputs["qkv_weight"][0], inputs["qkv_bias"][0],
                            inputs["proj_weight"][0], inputs["proj_bias"][0],
                            cpu_mask, 2)
        npu_out = self._mha(inputs["query"][1], embed_dim, num_head,
                            inputs["qkv_weight"][1], inputs["qkv_bias"][1],
                            inputs["proj_weight"][1], inputs["proj_bias"][1],
                            npu_mask, 2)
        self.assertRtolEqual(cpu_out, npu_out.cpu(), prec=1.e-3)

    def test_native_multi_head_attention_no_mask_float16(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float16)

        cpu_out = self._mha(inputs["query"][0], embed_dim, num_head,
                            inputs["qkv_weight"][0], inputs["qkv_bias"][0],
                            inputs["proj_weight"][0], inputs["proj_bias"][0],
                            None, None)
        npu_out = self._mha(inputs["query"][1], embed_dim, num_head,
                            inputs["qkv_weight"][1], inputs["qkv_bias"][1],
                            inputs["proj_weight"][1], inputs["proj_bias"][1],
                            None, None)
        self.assertRtolEqual(cpu_out, npu_out.cpu(), prec16=1.e-2)


if __name__ == "__main__":
    run_tests()
