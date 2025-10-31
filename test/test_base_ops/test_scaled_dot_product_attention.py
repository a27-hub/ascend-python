import unittest
import torch
import numpy as np
import torch.nn
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_utils import SupportedDevices


class TestScaledDotProductAttention(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_sdpa_fia(self):
        shape_format1 = [
            [[np.float16, 0, (1, 3, 10, 32)], [np.float16, 0, (1, 3, 10, 32)], [np.float16, 0, (1, 3, 10, 32)]],
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -1, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], -1, 1)
            cpu_output = torch.nn.functional.scaled_dot_product_attention(cpu_input1.to(torch.float32), cpu_input2.to(torch.float32), cpu_input3.to(torch.float32))
            npu_output = torch.nn.functional.scaled_dot_product_attention(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output.to(torch.float16), npu_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_sdpa_attn_mask_dim_3(self):
        shape_format1 = [
            [[np.float16, 0, (1, 3, 32, 32)], [np.float16, 0, (1, 3, 32, 32)], [np.float16, 0, (1, 3, 32, 32)], [np.float16, 0, (1, 32, 32)]],
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -1, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], -1, 1)
            cpu_input4, npu_input4 = create_common_tensor(item[3], -1, 1)

            cpu_output = torch.nn.functional.scaled_dot_product_attention(cpu_input1.to(torch.float32), cpu_input2.to(torch.float32), cpu_input3.to(torch.float32),  attn_mask=cpu_input4.bool())
            npu_output = torch.nn.functional.scaled_dot_product_attention(npu_input1, npu_input2, npu_input3,  attn_mask=npu_input4.bool())
            self.assertRtolEqual(cpu_output.to(torch.float16), npu_output, 0.001)

    @unittest.skipIf(torch.__version__ < "2.5.1", "enable_gqa is only supported on torch's version >= 2.5")
    @SupportedDevices(['Ascend910B'])
    def test_sdpa_attn_enable_gqa(self):
        query = torch.rand(32, 32, 64, 64, dtype=torch.float16)
        key = torch.rand(32, 4, 64, 64, dtype=torch.float16)
        value = torch.rand(32, 4, 64, 64, dtype=torch.float16)
        attn_mask = torch.rand(32, 64, 64, dtype=torch.float16).bool()
        cpu_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, enable_gqa=True)
        npu_output = torch.nn.functional.scaled_dot_product_attention(query.npu(), key.npu(), value.npu(), attn_mask=attn_mask.npu(), enable_gqa=True)
        self.assertRtolEqual(cpu_output, npu_output, 0.001)


    def fa_func_eager(self, q, k, v, attn_mask):
        with sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
            set_priority=True,
        ):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

    @torch.compile(backend="aot_eager")
    def fa_func(self, q, k, v, attn_mask):
        with sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
            set_priority=True,
        ):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

    @unittest.skipIf(torch.__version__ < "2.6.0", "enable_gqa is only supported on torch's version >= 2.6")
    @SupportedDevices(['Ascend910B'])
    def test_sdpa_graph(self):
        dtype = torch.half
        Eqk, Ev = 64, 64
        Sq, Skv = 38, 38
        Hq, Hkv = 32, 32
        for B in [1, 2, 4]:
            q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype).npu()
            k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype).npu()
            v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype).npu()
            attn_mask = torch.zeros(B, Hq, Sq, Skv, dtype=torch.bool).npu()
            res_eager = self.fa_func_eager(q, k, v, attn_mask)
            res = self.fa_func(q, k, v, attn_mask)
        self.assertRtolEqual(res_eager, res, 0.001)


if __name__ == "__main__":
    run_tests()
