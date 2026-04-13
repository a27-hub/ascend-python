import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


class TestNPUQuantFlashAttentionV2(TestCase):
    def supported_op_exec(self, query, key, value, dy, drop_mask=None, keep_prob=1.0):
        scale = 0.08838
        qk = torch.matmul(query, key.transpose(2, 3)).mul(scale)
        softmax_res, x_max, x_sum = tsoftmax(qk.to(torch.float32))
        dp = torch.matmul(dy, value.transpose(2, 3))
        if drop_mask == None or len(drop_mask.shape) == 0:
            drop_res = softmax_res
            dp_drop = dp
        else:
            drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob)
            dp_drop = dp * drop_mask * (1.0 / keep_prob)
        y = torch.matmul(drop_res, value)
        dv = torch.matmul(drop_res.transpose(2, 3), dy)
        softmax_grad_res = (tsoftmax_grad(dp_drop, softmax_res) * scale)
        dq = torch.matmul(softmax_grad_res, key)
        dk = torch.matmul(softmax_grad_res.transpose(2, 3), query)       
        dq = dq.transpose(1, 2)
        dq = dq.reshape(dq.shape[0], dq.shape[1], -1)
        dk = dk.transpose(1, 2)
        dk = dk.reshape(dk.shape[0], dk.shape[1], -1)
        dv = dv.transpose(1, 2)
        dv = dv.reshape(dv.shape[0], dv.shape[1], -1)
        return y, softmax_res, x_max, x_sum, dq, dk, dv

    def custom_op_exec(self, query, key, value, dy, softmax_max, softmax_sum, attention_in, keep_prob=1.0, numels=0, seed=2):
        scale = 0.08838
        return torch_npu.npu_quant_fusion_attention_backward(
            query, key, value, dy, head_num=32, input_layout="BSND", softmax_max=softmax_max, softmax_sum=softmax_sum,
            attention_in=attention_in, scale_value=scale, keep_prob=keep_prob, numels=numels, seed=seed)

    def npu_block_quant(self, tensor, scale = 1,PER_BLOCK_SIZE = 512):
        dim1 = tensor.shape[0]
        dim2 = tensor.shape[1]
        dim3 = tensor.shape[2]
        dim4 = tensor.shape[3]
        quanted_tensor = torch.zeros([dim1, dim2, dim3, dim4]).to(torch.float32)
        for b in range(dim1):
            for n in range(dim3):
                for s in range (0,dim2,PER_BLOCK_SIZE):
                    s_start = s // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                    s_end = min (s // PER_BLOCK_SIZE * PER_BLOCK_SIZE + PER_BLOCK_SIZE, dim2)
                    quanted_tensor[b, s_start:s_end, n, :] = tensor[b, s_start:s_end, n, :] * scale
        return quanted_tensor
    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_v2(self, device="npu"):
        query = torch.randn(1, 7200, 40, 128, dtype=torch.float32)
        key = torch.randn(1, 512, 40, 128, dtype=torch.float32)
        value = torch.randn(1, 512, 40, 128, dtype=torch.float32)
        dy = torch.randn(1, 7200, 40, 128, dtype=torch.float32)
        x_max = torch.randn(1, 40, 7200, 1, dtype=torch.float32)
        x_sum = torch.randn(1, 40, 7200, 1, dtype=torch.float32)
        attention_in = torch.randn(1, 7200, 40, 128, dtype=torch.bfloat16)
        dscale_q = torch.randn(1, 40, 15, 1, dtype=torch.float32)
        dscale_k = torch.randn(1, 40, 1, 1, dtype=torch.float32)
        dscale_v = torch.randn(1, 40, 1, 1, dtype=torch.float32)
        dscale_dy = torch.randn(1, 40, 15, 1, dtype=torch.float32)
        query_dtype=torch_npu.hifloat8
        head_num = 40
        scale = 1

        query_hifp8 = self.npu_block_quant(query).to(torch.uint8)
        key_hifp8 = self.npu_block_quant(key).to(torch.uint8)
        value_hifp8 = self.npu_block_quant(value).to(torch.uint8)
        dy_hifp8 = self.npu_block_quant(dy).to(torch.uint8)

        p_scale = torch.tensor([15]).float()
        ds_scale = torch.tensor([2**15]).float()
        npu_result = torch_npu.npu_quant_fusion_attention_backward(query_hifp8.npu(), key_hifp8.npu(), value_hifp8.npu(), dy_hifp8.npu(),head_num,
                                                                "BSND",
                                                                dscale_q.npu(),
                                                                dscale_k.npu(),
                                                                dscale_v.npu(),
                                                                dscale_dy.npu(),
                                                                p_scale=p_scale.npu(),
                                                                ds_scale=ds_scale.npu(),
                                                                softmax_max=x_max.npu(),
                                                                softmax_sum=x_sum.npu(),
                                                                attention_in=attention_in.npu(),
                                                                scale_value=scale,
                                                                query_dtype=query_dtype)

if __name__ == "__main__":
    run_tests()
