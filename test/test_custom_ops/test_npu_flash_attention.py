import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from einops import rearrange


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu


def broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B, S, D = kv_shape[0], kv_shape[2], kv_shape[3]
    kv_res = torch.zeros([B, numHeads, S, D]).to(dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    return y.div(x_sum), x_max, x_sum


def tforward(q, k, v, scale):
    qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    softmax_res, x_max, x_sum = tsoftmax(qk)
    y = torch.matmul(softmax_res, v)
    return y, softmax_res, x_max, x_sum


class TestNPUFlashAttention(TestCase):
    def supported_op_exec(self, query, key, value, scale):
        qk = torch.matmul(query, key.transpose(2, 3)).mul(scale)
        softmax_res = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        output = torch.matmul(softmax_res, value)
        output = output.transpose(1, 2)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        return output

    # pylint:disable = huawei-too-many-arguments
    def supported_op_exec_with_softmax_out(self, q, k, v, seqlens_list_q, seqlens_list_k, input_layout,
                                           tnd_softmax_out=1):
        gtype = torch.float32
        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)

        # 从输入推导维度
        S1, N1, D = q.shape
        S2, N2, _ = k.shape
        B = len(seqlens_list_q)
        scale = 1 / (D ** 0.5)

        out_golden = torch.zeros_like(q)
        x_max_tnd = torch.empty(0)
        x_sum_tnd = torch.empty(0)

        for i in range(B):
            if seqlens_list_q[i] != 0 and seqlens_list_k[i] != 0:
                qi = rearrange(q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]], 's n d -> 1 n s d')
                ki = rearrange(k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]], 's n d -> 1 n s d')
                vi = rearrange(v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]], 's n d -> 1 n s d')

                if N1 != N2:
                    ki = broadcastKV_sigle(N1, N2, ki, ki.dtype)
                    vi = broadcastKV_sigle(N1, N2, vi, vi.dtype)

                outi_golden, _, x_maxi, x_sumi = tforward(qi.to(gtype), ki.to(gtype), vi.to(gtype), scale)
                out_golden[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(outi_golden, '1 n s d -> s n d')

                x_maxi = x_maxi.broadcast_to(1, N1, seqlens_list_q[i], 8)
                x_sumi = x_sumi.broadcast_to(1, N1, seqlens_list_q[i], 8)

                if input_layout == "TND" and tnd_softmax_out == 1:
                    x_maxi_tnd = x_maxi.permute(0, 2, 1, 3).contiguous().view(-1)
                    x_sumi_tnd = x_sumi.permute(0, 2, 1, 3).contiguous().view(-1)
                    x_max_tnd = torch.cat([x_max_tnd, x_maxi_tnd], dim=0)
                    x_sum_tnd = torch.cat([x_sum_tnd, x_sumi_tnd], dim=0)

        return out_golden.detach(), x_max_tnd.detach().reshape(S1, N1, 8), x_sum_tnd.detach().reshape(S1, N1, 8)

    # pylint:disable = huawei-too-many-arguments
    def custom_op_exec(self, query, key, value, head_num, scale, input_layout, actual_seq_qlen=None,
                       actual_seq_kvlen=None, softmax_layout=""):
        return torch_npu.npu_fusion_attention(
            query, key, value, head_num=head_num, input_layout=input_layout, scale=scale,
            actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, softmax_layout=softmax_layout)

    def trans_BNSD2BSH(self, tensor: torch.Tensor):
        tensor = torch.transpose(tensor, 1, 2)
        tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
        return tensor

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention(self, device="npu"):
        query = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 128, 128, dtype=torch.float16)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        output = self.supported_op_exec(query.to(torch.float32),
                                        key.to(torch.float32),
                                        value.to(torch.float32),
                                        scale=0.08838).to(torch.float16)
        result = self.custom_op_exec(q_npu, k_npu, v_npu, head_num=32, scale=0.08838, input_layout="BSH")
        attention_score = result[0]
        self.assertRtolEqual(output, attention_score, prec=0.005, prec16=0.005)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tnd(self, device="npu"):
        B, N1, N2, D = 3, 8, 2, 128
        scale = 1 / (D ** 0.5)
        seqlens_list_q = np.array([1, 2, 3])
        seqlens_list_k = np.array([3, 4, 5])
        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
        S1 = seqlens_list_q.sum()
        S2 = seqlens_list_k.sum()
        pttype = torch.float16
        q = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)

        attention_score_golden, softmax_max_golden, softmax_sum_golden = (
            self.supported_op_exec_with_softmax_out(q, k, v, seqlens_list_q, seqlens_list_k, "TND"))
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        torch.npu.manual_seed(2)
        torch.npu.synchronize()
        result = self.custom_op_exec(
            q, k, v, N1,
            scale=scale,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
            softmax_layout="TND")

        attention_score_v4 = result[0]
        softmax_max_v4 = result[1]
        softmax_sum_v4 = result[2]
        torch.npu.synchronize()

        self.assertRtolEqual(attention_score_golden, attention_score_v4, prec=0.005, prec16=0.005)
        self.assertRtolEqual(softmax_max_golden, softmax_max_v4, prec=0.005, prec16=0.005)
        self.assertRtolEqual(softmax_sum_golden, softmax_sum_v4, prec=0.005, prec16=0.005)


if __name__ == "__main__":
    run_tests()

