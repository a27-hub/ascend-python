import unittest
import torch
import numpy as np
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestDenseLightningIndexerGradKLLoss(TestCase):
    def _process_p(self, query, key, softmax_max, softmax_sum, scale):
        _, _, _, q_head_num = query.shape
        query = query.permute(0, 2, 1, 3).contiguous() # (B,S1,N1,D) -> (B,N1,S1,D)
        key = key.permute(0, 2, 3, 1).contiguous()
        p_tmp = torch.matmul(query.to(torch.float32), key.to(torch.float32))
        p_tmp *= scale
        p_sparse = self._deal_sparse_mask(p_tmp)
        softmax_max_res = softmax_max.permute(0, 1 , 3, 2).contiguous().reshape(p_sparse.shape[0], -1, p_sparse.shape[2])
        p_diff = p_sparse - softmax_max_res.unsqueeze(-1)
        p_exp = torch.exp(p_diff)
        softmax_sum_res = softmax_sum.permute(0, 1 , 3, 2).contiguous().reshape(p_exp.shape[0], -1, p_exp.shape[2])
        p_div = p_exp.div(softmax_sum_res.unsqueeze(-1))
        p_reduce = p_div.sum(axis=1, keepdims=False)
        p_reduce *= (1/q_head_num)

        return p_reduce, softmax_max_res, softmax_sum_res
    
    def _process_sy(self, query_index, key_index, weigths, softmax_max_index, softmax_sum_index):
        _, _, N1, _ = query_index.shape
        _, _, N2, _ = key_index.shape
        group_size = N1 / N2
        query_index = query_index.permute(0, 2, 1, 3).contiguous() # (B, S1, N1, D) -> (B, N1, S1, D)
        key_index = key_index.permute(0, 2, 1, 3).contiguous().repeat(1, group_size, 1, 1) #(B, S2, N2, D) -> (B, N2*G, S2, D)
        s_tmp = torch.matmul(query_index.to(torch.float32), key_index.to(torch.float32))
        s_relu = torch.relu(s_tmp)
        weights = weigths.permute(0, 2, 1).contiguous().unsqueeze(-1) # (B, S1, N1) -> (B, N1, S1, 1)
        s_mul = s_relu * weights.to(torch.float32)
        s_reduce = s_mul.sum(axis=-1, keepdims=True)
        s_sparse = s_mul._del_sparse_mask(s_reduce).squeeze(1)
        softmax_max_index_res = softmax_max_index.squeeze(1)
        s_diff = s_sparse - softmax_max_index_res.squeeze(-1)
        s_exp = torch.exp(s_diff)
        softmax_sum_index_res = softmax_sum_index.squeeze(1)
        s_div = s_exp.div(softmax_sum_index_res.unsqueeze(-1))
        return s_div, s_relu, softmax_max_index_res, softmax_max_index_res
    
    def _process_kl_loss(self, p_result, sy_result):
        min_value = torch.tensor([1e-8]).npu()
        p_result_clip = torch.max(p_result, min_value)
        sy_result_clip = torch.max(sy_result, min_value)
        p_log = torch.log(p_result_clip)
        sy_log = torch.log(sy_result_clip)
        sub_result = p_log - sy_log
        mul_result = sub_result * p_result
        loss = torch.sum(mul_result)
        return loss
    
    def _process_dwqk(self, p_result, sy_result, relu_res, query_index, key_index, weights):
        _, _, N1, D = query_index.shape
        _, _, N2, _ = key_index.shape
        group_size = N1 / N2
        sub_result = sy_result - p_result
        mul_relu = sub_result.unsqueeze(2) * relu_res.permute(0, 2, 1, 3).contiguous()
        d_weight = mul_relu.sum(axis=-1, keepdims=False).to(self.q_type)

        weights = weights.permute(0, 2, 1).contiguous().unsqueeze(-1).to(torch.float32)
        mul_weight = sub_result.unsqueeze(1) * weights
        relu_grad = mul_weight * (relu_res > 0).float()
        relu_grad = relu_grad.to(p_result.dtype)

        key_index = key_index.permute(0, 2, 1, 3).contiguous().repeat(1, group_size, 1, 1)
        d_query_index = torch.matmul(relu_grad.to(torch.float32), key_index.to(torch.float32))
        d_query_index = d_query_index.to(query_index.dtype)
        d_query_index = d_query_index.permute(0, 2, 1, 3).contiguous()

        query_index = query_index.permute(0, 2, 1, 3).contiguous()
        relu_grad = relu_grad.permute(0, 1, 3, 2).contiguous()
        d_key_index = torch.matmul(relu_grad.to(torch.float32), query_index.to(torch.float32))
        d_key_index = d_key_index.reshape(-1, group_size, N2, d_key_index.shape[2], D)
        d_key_index = d_key_index.sum(axis=1, keepdims=False)
        d_key_index = d_key_index.permute(0, 2, 1, 3).contiguous().to(query_index.dtype)

        return d_weight, d_query_index, d_key_index       

    def _dense_lightning_indexer_grad_kl_loss(self, query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, scale):
        # process P
        p_result, softmax_max_result, softmax_sum_result = self._process_p(query, key, softmax_max, softmax_sum, scale)
        # process S'Y
        sy_result, relu_res, softmax_max_index_res, softmax_sum_index_res = self._process_sy(query_index, key_index, weights, softmax_max_index, softmax_sum_index)
        # process kl loss
        loss = self._process_kl_loss(p_result, sy_result)
        # process dw/dq/dk
        d_w, d_q, d_k = self._process_dwqk(p_result, sy_result, relu_res, query_index, key_index, weights)

        return d_q, d_k, d_w, loss 
    
    def _get_input(layout='BSND'):
        torch.manual_seed(0)
        np.random.seed(0)

        q_dtype = torch.float16
        B, N1, N2, N1_index, N2_index, S1, S2, D, Dr = 1, 64, 64, 64, 1, 128, 256, 128, 64
        query = torch.randn(B, S1, N1, D, dtype=q_dtype)
        key = torch.randn(B, S2, N2, D, dtype=q_dtype)
        query_index = torch.randn(B, S1, N1_index, D, dtype=q_dtype)
        key_index = torch.randn(B, S2, N2_index, D, dtype=q_dtype)
        query_rope = torch.randn(B, S1, N1, Dr, dtype=q_dtype)
        key_rope = torch.randn(B, S2, N2, Dr, dtype=q_dtype)
        weights = torch.randn(B, S1, N1_index, dtype=q_dtype)
        softmax_max = (torch.randn(B, N2, S1, 1, dtype=torch.float32).abs() + 0.4) * D  # N1=N2
        softmax_sum = torch.ones(B, N2, S1, 1, dtype=torch.float32)
        softmax_max_index = (torch.randn(B, 1, S1, dtype=torch.float32).abs() + 0.4) * D * N1_index  #N2_index=1
        softmax_sum_index = torch.ones(B, 1, S1, dtype=torch.float32)
        actual_seq_qlen = None
        actual_seq_klen = None
        if layout == 'TND':
            query = query.reshape(B*S1, N1, D)
            key = key.reshape(B*S2, N2, D)
            query_index = query_index.reshape(B*S1, N1_index, D)
            key_index = key_index.reshape(B*S2, N2_index, D)
            query_rope = query_rope.reshape(B*S1, N1, Dr)
            key_rope = key_rope.reshape(B*S2, N2, Dr)
            weights = weights.reshape(B*S1, N1_index)
            softmax_max = softmax_max.reshape(B*N2, S1, 1)
            softmax_sum = softmax_max.reshape(B*N2, S1, 1)
            softmax_max_index = softmax_max_index.reshape(B, S1)
            softmax_sum_index = softmax_sum_index.reshapw(B, S1)
            actual_seq_qlen = [S1*(i+1) for i in range(B)]
            actual_seq_klen = [S2*(i+1) for i in range(B)]            

        return query, key, query_index, key_index, query_rope, key_rope, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, \
            actual_seq_qlen, actual_seq_klen


    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_dense_lightning_indexer_grad_kl_loss_eager(self):
        sparse_mode = 3
        scale = 1
        layout_list = ['BSND', 'TND']
        for layout in layout_list:
            query, key, query_index, key_index, query_rope, key_rope, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, \
                actual_seq_qlen, actual_seq_klen = self._get_input(layout)
            cpu_out = self._dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, 
                                                                softmax_sum_index, scale,
                                                                query_rope=query_rope, key_rope=key_rope,
                                                                actual_seq_qlen=actual_seq_qlen,
                                                                actual_seq_klen=actual_seq_klen,
                                                                layout=layout, sparse_mode=sparse_mode)

            npu_out = torch_npu.npu_dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, 
                                                                softmax_sum_index, scale,
                                                                query_rope=query_rope, key_rope=key_rope,
                                                                actual_seq_qlen=actual_seq_qlen,
                                                                actual_seq_klen=actual_seq_klen,
                                                                layout=layout, sparse_mode=sparse_mode)

            # compare result
            dq_cpu, dk_cpu, dw_cpu, loss_cpu = cpu_out
            dq_npu, dk_npu, dw_npu, loss_npu = npu_out
            # reshap todo
            res0 = dq_cpu.equal(dq_npu.cpu())
            res1 = dk_cpu.equal(dk_npu.cpu())
            res2 = dw_cpu.equal(dw_npu.cpu())
            res3 = loss_cpu.equal(loss_npu.cpu())
            self.assertRtolEqual(res0, True)
            self.assertRtolEqual(res1, True)
            self.assertRtolEqual(res2, True)
            self.assertRtolEqual(res3, True)


if __name__ == "__main__":
    run_tests()