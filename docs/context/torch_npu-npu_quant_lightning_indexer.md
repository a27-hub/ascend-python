# torch_npu.npu_quant_lightning_indexer

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明

-   API功能：QuantLightningIndexer是推理场景下，SparseFlashAttention（SFA）前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。主要计算过程为
    -   将某个 token 对应的输入Index Query $Q_{index}^{INT8}\in\R^{g\times d}$ 乘以给定上下文 Index Key $K_{index}^{INT8}\in\R^{S_{k}\times d}$ ，得到相关性，然后与Index Query和Index Key的反量化系数 $ Scale_Q $ 和 $ Scale_K^T $ 相乘，通过激活函数 $ \text{ReLU}\ $ 过滤无效负相关信号后，得到当前 Token 与所有前序 Token 的相关性分数向量。将其与权重系数 $ (W@[1]_{1\times S_{k}}) $相乘后，沿g的方向，选取前 $ \text{Top-k}\ $ 个索引值得到输出 $ out $ ，作为SparseFlashAttention的输入。
    

-   计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{INT8}@{\left(K_{index}^{INT8}\right)}^T\right)\right)\right]\right\}$$


## 函数原型

```
torch_npu.npu_quant_indexer_lightning(query, key, weights, query_dequant_scale, key_dequant_scale, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, query_quant_mode=0, key_quant_mode=0, layout_query='BSND', layout_key='BSND', sparse_count=2048, sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807) -> Tensor
```

## 参数说明

-   **query**（`Tensor`）：必选参数，对应公式中的 $Q_{index}^{INT8}\in\R^{g\times d}$ 。不支持非连续，数据格式支持$ND$，数据类型支持`int8`，支持输入shape[B,S1,N1,D]、[T,N1,D]。
    
-   **key**（`Tensor`）：必选参数，对应公式中的 $K_{index}^{INT8}\in\R^{S_{k}\times d}$ 。不支持非连续，数据格式支持$ND$，数据类型支持`int8`，layout\_key为PA_BSND时shape为[block\_count, block\_size, N2, D]，其中block\_count为PageAttention时block总数，block\_size为一个block的token数。
    
-   **weights**（`Tensor`）：必选参数，对应公式中的 $ W $ 。不支持非连续，数据格式支持$ND$，数据类型支持`float16`，支持输入shape[B,S1,N1]、[T,N1]。

-   **query_dequant_scale**（`Tensor`）：必选参数，对应公式中的$ Scale_Q $ 。不支持非连续，数据格式支持$ND$，数据类型支持`float16`，支持输入shape[B,S1,N1]、[T,N1]。

-   **key_dequant_scale**（`Tensor`）：必选参数，对应公式中的 $ Scale_K^T $ 。不支持非连续，数据格式支持$ND$，数据类型支持`float16`，layout\_key为PA_BSND时shape为[block\_count, block\_size, N2]，其中block\_count为PageAttention时block总数，block\_size为一个block的token数。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。
    -   该入参中每个Batch的有效token数不超过`query`中的维度S大小。支持长度为B的一维tensor。当`query`的input\_layout为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值。不能出现负值。

-   **actual\_seq\_lengths\_key**（`Tensor`）：可选参数，表示不同Batch中`key`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和key的shape的S长度相同。支持长度为B的一维tensor。

-   **block\_table**（`Tensor`）：可选参数，表示PageAttention中KV存储使用的block映射表，数据格式支持$ND$，数据类型支持`int32`。
    -   PageAttention场景下，block\_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq(maxBlockNumPerSeq为每个batch中最大actual\_seq\_lengths\_key对应的block数量)

-   **query\_quant\_mode**（`int`）：可选参数，用于标识输入`query`的量化模式，当前支持Per-Token-Head量化模式，默认值0。

    -   query\_quant\_mode为0: Per-Token-Head。

-   **key\_quant\_mode**（`int`）：可选参数，用于标识输入`key`的量化模式，当前支持Per-Token-Head量化模式，默认值0。

-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，当前支持BSND、TND，默认值"BSND"。

-   **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，当前支持PA_BSND、BSND、TND，默认值"BSND"。在非PageAttention场景下，layout\_key应与layout\_query保持一致。

-   **sparse\_count**（`int`）：可选参数，代表topK阶段需要保留的block数量，支持1-2048，数据类型支持`int32`。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式，支持0/3，数据类型支持`int32`。
    
    -   sparse\_mode为0时，代表defaultMask模式。
    -   sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

-   **pre\_tokens**（`int`）：预留可选参数，默认值9223372036854775807。

-   **next\_tokens**（`int`）：预留可选参数，默认值9223372036854775807。

## 返回值说明

-   **out**（`Tensor`）：公式中的输出，数据类型支持`int32`。数据格式支持ND，支持输出shape[B,S1,N2,k]，[T,N2,k]

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   参数query中的N支持64，key中的N支持1。
-   参数query中的D和参数key中的D值相等为128。
-   支持block_size取值为16的整数倍，最大支持到1024。
-   该接口要求$W \odot Scale_Q$的结果可以被`float16`表示。
-   该接口要求**query_dequant_scale**、 **key_dequant_scale**的值均大于等于0。
-   该接口的TopK过程对NAN排序是未定义行为。


## 调用示例<a name="zh-cn_topic_0000002313328922_section983519211229"></a>

-   单算子模式调用
     ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math
    import unittest

    from torch_npu.testing.common_utils import SupportedDevices
    from torch_npu.testing.testcase import TestCase, run_tests



    class TestCustomQuantLightningIndexer(TestCase):
        def _get_data_from_pa_cache(self, key, block_table, act_s2):
            block_num, block_size, n2, d = key.shape
            if n2 != 1:
                raise ValueError("n2 only support 1")
            need_blcok_num = (act_s2 + block_size - 1) // block_size
            act_s2_align = need_blcok_num * block_size
            out = torch.zeros((act_s2_align, d), dtype=key.dtype, device=key.device)
            for i in range(need_blcok_num):
                out[i * block_size:(i + 1) * block_size, :] = key[block_table[i], ...].reshape(block_size, d)

            return out[:act_s2, :]


        def _get_k_scale(self, key_dequant_scale, block_table, act_s2):
            block_num, block_size, n2 = key_dequant_scale.shape
            if n2 != 1:
                raise ValueError("n2 only support 1")
            need_blcok_num = (act_s2 + block_size - 1) // block_size
            act_s2_align = need_blcok_num * block_size
            out = torch.zeros((act_s2_align), dtype=key_dequant_scale.dtype, device=key_dequant_scale.device)
            key_dequant_scale = key_dequant_scale.reshape(block_num, block_size)
            for i in range(need_blcok_num):
                out[i * block_size:(i + 1) * block_size] = key_dequant_scale[block_table[i], ...].reshape(block_size)

            return out[:act_s2]


        def _quant_lightning_indexer(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query,
                                    actual_seq_lengths_key, block_table,
                                    layout_query="BSND", sparse_count=2048, sparse_mode=3):
            batch_size = query.shape[0]
            if layout_query == "TND":
                batch_size = actual_seq_lengths_query.shape[0]
            out_shape = list(query.shape)
            n2 = key.shape[2]
            d = query.shape[-1]
            n1 = query.shape[-2]
            out_shape[-1] = sparse_count
            out_shape[-2] = n2
            out = torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(-1, n2, sparse_count) - 1
            act_s1 = 0
            act_s2 = 0
            process_q_len = 0
            for batch_id in range(batch_size):
                if actual_seq_lengths_query is None:
                    act_s1 = query.shape[1]
                else:
                    if layout_query == "TND":
                        act_s1 = actual_seq_lengths_query[batch_id] - process_q_len
                    else:
                        act_s1 = actual_seq_lengths_query[batch_id]
                act_s2 = actual_seq_lengths_key[batch_id]
                # n1, s1, d
                now_q = query.reshape(-1, n1, d)[process_q_len:process_q_len + act_s1, :, :].transpose(0, 1).to(torch.int32)
                now_weights = weights.reshape(-1, n1, 1)[process_q_len:process_q_len + act_s1, :, :]
                now_query_scale = query_dequant_scale.reshape(-1, n1, 1)[process_q_len:process_q_len + act_s1, :, :]
                # s1, n1, 1
                weights_scale = now_weights * now_query_scale # float16
                process_q_len += act_s1
                now_block_table = block_table[batch_id, :]
                # d s2
                now_k = self._get_data_from_pa_cache(key, now_block_table, act_s2).transpose(0, 1).to(torch.int32)
                # s2
                now_k_scale = self._get_k_scale(key_dequant_scale, now_block_table, act_s2).to(torch.float32)
                # n1,s1,d @ d,s2 -> n1,s1,s2
                s_out = (torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0)).to(torch.float32)) / 1024.0
                # n1,s1,s2 -> s1,n1,s2  to fp16
                s_out = s_out.to(torch.float16).transpose(0, 1).to(torch.float32)
                # s1,n1,1 -> s1,1,n1
                weights_scale = weights_scale.transpose(1, 2).to(torch.float32)
                # s1,1,n1 @ s1,n1,s2 -> s1,1,s2 -> s1,s2
                topk_in = torch.bmm(weights_scale, s_out).squeeze(1)
                # s1,s2 * s2
                topk_in = topk_in * now_k_scale
                tmp_s1 = topk_in.shape[0]
                tmp_s2 = topk_in.shape[1]
                if sparse_mode == 3:
                    for i in range(tmp_s1):
                        topk_in[-1 - i, tmp_s2 - i:] = float('-inf')
                sorted_value, sorted_indices = torch.sort(topk_in, dim=1, descending=True, stable=True)
                if sparse_mode == 3:
                    for i in range(tmp_s1):
                        sorted_indices[-1 - i, tmp_s2 - i:] = -1
                return_s2 = min(sparse_count, tmp_s2)
                out[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]

            out = out.reshape(out_shape)
            return out

        def cpu_op_exec(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query,
                        actual_seq_lengths_key, block_table, layout_query, sparse_count, sparse_mode):
            output = self._quant_lightning_indexer(query, key, weights, query_dequant_scale, key_dequant_scale,
                                            actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                                            layout_query, sparse_count, sparse_mode)
            output = output.cpu()

            return output

        def npu_op_exec_eager(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query,
                            actual_seq_lengths_key, block_table, query_quant_mode, key_quant_mode, layout_query,
                            layout_key, sparse_count, sparse_mode):
            npu_out = torch_npu.npu_quant_lightning_indexer(query, key, weights, query_dequant_scale, key_dequant_scale,
                                                            actual_seq_lengths_query=actual_seq_lengths_query,
                                                            actual_seq_lengths_key=actual_seq_lengths_key,
                                                            block_table=block_table,
                                                            query_quant_mode=query_quant_mode,
                                                            key_quant_mode=key_quant_mode,
                                                            layout_query=layout_query,
                                                            layout_key=layout_key, sparse_count=sparse_count,
                                                            sparse_mode=sparse_mode)
            npu_out = npu_out.cpu()

            return npu_out

        def quant_lightning_indexer_result(self, layout_query, b, t, s1, s2, act_seq_q, act_seq_k, sparse_mode,
                                        sparse_count = 2048):
            n1 = 64
            n2 = 1
            d = 128
            block_size = 128
            layout_key = 'PA_BSND'
            query_quant_mode = 0
            key_quant_mode = 0
            np.random.seed(0)
            # -------------
            max_block_table_num = (s2 + block_size - 1) // block_size
            block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
            key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8)
            key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2)))
            key_dequant_scale = key_dequant_scale.to(torch.float16)
            if layout_query == 'BSND':
                query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
                query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16)
                weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
                actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                                        if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
                actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                                        if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)
                print(f"------- test LIQuant BSND case b:{b} s1:{s1} s2:{s2} sparse_mode:{sparse_mode} ----------")
            else:
                query = torch.tensor(np.random.uniform(-128, 127, (t, n1, d))).to(torch.int8)
                query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (t, n1))).to(torch.float16)
                weights = torch.tensor(np.random.uniform(0, 0.01, (t, n1))).to(torch.float16)
                actual_seq_lengths_query = torch.tensor(act_seq_q).to(torch.int32)
                actual_seq_lengths_key = torch.tensor(act_seq_k).to(torch.int32)
                print(f"------- test LIQuant TND case b:{b} t:{t} s2:{s2} act_seq_q:{act_seq_q} act_seq_k:{act_seq_k} ",
                    f"sparse_mode:{sparse_mode} -------")

            cpu_out = self.cpu_op_exec(query.cpu(), key.cpu(), weights.cpu(), query_dequant_scale.cpu(), key_dequant_scale.cpu(),
                                    actual_seq_lengths_query.cpu(), actual_seq_lengths_key.cpu(), block_table.cpu(),
                                    layout_query, sparse_count, sparse_mode)

            npu_eager_out = self.npu_op_exec_eager(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(), key_dequant_scale.npu(),
                                                actual_seq_lengths_query.npu(), actual_seq_lengths_key.npu(), block_table.npu(),
                                                query_quant_mode, key_quant_mode,
                                                layout_query, layout_key, sparse_count, sparse_mode)
            res = npu_eager_out.equal(cpu_out)
            self.assertRtolEqual(res, True)

        @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
        def test_quant_lightning_indexer(self):
            # layout_query, b, t, s1, s2, act_seq_q, act_seq_k, sparse_mode
            test_case_list = [
                ("BSND", 24, None, 4, 512, None, None, 0),
            ]
            for case in test_case_list:
                self.quant_lightning_indexer_result(*case)


    if __name__ == "__main__":
        run_tests()

