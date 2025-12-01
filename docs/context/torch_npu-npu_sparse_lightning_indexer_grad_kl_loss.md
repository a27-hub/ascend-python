# torch_npu.npu_sparse_lightning_indexer_grad_kl_loss

## 产品支持情况
| 产品                           | 是否支持 |
|------------------------------| :------: |
| <term>Atlas A3 训练系列产品</term> | √  |
| <term>Atlas A2 训练系列产品</term> | √  |

## 功能说明

- API功能：该接口实现了npu_lightning_indexer的反向功能，再额外融合了Loss的计算功能。npu_lightning_indexer将Attention的query和Attention的key之间的最高内在联系的top_k个筛选出来，存放在sparse_indices中，从而减少长序列场景下Attention的计算量，加速长序列的网络的推理和训练的性能。

- 计算公式：
  用于取Top-k的value的计算公式可以表示为：
  $$
  I_{t,:}=W_{t,:}@ReLU(q_{t,:}@(K_{:t,:})^T)
  $$
  其中， $W$是第$t$个token对应的weights，$q$是第$t$个token对应的$G$个query头合轴后的矩阵，$K$为$t$行$K$矩阵。
  
  LightningIndexer会单独训练，对应的loss function为：
  $$
  L(I){=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
  $$
  其中，$p$是target distribution，通过对main attention score 进行所有的head的求和，然后把求和结果沿着上下文方向进行L1正则化得到。$D_{KL}$为KL散度，其表达式为：
  $$
  D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
  $$
  通过求导可得Loss的梯度表达式：
  $$
  dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$
  利用链式法则可以进行weights，query和key矩阵的梯度计算：
  $$
  dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left) \left) \mathop{{}}\nolimits^{{T}}\right. \right. \right. \right.
  $$
  $$
  d\mathop{{q}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@K\mathop{{}}\nolimits_{{:t,:}}
  $$
  $$
  dK\mathop{{}}\nolimits_{{:t,:}}=\left(dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{{T}}@q\mathop{{}}\nolimits_{{:t, :}}\right. \right.
  $$
  其中，S为QK矩阵softmax的结果。

## 函数原型

```
npu_sparse_lightning_indexer_grad_kl_loss(Tensor query, Tensor key, Tensor query_index, Tensor key_index, Tensor weights, Tensor sparse_indices, Tensor softmax_max, Tensor softmax_sum, float scale_value=1, *, Tensor? query_rope=None, Tensor? key_rope=None, SymInt[]? actual_seq_qlen=None, SymInt[]? actual_seq_klen=None, str? layout='BSND', int? sparse_mode=3, int? pre_tokens=9223372036854775807, int? next_tokens=9223372036854775807) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明
**query**(`Tensor`)：必选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, S1, N1, D)`、`(T1, N1, D)`。

**key**(`Tensor`)：必选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, S2, N2, D)`、`(T2, N2, D)`。

**query_index**(`Tensor`)：必选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, S1, N1index, D)`、`(T1, N1index, D)`。

**key_index**(`Tensor`)：必选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, S1, N1index, D)`、`(T1, N1index, D)`。

**weights**(`Tensor`)：必选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, S1, N1)`、`(T1, N1)`。

**sparse_indices**(`Tensor`)：必选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, S1, topK)`、`(T1, topK)`。

**softmax_max**(`Tensor`)：必选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, N2, S1, G)`、`(N2, T1, G)`。

**softmax_sum**(`Tensor`)：必选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, N2, S1, G)`、`(N2, T1, G)`。

**scale_value**(`float`)：必选参数，表示缩放系数，数据类型支持`float`。

**query_rope**(`Tensor`)：可选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, S1, N1, Dr)`、`(T1, N1, Dr)`。

**key_rope**(`Tensor`)：可选参数，数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持输入shape`(B, S2, N2, Dr)`、`(T2, N2, Dr)`。

**actual_seq_qlen**(`list[int]`)：可选参数，int类型数组，TND场景时需传入此参数。表示query每个S的累加和长度，数据类型支持`int64`，数据格式支持ND，默认值为`None`。

**actual_seq_klen**(`list[int]`)：可选参数，int类型数组，TND场景时需传入此参数。表示key每个S的累加和长度，数据类型支持`int64`，数据格式支持ND，默认值为`None`。

**layout**(`str`)：可选参数，用于标识输入`query`的数据排布格式，数据类型支持`str`。当前支持BSND、TND，默认值为"BSND"。

**sparse_mode**(`int`)：可选参数，表示sparse的模式，数据类型支持`int32`.默认值为`3`。

**pre_tokens**(`int`)：必选参数，数据类型支持`int64`，默认值`9223372036854775807`。

**next_tokens**(`int`)：必选参数，数据类型支持`int64`，默认值`9223372036854775807`。




## 返回值说明
-   **d\_query\_index**(`Tensor`)：表示query_index的梯度，数据类型支持`bfloat16`、`float16`。
-   **d\_key\_index**(`Tensor`)：表示key_index的梯度，数据类型支持`bfloat16`、`float16`。
-   **d\_weights**(`Tensor`)：表示weights的梯度，数据类型支持`bfloat16`、`float16`。
-   **loss**(`Tensor`)：表示网络正向输出和golden值的差异，数据类型支持`float`。


## 调用示例
- 单算子模式调用

    ```python
    import torch
    import torch_npu

    def gen_inputs(seqlens_list_array, seqlens_list_kv_array, isTnd):
        B = 1
        NQuery = 64
        NQueryIndex = 64
        N2 = 1
        S1 = 128
        S2 = 128
        topK = 2048
        D = 512
        DIndex = 128
        DR = 64
        output_dtype = torch.float16
        q = torch.randn(B, S1, NQuery, D, dtype=output_dtype, device=torch.device('npu'))
        k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))
    
        q_index = torch.randn(B, S1, NQueryIndex, DIndex, dtype=output_dtype, device=torch.device('npu'))
        k_index = torch.randn(B, S2, N2, DIndex, dtype=output_dtype, device=torch.device('npu'))
        if DR != 0:
            q_rope = torch.randn(B, S1, NQuery, DR, dtype=output_dtype, device=torch.device('npu'))
            k_rope = torch.randn(B, S2, N2, DR, dtype=output_dtype, device=torch.device('npu'))
        else:
            q_rope = None
            k_rope = None
        weights = torch.randn(B, S1, NQueryIndex, dtype=output_dtype, device=torch.device('npu'))
        a = -0.05  # 最小值
        b = 0.05 # 最大值
        kk = 3.0  # 控制分布范围（3σ 覆盖绝大多数值）
        scale = (b - a) / (2 * kk)
        shift = (a + b) / 2
        weights = weights * scale + shift
        if isTnd:
            sparse_indices = torch.zeros(S1, N2, topK).to(torch.int32).npu()
            tIdx = 0
            for bIdx in range(B):
                for s1Idx in range(seqlens_list_array[bIdx]):
                    s2RealSize = (int)((seqlens_list_kv_array[bIdx] - seqlens_list_array[bIdx]) + s1Idx + 1)
                    if s2RealSize <= 0:
                        s2RealSize = seqlens_list_kv_array[bIdx]
                    
                    if s2RealSize > topK:
                        s2RealLen = topK
                    else:
                        s2RealLen = s2RealSize
                    #处理S2无效行场景，把对应的sparse indices置为-1
                    sparse_indices[tIdx, :, 0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
                    sparse_indices[tIdx, :, s2RealLen : topK] = -1
                    tIdx = tIdx + 1
            q_tnd = q.squeeze(dim=0)
            k_tnd = k.squeeze(dim=0)
            q_index_tnd = q_index.squeeze(dim=0)
            k_index_tnd = k_index.squeeze(dim=0)
            if q_rope is not None:
                q_rope_tnd = q_rope.squeeze(dim=0)
                k_rope_tnd = k_rope.squeeze(dim=0)
            else :
                q_rope_tnd = None
                k_rope_tnd = None
            weights_tnd = weights.squeeze(dim=0)
            
            softmax_max = torch.randn(N2, S1, NQueryIndex, dtype=torch.float, device=torch.device('npu'))
            softmax_sum = torch.randn(N2, S1, NQueryIndex, dtype=torch.float, device=torch.device('npu'))
            return q_tnd, k_tnd, q_index_tnd, k_index_tnd, q_rope_tnd, k_rope_tnd, weights_tnd, sparse_indices, softmax_max, softmax_sum
        else :
            sparse_indices = torch.zeros(B, S1, N2, topK).to(torch.int32).npu()
            for s1Idx in range(S1):
                s2RealSize = (int)(S2 - S1 + s1Idx + 1)
                if s2RealSize <= 0:
                    s2RealSize = S2
    
                if s2RealSize > topK:
                    s2RealLen = topK
                else:
                    s2RealLen = s2RealSize
                sparse_indices[:, s1Idx, 0,  0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
                sparse_indices[:, s1Idx, 0,  s2RealLen : topK] = -1
    
            softmax_max = torch.randn(B, N2, S1, NQueryIndex, dtype=torch.float, device=torch.device('npu'))
            softmax_sum = torch.randn(B, N2, S1, NQueryIndex, dtype=torch.float, device=torch.device('npu'))
            return q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum


        actual_seq_qlen = [128]
        actual_seq_kvlen = [128]
        input_layout = 'TND'
        isTnd = True
        sparse_mode = 3
        scale = 1.0
        q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum = gen_inputs(actual_seq_qlen, actual_seq_kvlen, isTnd)
        
        torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
                q, k, q_index, k_index, weights, sparse_indices, softmax_max, softmax_sum, scale,
                query_rope=q_rope, key_rope=k_rope, actual_seq_qlen=actual_seq_qlen,
                actual_seq_klen=actual_seq_kvlen, layout=input_layout, sparse_mode=sparse_mode,
                pre_tokens=65536, next_tokens=65536)
    ```
