# torch_npu.npu_block_sparse_attention_backward

## 产品支持情况

| 产品                                                   | 是否支持 |
|:-----------------------------------------------------|:----:|
| 昇腾910_95 AI处理器                                       |  ×   |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                      |  √   |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |  √   |
| Atlas 200I/500 A2 推理产品                               |  ×   |
| Atlas 推理系列产品                                         |  ×   |
| Atlas 训练系列产品                                         |  ×   |
| Atlas 200I/300/500 推理产品                              |  ×   |

## 功能说明

- **API 功能**：BlockSparseAttention 稀疏注意力反向计算，块级稀疏模式与正向一致，通过 `block_sparse_mask` 指定稀疏块，计算 query、key、value 的梯度。

- **计算公式**：已知正向计算公式为：
  
    $$
    attentionOut=Softmax(Mask(scale⋅query⋅key_{sparse}^{T},  atten\_mask))⋅value_{sparse}
    $$

    为方便表达，以变量$S$和$P$表示计算公式：

    $$
    S = Mask(scale⋅query⋅key_{sparse}^{T}+atten\_mask)
    $$

    $$
    P = SoftMax(S)
    $$

    $$
    Out = PV
    $$
  
    则反向计算公式为：
    $$
    softmax\_grad = softmaxGrad(dOut, attentionOut)
    $$

    $$
    dP=dOut * V^T
    $$
        
    $$
    dS = P * (dP-softmax\_grad)
    $$
        
    $$
    dV=P^T * dOut
    $$
        
    $$
    dQ=(dS*K)*scale
    $$
        
    $$
    dK=(dS^T*Q)*scale
    $$

- 输入 d_out、query、key、value、attention_out 的排布由 `q_input_layout`、`kv_input_layout` 指定，当前仅支持 `"TND"`、`"BNSD"`。

## 函数原型

```
torch_npu.npu_block_sparse_attention_backward(d_out, query, key, value, attention_out, softmax_lse, block_sparse_mask, block_shape=None, actual_seq_lengths=None, actual_seq_lengths_kv=None, q_input_layout, kv_input_layout, num_key_value_heads, scale_value) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **d_out** (`Tensor`)：必选，正向输出的梯度。shape 与正向 attention_out 一致，数据格式 ND，数据类型 `float16` 或 `bfloat16`。

- **query** (`Tensor`)：必选，正向输入的 query。shape 与正向一致，TND 为 3 维，BNSD 为 4 维。

- **key** (`Tensor`)：必选，正向输入的 key。

- **value** (`Tensor`)：必选，正向输入的 value。

- **attention_out** (`Tensor`)：必选，正向输出的注意力结果（npu_block_sparse_attention 的第一个返回值）。

- **softmax_lse** (`Tensor`)：必选，正向输出的 softmax log-sum-exp（npu_block_sparse_attention 的第二个返回值），**数据类型为 float32**。
  - TND 时，shape 为 `[totalQTokens, headNum, 1]`；
  - BNSD 时，shape 为 `[batch, headNum, maxQSeqLength, 1]`。

- **block_sparse_mask** (`Tensor`)：必选，与正向相同的块稀疏掩码。shape 为 `[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]`。

- **block_shape** (`list[int]`, 可选)：若传入则至少两个元素 `[blockShapeX, blockShapeY]` 且均大于 0；**blockShapeY 必须为 128 的倍数**。

- **actual_seq_lengths** (`list[int]`)：描述每个 batch 的 query 实际序列长度，用于变长序列场景。
  - **当 `q_input_layout` 为 `"TND"` 时必选**：TND 下 query 的 shape 为 `[totalQTokens, headNum, headDim]`，无 batch 维，算子无法从 shape 推断各 batch 的 query 长度。
  - **当为 `"BNSD"` 时可选**：BNSD 下 query 的 shape 为 `[batch, headNum, maxQSeqLength, headDim]`，不传时算子按 shape 中的 S（maxQSeqLength）作为序列长度处理；传入时按本参数指定的实际长度处理。

- **actual_seq_lengths_kv** (`list[int]`)：描述每个 batch 的 key/value 实际序列长度，用于变长序列场景。
  - **当 `kv_input_layout` 为 `"TND"` 时必选**：TND 下 key/value 的 shape 为 `[totalKvTokens, numKeyValueHeads, headDim]`，无 batch 维，算子无法从 shape 推断各 batch 的 kv 长度。
  - **当为 `"BNSD"` 时可选**：BNSD 下 key/value 的 shape 为 `[batch, numKeyValueHeads, maxKvSeqLength, headDim]`，不传时算子按 shape 中的 S 作为序列长度处理；传入时按本参数指定的实际长度处理。

- **q_input_layout** (`str`)：必选，仅支持 `"TND"`、`"BNSD"`。

- **kv_input_layout** (`str`)：必选，仅支持 `"TND"`、`"BNSD"`，需与 `q_input_layout` 一致。

- **num_key_value_heads** (`int`)：必选，key/value 的 head 数。

- **scale_value** (`float`)：必选，与正向相同的缩放系数，一般取 $D^{-0.5}$。

## 返回值说明

`(Tensor, Tensor, Tensor)`

- **d_query**：与 query 的 shape 和 dtype 一致。
- **d_key**：与 key 的 shape 和 dtype 一致。
- **d_value**：与 value 的 shape 和 dtype 一致。

## 约束说明

- 需保证 CANN 与 PyTorch 版本匹配。
- `block_shape` 若传入，则至少两个元素且大于 0；**blockShapeY 必须为 128 的倍数**。
- 当 `q_input_layout` 为 `"TND"` 时，`actual_seq_lengths` 必选；当 `kv_input_layout` 为 `"TND"` 时，`actual_seq_lengths_kv` 必选。
- query 的 head 数 $N1$ 与 key/value 的 head 数 $N2$ 需满足 $N1 ≥ N2$ 且 $N1 \% N2 = 0$。
- head 维度（headDim）≤ 128。
- 序列长度不需要被 block_shape 整除，分块数按向上取整计算。

## 调用示例

- BNSD 布局下先做正向再做反向

    ```python
    import torch
    import torch_npu
    
    B, N, S, D = 1, 2, 4, 8
    num_kv_heads = 2
    scale_value = 1.0 / (D ** 0.5)
    block_shape = [128, 128]  # blockShapeY 须为 128 的倍数
    ceil_q = (S + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (S + block_shape[1] - 1) // block_shape[1]
    
    query = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    key = torch.randn(B, num_kv_heads, S, D, dtype=torch.float16).npu()
    value = torch.randn(B, num_kv_heads, S, D, dtype=torch.float16).npu()
    block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int32).npu()
    
    attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
        query, key, value, block_sparse_mask, block_shape,
        q_input_layout="BNSD", kv_input_layout="BNSD",
        num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
    )
    d_out = torch.randn_like(attention_out).npu()
    
    d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
        d_out, query, key, value, attention_out, softmax_lse, block_sparse_mask,
        block_shape=block_shape,
        q_input_layout="BNSD", kv_input_layout="BNSD",
        num_key_value_heads=num_kv_heads, scale_value=scale_value,
    )
    print(d_query.shape, d_key.shape, d_value.shape)  # 与 query/key/value 一致
    ```

- TND 布局（需传 actual_seq_lengths / actual_seq_lengths_kv）

    ```python
    T, N, D = 4, 2, 8
    num_kv_heads = 2
    scale_value = 1.0 / (D ** 0.5)
    block_shape = [128, 128]  # blockShapeY 须为 128 的倍数
    ceil_q = (T + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (T + block_shape[1] - 1) // block_shape[1]
    query = torch.randn(T, N, D, dtype=torch.float16).npu()
    key = torch.randn(T, num_kv_heads, D, dtype=torch.float16).npu()
    value = torch.randn(T, num_kv_heads, D, dtype=torch.float16).npu()
    block_sparse_mask = torch.ones(1, N, ceil_q, ceil_kv, dtype=torch.int32).npu()
    
    attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
        query, key, value, block_sparse_mask, block_shape,
        q_input_layout="TND", kv_input_layout="TND",
        num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
        actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
    )
    d_out = torch.randn_like(attention_out).npu()
    d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
        d_out, query, key, value, attention_out, softmax_lse, block_sparse_mask,
        block_shape=block_shape,
        q_input_layout="TND", kv_input_layout="TND",
        num_key_value_heads=num_kv_heads, scale_value=scale_value,
        actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
    )
    ```
