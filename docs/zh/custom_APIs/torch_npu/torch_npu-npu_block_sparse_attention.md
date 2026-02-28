# torch_npu.npu_block_sparse_attention

## 产品支持情况

| 产品                                                                | 是否支持 |
|:------------------------------------------------------------------|:----:|
| 昇腾910_95 AI处理器                                                    |  √   |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                      |  √   |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |  √   |
| Atlas 200I/500 A2 推理产品                                            |  ×   |
| Atlas 推理系列产品                                                      |  ×   |
| Atlas 训练系列产品                                                      |  ×   |
| Atlas 200I/300/500 推理产品                                           |  ×   |

## 功能说明

- **API 功能**：BlockSparseAttention 稀疏注意力正向计算，支持块级稀疏模式，通过 `block_sparse_mask` 指定每个 Q 块选择的 KV 块，实现高效稀疏注意力。

- **计算公式**：稀疏块大小为 blockShapeX × blockShapeY，BlockSparseMask 指定稀疏模式：

$$
attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
$$

- 输入 query、key、value 的排布由 `q_input_layout`、`kv_input_layout` 指定。维度含义：B 为 batch，T 为 total tokens，S 为序列长度，N 为 head 数，D 为 head 维度（Head-Dim）。当前支持的布局：`q_input_layout` / `kv_input_layout` 仅支持 `"TND"`、`"BNSD"`。

## 函数原型

```
torch_npu.npu_block_sparse_attention(query, key, value, block_sparse_mask, q_input_layout, kv_input_layout, num_key_value_heads, scale_value, inner_precise, *, block_shape=None, actual_seq_lengths=None, actual_seq_lengths_kv=None, softmax_lse_flag=0) -> (Tensor, Tensor)
```

## 参数说明

- **query** (`Tensor`)：必选，公式中的 query。数据格式为 ND。
  - TND：shape 为 `[totalQTokens, headNum, headDim]`
  - BNSD：shape 为 `[batch, headNum, maxQSeqLength, headDim]`
  数据类型支持 `float16`、`bfloat16`。

- **key** (`Tensor`)：必选，公式中的 key。数据格式为 ND。
  - TND：`[totalKTokens, numKeyValueHeads, headDim]`
  - BNSD：`[batch, numKeyValueHeads, maxKvSeqLength, headDim]`
  数据类型与 query 一致。

- **value** (`Tensor`)：必选，公式中的 value。shape 与 key 一致，数据类型与 query 一致。

- **block_sparse_mask** (`Tensor`)：必选，块稀疏掩码。shape 为 `[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]`，表示按块划分后哪些块参与计算（1 参与，0 不参与）。数据格式为 ND。

- **q_input_layout** (`str`)：必选，query 的排布。仅支持 `"TND"`、`"BNSD"`。

- **kv_input_layout** (`str`)：必选，key、value 的排布。仅支持 `"TND"`、`"BNSD"`，需与 `q_input_layout` 一致。

- **num_key_value_heads** (`int`)：必选，key/value 的 head 数。

- **scale_value** (`float`)：必选，缩放系数，一般取 $D^{-0.5}$。

- **inner_precise** (`int`)：必选，Softmax 计算精度。`0` 表示 fp32 中间结果（高精度），`1` 表示 fp16 中间结果（性能更优）。

- <strong>*</strong>：其后的参数为关键字参数。

- **block_shape** (`list[int]`, 可选)：稀疏块形状 `[blockShapeX, blockShapeY]`。若传入则至少两个元素且均大于 0。不传时算子默认 `[128, 128]`。

- **actual_seq_lengths** (`list[int]`, 可选)：各 batch 的 query 实际序列长度。当 `q_input_layout` 为 `"TND"` 时必选；为 `"BNSD"` 时可选，不传则按 query 的 S 维处理。

- **actual_seq_lengths_kv** (`list[int]`, 可选)：各 batch 的 key/value 实际序列长度。当 `kv_input_layout` 为 `"TND"` 时必选；为 `"BNSD"` 时可选。

- **softmax_lse_flag** (`int`, 可选)：默认 `0`。`0` 表示不输出 softmax_lse；`1` 表示输出 softmax_lse（可能有性能损失）。

## 返回值说明

`(Tensor, Tensor)`

- 第一个 Tensor：`attention_out`，与 query 的 dtype 和 layout 一致，最后一维与 value 的 headDim 一致。
- 第二个 Tensor：`softmax_lse`，当 `softmax_lse_flag=1` 时有效；TND 为 `[totalQTokens, headNum, 1]`，BNSD 为 `[batch, headNum, maxQSeqLength, 1]`。

## 约束说明

- 需保证 CANN 与 PyTorch 版本匹配。
- `q_input_layout`、`kv_input_layout` 仅支持 `"TND"`、`"BNSD"`。
- query、key、value 数据类型必须一致，且为 `float16` 或 `bfloat16`。
- query 的 head 数 N1 与 key/value 的 head 数 N2 需满足 N1 ≥ N2 且 N1 % N2 == 0。
- `block_sparse_mask` 必传，shape 需为 `[batch, headNum, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKVS, blockShapeY)]`；若传 `block_shape`，则至少两元素且大于 0。
- `inner_precise` 仅支持 0 或 1。
- 当 `q_input_layout` 为 `"TND"` 时，`actual_seq_lengths` 必选；当 `kv_input_layout` 为 `"TND"` 时，`actual_seq_lengths_kv` 必选。
- 序列长度不需要被 block_shape 整除，分块数按向上取整计算。

## 调用示例

- BNSD 布局

    ```python
    import torch
    import torch_npu
    
    B, N, S, D = 1, 2, 4, 8
    num_kv_heads = 2
    scale_value = 1.0 / (D ** 0.5)
    block_shape = [1, 1]
    ceil_q = (S + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (S + block_shape[1] - 1) // block_shape[1]
    
    query = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    key = torch.randn(B, num_kv_heads, S, D, dtype=torch.float16).npu()
    value = torch.randn(B, num_kv_heads, S, D, dtype=torch.float16).npu()
    block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int32).npu()
    
    attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
        query, key, value, block_sparse_mask,
        "BNSD", "BNSD", num_key_value_heads=num_kv_heads,
        scale_value=scale_value, inner_precise=1,
    )
    print(attention_out.shape)  # (B, N, S, D)
    ```

- TND 布局（需传 actual_seq_lengths / actual_seq_lengths_kv）

    ```python
    import torch
    import torch_npu
    
    T, N, D = 4, 2, 8
    num_kv_heads = 2
    scale_value = 1.0 / (D ** 0.5)
    block_shape = [1, 1]
    ceil_q = (T + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (T + block_shape[1] - 1) // block_shape[1]
    
    query = torch.randn(T, N, D, dtype=torch.float16).npu()
    key = torch.randn(T, num_kv_heads, D, dtype=torch.float16).npu()
    value = torch.randn(T, num_kv_heads, D, dtype=torch.float16).npu()
    block_sparse_mask = torch.ones(1, N, ceil_q, ceil_kv, dtype=torch.int32).npu()
    
    attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
        query, key, value, block_sparse_mask,
        "TND", "TND", num_key_value_heads=num_kv_heads,
        scale_value=scale_value, inner_precise=1,
        actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
    )
    print(attention_out.shape)  # (T, N, D)
    ```
