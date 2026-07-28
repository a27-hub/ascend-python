# torch_npu.npu_dense_lightning_indexer_softmax_lse

> [!NOTICE]
> 此接口为本版本新增功能，具体依赖要求请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.10.0-26.0.0/docs/zh/release_notes/release_notes.md)”。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |

## 功能说明

- API功能：该接口是`npu_dense_lightning_indexer_grad_kl_loss`算子计算Softmax输入的一个分支算子，用于计算Index Attention的Softmax过程中的最大值与累加和。
     - 与`npu_dense_lightning_indexer_grad_kl_loss`配合使用：本接口计算出的`softmax_max_index`与`softmax_sum_index`作为`npu_dense_lightning_indexer_grad_kl_loss`的输入，两个接口配合使用可以优化显存占用。

- 计算公式：

    1. 对`query_index`与`key_index`的矩阵乘结果加权求和并加上AttentionMask：

        $$
        \text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^\top\right)\right)\right)
        $$

    2. 计算`res`的最大值：

        $$
        \text{maxIndex}=\text{max}\left(\text{res}\right)
        $$

    3. 计算`res`减去最大值后取指数的累加和：

        $$
        \text{sumIndex}=\text{ReduceSum}\left(\text{exp}\left(\text{res}-\text{maxIndex}\right)\right)
        $$

    `maxIndex`、`sumIndex`作为输出传递给算子`npu_dense_lightning_indexer_grad_kl_loss`作为输入计算Softmax使用。

## 函数原型

```python
torch_npu.npu_dense_lightning_indexer_softmax_lse(query_index, key_index, weight, *, actual_seq_qlen=None, actual_seq_klen=None, layout='TND', sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807) -> (Tensor, Tensor)
```

## 参数说明

**query_index**(`Tensor`)：必选参数，表示Lightning Indexer结构的输入query，对应公式中的$Q_{index}$。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`。shape支持$(B, S1, Nidx1, D)$、$(T1, Nidx1, D)$。

**key_index**(`Tensor`)：必选参数，表示Lightning Indexer结构的输入key，对应公式中的$K_{index}$。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`。shape支持$(B, S2, Nidx2, D)$、$(T2, Nidx2, D)$。

**weight**(`Tensor`)：必选参数，表示Lightning Indexer的权重系数，对应公式中的$W$。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`、`float32`。shape支持$(B, S1, Nidx1)$、$(T1, Nidx1)$。

**\***：位置参数与关键字参数的分隔符，其之前的参数为位置参数，需按顺序传入；其之后的参数为关键字参数，未赋值时使用默认值。

**actual_seq_qlen**(`list[int]`)：可选参数，TND场景时需传入此参数，表示query每个Batch的有效token数（多个Batch的S1累加）。数据类型支持`int64`，默认值为`None`。

**actual_seq_klen**(`list[int]`)：可选参数，TND场景时需传入此参数，表示key每个Batch的有效token数（多个Batch的S2累加）。数据类型支持`int64`，默认值为`None`。

**layout**(`str`)：可选参数，用于标识输入的数据排布格式。当前支持$BSND$、$TND$，默认值为$TND$。

**sparse_mode**(`int`)：可选参数，表示sparse的模式，默认值为`3`。当前仅支持模式`3`（rightDownCausal模式，对应以右顶点为划分的下三角场景）。

**pre_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和前几个token计算关联，数据类型支持`int64`，默认值为$2^{63}-1$。

**next_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和后几个token计算关联，数据类型支持`int64`，默认值为$2^{63}-1$。

## 返回值说明

- **softmax\_max\_index**(`Tensor`)：表示Index Attention softmax结果中的最大值，对应公式中的$\text{maxIndex}$。数据类型支持`float32`。shape支持$(B, Nidx2, S1)$、$(Nidx2, T1)$，其中$Nidx2$与`key_index`的$Nidx2$保持一致。
- **softmax\_sum\_index**(`Tensor`)：表示Index Attention softmax结果的求和，对应公式中的$\text{sumIndex}$。数据类型支持`float32`。shape支持$(B, Nidx2, S1)$、$(Nidx2, T1)$，其中$Nidx2$与`key_index`的$Nidx2$保持一致。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式。
- 参数`query_index`、`key_index`的数据类型应保持一致。
- 参数`weight`不为`float32`时，参数`query_index`、`key_index`、`weight`的数据类型应保持一致。
- shape数值约束：

| 规格项    | 规格          | 规格说明                                                                                                                              |
|-----------|---------------|---------------------------------------------------------------------------------------------------------------------------------------|
| B         | 1~256         | -                                                                                                                                     |
| S1、S2    | 0~128K        | S1、S2支持不等长。`layout`为$BSND$时S1<=S2；`layout`为$TND$时actualSeqLengthsQuery小于等于actualSeqLengthsKey相同索引位置的值，且相同索引位置S1<=S2。 |
| Nidx1     | 8、16、32、64 | -                                                                                                                                     |
| Nidx2     | 1             | -                                                                                                                                     |
| D         | 128           | -                                                                                                                                     |
| layout    | BSND/TND      | -                                                                                                                                     |

- `sparse_mode`不同模式的含义如下，当前仅支持模式`3`：

| sparseMode | 含义                                                                                                                  | 是否支持 |
|------------|-----------------------------------------------------------------------------------------------------------------------|:--------:|
| 0          | defaultMask模式，如果attenmask未传入则不做mask操作，忽略preTokens和nextTokens；如果传入，则需要传入完整的attenmask矩阵   |    ×     |
| 1          | allMask，必须传入完整的attenmask矩阵                                                                                  |    ×     |
| 2          | leftUpCausal模式的mask，需要传入优化后的attenmask矩阵                                                                  |    ×     |
| 3          | rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵                              |    √     |
| 4          | band模式的mask，需要传入优化后的attenmask矩阵                                                                         |    ×     |
| 5          | prefix                                                                                                                |    ×     |
| 6          | global                                                                                                                |    ×     |
| 7          | dilated                                                                                                               |    ×     |
| 8          | block-local                                                                                                           |    ×     |

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> # TND layout: query_index=(T1, Nidx1, D), key_index=(T2, Nidx2, D)
    >>> T1, T2, Nidx1, Nidx2, D = 128, 256, 8, 1, 128
    >>> q_index = torch.randn(T1, Nidx1, D, dtype=torch.float16, device='npu')
    >>> k_index = torch.randn(T2, Nidx2, D, dtype=torch.float16, device='npu')
    >>> weight = torch.randn(T1, Nidx1, dtype=torch.float16, device='npu')
    >>> softmax_max_index, softmax_sum_index = torch_npu.npu_dense_lightning_indexer_softmax_lse(
    ...     q_index, k_index, weight,
    ...     actual_seq_qlen=[T1], actual_seq_klen=[T2],
    ...     layout='TND', sparse_mode=3)
    >>> softmax_max_index.shape
    torch.Size([1, 128])
    >>> softmax_max_index.dtype
    torch.float32
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    # BSND layout: query_index=(B, S1, Nidx1, D), key_index=(B, S2, Nidx2, D)
    B, S1, S2, Nidx1, Nidx2, D = 1, 128, 256, 8, 1, 128
    q_index = torch.randn(B, S1, Nidx1, D, dtype=torch.float16, device='npu')
    k_index = torch.randn(B, S2, Nidx2, D, dtype=torch.float16, device='npu')
    weight = torch.randn(B, S1, Nidx1, dtype=torch.float16, device='npu')


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q_index, k_index, weight):
            return torch_npu.npu_dense_lightning_indexer_softmax_lse(
                q_index, k_index, weight, layout='BSND', sparse_mode=3)


    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    softmax_max_index, softmax_sum_index = model(q_index, k_index, weight)
    print(softmax_max_index.shape, softmax_sum_index.shape)
    ```
