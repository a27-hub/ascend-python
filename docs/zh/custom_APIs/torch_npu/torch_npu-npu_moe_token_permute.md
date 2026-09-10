# torch_npu.npu_moe_token_permute

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term> | √ |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |

## 功能说明

- API功能：MoE计算中，根据索引`indices`将输入`tokens`扩展，并按照专家索引排序，返回排序后的token及其与原始token的映射关系。通常与[torch_npu.npu_moe_token_unpermute](torch_npu-npu_moe_token_unpermute.md)配套使用。

- 计算公式：

    记`indices`的元素个数为$F$（`indices`为2维时$F = NUM\_TOKENS * K$）：

    $$
    sorted\_indices = stableArgsort(flatten(indices)) \\
    permuted\_tokens[i] = tokens[sorted\_indices[i] // K]
    $$

    `num_out_tokens`不为None或0时，对排序后的结果按行数截断或从尾部删除。

- 等价计算逻辑

    ```python
    import numpy as np

    def moe_token_permute_np(tokens, indices, num_out_tokens=None):
        num_tokens = tokens.shape[0]
        flat_indices = indices.reshape(-1)
        k = flat_indices.shape[0] // num_tokens
        sorted_indices = np.argsort(flat_indices, kind="stable")
        f = flat_indices.shape[0]
        if num_out_tokens is None or num_out_tokens == 0:
            m = f
        elif num_out_tokens > 0:
            m = min(num_out_tokens, f)
        else:
            m = f + num_out_tokens
        permuted_tokens = tokens[sorted_indices[:m] // k]
        return permuted_tokens, sorted_indices.astype(np.int32)
    ```

## 函数原型

```python
torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=None, padded_mode=False) -> (Tensor, Tensor)
```

## 参数说明

>
> [!NOTE]
> shape中的符号说明：
>
> - $NUM\_TOKENS$：输入token个数。
> - $K$：每个token对应的专家个数，`indices`为1维时$K=1$。
> - $H$：每个token的特征长度，为列数。
> - $F$：`indices`的元素个数，$F = NUM\_TOKENS * K$。

- **tokens** (`Tensor`)：必选参数，MoE的输入token特征，要求为一个2维张量，shape为(NUM_TOKENS, H)。数据格式要求为$ND$。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`、`torch.int8`（`torch.int8`按非量化方式处理）。
- **indices** (`Tensor`)：必选参数，token对应的专家索引，要求为1维或2维张量，shape为(NUM_TOKENS, )或(NUM_TOKENS, K)。数据类型支持`torch.int32`、`torch.int64`，数据格式要求为$ND$。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：元素个数需要小于16777215，元素值需要大于等于0且小于16777215。
    - <term>Ascend 950PR/Ascend 950DT</term>：元素个数需要小于16777215，元素值需要大于等于0且小于16777215；`tokens`数据类型为`torch.int8`时，元素表示专家id，取值范围为[0, 10240)。
- **num_out_tokens** (`int`)：可选参数，默认值为None，控制输出`permuted_tokens`的有效行数$M$。None或0表示保留全部行，即$M=F$；正数表示按排序后顺序最多保留`num_out_tokens`行，即$M=min(num\_out\_tokens, F)$；负数表示从完整结果尾部删除对应行数，即$M=max(F+num\_out_tokens, 0)$。
- **padded_mode** (`bool`)：可选参数，默认值为False，表示是否使用padded模式。如果为True，表示`indices`已被填充为每个专家选中的`token`索引，此时不对`indices`进行排序。当前仅支持`False`。

## 返回值说明

- **permuted_tokens** (`Tensor`)：根据`indices`扩展并按专家索引排序后的token特征，要求是一个2维张量，shape为(M, H)，数据类型与`tokens`一致，数据格式要求为$ND$。
- **sorted_indices** (`Tensor`)：`permuted_tokens`与原始`tokens`的行映射关系，要求是一个1维张量，shape为(F, )，数据类型支持`torch.int32`，数据格式要求为$ND$。

## 约束说明

- 该接口支持训练和推理场景下使用，支持对`tokens`的自动微分。
- 该接口支持PTA图模式。
- 该接口默认支持确定性计算。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>上，$K$不大于512。
- <term>Ascend 950PR/Ascend 950DT</term>上，该接口由框架内部转调其他算子实现，报错信息中的参数名可能与本文档不一致。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    dtype = torch.bfloat16
    tokens = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0]]).npu().to(dtype)
    indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
    num_out_tokens = indices.numel()
    permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens)
    ```
