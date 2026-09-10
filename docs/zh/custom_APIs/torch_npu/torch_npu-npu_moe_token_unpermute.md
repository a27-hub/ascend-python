# torch_npu.npu_moe_token_unpermute

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term> | √ |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |

## 功能说明

- API功能：MoE计算中，[torch_npu.npu_moe_token_permute](torch_npu-npu_moe_token_permute.md)的逆操作，根据`sorted_indices`将扩展排序后的token特征映射回原始token位置并累加；传入`probs`时先按`probs`加权再累加。

- 计算公式：

    记`permuted_tokens`的行数为$F = NUM\_TOKENS * K$（`probs`为None时$K=1$，否则$K$为`probs`的第二维）：

    $$
    out[sorted\_indices[i] // K] \mathrel{+}= permuted\_tokens[i] * probs[sorted\_indices[i] // K, sorted\_indices[i] \% K]
    $$

    `probs`为None时权重视为1。

- 等价计算逻辑

    ```python
    import numpy as np

    def moe_token_unpermute_np(permuted_tokens, sorted_indices, probs=None):
        h = permuted_tokens.shape[1]
        k = 1 if probs is None else probs.shape[1]
        num_tokens = sorted_indices.shape[0] // k
        out = np.zeros((num_tokens, h), dtype=permuted_tokens.dtype)
        for i, s in enumerate(sorted_indices):
            row = s // k
            w = 1.0 if probs is None else probs[row, s % k]
            out[row] += permuted_tokens[i] * w
        return out
    ```

## 函数原型

```python
torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None) -> Tensor
```

## 参数说明

>
> [!NOTE]
> shape中的符号说明：
>
> - $NUM\_TOKENS$：原始token个数。
> - $K$：每个token对应的专家个数，`probs`为None时$K=1$。
> - $H$：每个token的特征长度，为列数。
> - $F$：扩展后的行数，$F = NUM\_TOKENS * K$。

- **permuted_tokens** (`Tensor`)：必选参数，经过扩展排序后的token特征（通常为[torch_npu.npu_moe_token_permute](torch_npu-npu_moe_token_permute.md)的第一个输出经过专家计算后的结果），要求为一个2维张量，shape为(F, H)。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式要求为$ND$。
- **sorted_indices** (`Tensor`)：必选参数，扩展排序后的行映射关系（通常为[torch_npu.npu_moe_token_permute](torch_npu-npu_moe_token_permute.md)的第二个输出），要求为一个1维张量，shape为(F, )。数据类型支持`torch.int32`，数据格式要求为$ND$。元素取值范围为[0, F-1]，且不存在重复索引。
- **probs** (`Tensor`)：可选参数，默认值为None，表示各专家处理结果的权重，要求为一个2维张量，shape为(NUM_TOKENS, K)。数据类型要求与`permuted_tokens`一致，数据格式要求为$ND$。
- **padded_mode** (`bool`)：可选参数，默认值为False，表示是否使用padded模式，当前仅支持`False`。
- **restore_shape** (`List[int]`)：可选参数，默认值为None，`padded_mode`为True时用于指定输出shape，当前仅支持`None`。

## 返回值说明

`Tensor`

输出参数`out`，代表映射回原始token位置并累加后的结果。要求是一个2维张量，shape为(NUM_TOKENS, H)，数据类型与`permuted_tokens`一致，数据格式要求为$ND$。

## 约束说明

- 该接口支持训练和推理场景下使用，支持对`permuted_tokens`和`probs`的自动微分。
- 该接口支持PTA图模式。
- 该接口默认支持确定性计算。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>上，$K$不大于512。
## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    dtype = torch.bfloat16
    permuted_tokens = torch.tensor([[1., 1., 1.],
                                        [0., 0., 0.],
                                        [0., 0., 0.],
                                        [3., 3., 3.],
                                        [2., 2., 2.],
                                        [1., 1., 1.],
                                        [2., 2., 2.],
                                        [3., 3., 3.]]).npu().to(dtype)
    sorted_indices = torch.tensor([0, 6, 7, 5, 3, 1, 2, 4], dtype=torch.int32).npu()
    indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
    probs = (torch.ones_like(indices) / 2).to(dtype)
    unpermuted_tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)
    ```