# torch_npu.fused_linear_online_max_sum

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- API功能：并行场景下交叉熵计算的一部分，解决超大规模词汇表下的显存和计算效率问题，功能等价Megatron的matmul与fused\_vocab\_parallel\_cross\_entropy的实现，支持vocabulary\_size维度切卡融合matmul与cross\_entropy\_loss，中间根据通信拆分为torch_npu.fused_linear_online_max_sum、torch_npu.fused_cross_entropy_loss_with_max_sum，需要依次调用实现完整功能。torch_npu.fused_linear_online_max_sum为第一阶段接口，支持高性能和省显存两种模式，高性能模式支持返回matmul计算结果，省显存模式不返回matmul计算结果。

- 计算公式：
    $$
    vocab\_parallel\_logits = input @ weight^T
    $$

    $$
    logits\_max\_local = max(vocab\_parallel\_logits, dim=-1)
    $$

    $$
    subRes = vocab\_parallel\_logits - logits\_max\_local
    $$

    $$
    sum\_exp\_logits\_local = sum(exp(subRes), dim=-1)
    $$

    $$
    target\_mask = (target < vocab\_start\_index) | (target > vocab\_end\_index)
    $$

    $$
    masked\_target[b] =
    \begin{cases}
    0 & \text{target\_mask[b]=true}\\
    target[b] - vocab\_start\_index & \text{target\_mask[b]=false}
    \end{cases}
    $$

    $$
    predicted\_logits\_local[b] =
    \begin{cases}
    0 & \text{target\_mask[b]=true}\\
    subRes[b][masked\_target[b]] & \text{target\_mask[b]=false}
    \end{cases}
    $$

    $$
    align\_num = (input.size(0) + 7) / 8 * 8\\
    maskBit[p] = \begin{cases}
    uint8(target_mask[p]) & \text{p < input.size(0)}\\
    1 & \text{input.size(0) <= p < align\_num}
    \end{cases} \\
    target_mask[k] = 0b(maskBit[8*k:8*k+8])
    $$

其中$0 \le b \lt input.size(0), 0 \le p \lt align\_num, 0 \le k \lt align\_num / 8$。

## 函数原型

```
torch_npu.fused_linear_online_max_sum(input, weight, target, vocab_start_index, vocab_end_index, return_logits=False) -> logits_max_local, sum_exp_logits_local, predicted_logits_local, target_mask, masked_target, vocab_parallel_logits
```

## 参数说明

  - **input**(`Tensor`)：表示matmul计算的左矩阵，公式中的`input`，Device侧的Tensor。数据类型支持BFLOAT16、FLOAT16，shape支持2维且`input.size(1)`需要小于等于65534，支持非连续Tensor，支持空tensor，支持ND。
  - **weight**(`Tensor`)：表示matmul计算的右矩阵，公式中的`weight`，Device侧的Tensor。数据类型支持BFLOAT16、FLOAT16，数据类型需要与`logits`一致，shape支持2维且`weight.size(1)`需要与`input.size(1)`一致，支持非连续Tensor，支持空tensor，支持ND。
  - **target**(`Tensor`)：表示目标索引，公式中的`target`，Device侧的Tensor。数据类型支持INT32、INT64，shape支持1维且`target.size(0)`需要与input.size(0)一致，支持非连续Tensor，支持空tensor，支持ND。
  - **vocab_start_index**(int64_t)：表示分到本卡上的开始索引，公式中的`vocab_start_index`，Host侧的int64常量。取值范围为[0, max(target) - 1]
  - **vocab_end_index**(int64_t)：表示分到本卡上的结束索引，公式中的vocab_end_index，Host侧的int64常量。取值范围为`[vocab_start_index, min(vocab_start_index + weight.size(0) - 1, max(target) - 1)]`
  - **return_logits**(bool)：表示是否返回matmul计算结果，即公式中的`vocab_parallel_logits`。


## 返回值说明

  - **logits_max_local**(`Tensor`)：表示matmul计算后各行的最大值，公式中的`logits_max_local`，Device侧的Tensor。数据类型支持FLOAT32，shape支持1维且`logits_max_local.size(0)`需要与`input.size(0)`一致，格式为ND。
  - **sum_exp_logits_local**(`Tensor`)：表示matmul计算结果与其各行的最大值作差后经过`exp`计算后的结果，公式中的`sum_exp_logits_local`，Device侧的Tensor。数据类型支持FLOAT32，shape支持1维且`sum_exp_logits_local.size(0)`需要与`input.size(0)`一致，格式为ND。
  - **predicted_logits_local**(`Tensor`)：表示matmul计算结果与其各行的最大值作差后`masked_target`筛选后的结果，公式中的`predicted_logits_local`，Device侧的Tensor。数据类型支持FLOAT32，shape支持1维且`predicted_logits_local.size(0)`需要与input.size(0)一致，格式为ND。
  - **target_mask**(`Tensor`)：表示用于筛选词表的mask，公式中的`target_mask`，Device侧的Tensor。数据类型支持UINT8，shape支持1维，shape为`[(input.size(0) + 7) / 8]`，格式为ND。
  - **masked_target**(`Tensor`)：表示target经过target_mask过滤后的结果，公式中的masked_target，Device侧的Tensor。数据类型支持INT32、INT64，且数据类型需要与target一致，shape支持1维且`masked_target.size(0)`需要与`input.size(0)`一致，格式为ND。
  - **vocab_parallel_logits**(`Tensor`)：表示matmul计算结果,公式中的`vocab_parallel_logits`。可选输出，当`return_logits`为True时返回，，Device侧的Tensor。数据类型支持BFLOAT16、FLOAT16，且数据类型需要`input`一致，shape支持2维，shape为`[input.size(0), weight.size(0)]`，格式为ND。


## 约束说明



## 调用示例
