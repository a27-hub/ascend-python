
# torch_npu.fused_cross_entropy_loss_with_max_sum

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- API功能：并行场景下交叉熵计算的一部分，解决超大规模词汇表下的显存和计算效率问题，功能等价Megatron的matmul与fused\_vocab\_parallel\_cross\_entropy的实现，支持vocabulary\_size维度切卡融合matmul与cross\_entropy\_loss，中间根据通信拆分为torch_npu.fused_linear_online_max_sum、torch_npu.fused_cross_entropy_loss_with_max_sum，需要依次调用实现完整功能。torch_npu.fused_cross_entropy_loss_with_max_sum为第一阶段接口，用于计算loss与softmax。支持高性能和省显存两种模式，高性能模式下需要传入vocab\_parallel\_logits，返回softmax计算结果。
- 计算公式：
          $$ 
          loss = log(sum_exp_logits) - predicted_logits
          $$
          $$
          softmax = exp(vocab_parallel_logits -logits_max.unsqueeze(dim = -1)) \ sum_exp_logits.unsqueeze(dim = -1)
          $$
## 函数原型

```
torch_npu.fused_cross_entropy_loss_with_max_sum(logits_max, sum_exp_logits, predicted_logits, label_smoothing=0.0, input=None, weight=None, vocab_parallel_logits=None) -> (loss, softmax)
```

## 参数说明

  - **logits_max**(`Tensor`)：表示matmul计算后各行的最大值，公式中的`logits_max`，Device侧的Tensor。数据类型支持FLOAT32，shape支持1维且`logits_max.size(0)`需要与`input.size(0)`一致，支持非连续Tensor，支持ND。

  - **sum_exp_logits**(`Tensor`)：表示matmul计算结果与其各行的最大值作差后经过`exp`计算后的结果，公式中的`sum_exp_logits`，Device侧的Tensor。数据类型支持FLOAT32，shape支持1维且`sum_exp_logits.size(0)`需要与`input.size(0)`一致，支持非连续Tensor，支持ND。
  
  - **predicted_logits**(`Tensor`)：表示matmul计算结果与其各行的最大值作差后`masked_target`筛选后的结果，公式中的`predicted_logits`，Device侧的Tensor。数据类型支持FLOAT32，shape支持1维且`predicted_logits.size(0)`需要与`input.size(0)`一致，支持非连续Tensor，支持ND。

  - **label_smoothing**(float)：关键字可选入参，表示标签平滑系数，用于缓解过拟合，当前只支持0。

  - **input**(`Tensor`)：关键字可选入参，表示matmul计算的左矩阵，公式中的`input`，Device侧的Tensor。当前只支持传入None。
  
  - **weight**(`Tensor`)：关键字可选入参，表示matmul计算的右矩阵，公式中的`weight`，Device侧的Tensor。当前只支持传入None。

  - **vocab_parallel_logits**(`Tensor`)：关键字可选入参，表示一阶段matmul计算结果，公式中的`vocab_parallel_logits`，Device侧的Tensor。数据类型支持BFLOAT16、FLOAT16，shape支持2维，shape为`[input.size(0), weight.size(0)]`，支持非连续Tensor，支持ND。



## 返回值说明

  - **loss**(`Tensor`)：表示计算得到的损失值，公式中的`loss`，Device侧的Tensor。数据类型支持FLOAT32，shape支持1维且`loss_out.size(0)`需要与`input.size(0)`一致，格式为ND。

  - **softmax**(`Tensor`)：表示softmax计算结果，公式中的`softmax`，Device侧的Tensor。数据类型支持FLOAT32，shape支持2维，shape为`[input.size(0), weight.size(0)]`，格式为ND。


## 约束说明
