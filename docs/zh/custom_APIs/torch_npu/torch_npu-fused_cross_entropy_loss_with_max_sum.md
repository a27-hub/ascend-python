# torch\_npu.fused\_cross\_entropy\_loss\_with\_max\_sum

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- API功能：词汇表并行场景下交叉熵计算模块的一部分，解决超大规模词汇表下的显存和计算效率问题。本算子为计算loss与softmax结果的部分，需配合fused\_linear\_online\_max\_sum使用。在多卡场景下，需先对`logits_max`和`sum_exp_logits`执行全局通信（all-reduce），再调用本接口。

- 计算公式：

    $$lossOut = \log(sum\_exp\_logits) - predicted\_logits$$

    $$softMaxOut = \frac{\exp(vocab\_parallel\_logits - logits\_max.unsqueeze(-1))}{sum\_exp\_logits.unsqueeze(-1)}$$

## 函数原型

```python
torch_npu.fused_cross_entropy_loss_with_max_sum(logits_max, sum_exp_logits, predicted_logits, *, label_smoothing=0.0, input=None, weight=None, vocab_parallel_logits=None) -> (Tensor, Tensor)
```

## 参数说明

- **logits\_max**（`Tensor`）：必选参数，全局通信后的matmul结果各行最大值，1维Tensor。数据类型`float32`，数据格式ND。
- **sum\_exp\_logits**（`Tensor`）：必选参数，全局通信后的exp累加结果，1维Tensor。数据类型`float32`，数据格式ND。shape与`logits_max`一致。
- **predicted\_logits**（`Tensor`）：必选参数，全局通信后的预测logits，1维Tensor。数据类型`float32`，数据格式ND。shape与`logits_max`一致。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **label\_smoothing**（`float`）：可选参数，标签平滑系数，当前仅支持0.0，默认值0.0。
- **input**（`Tensor`）：可选参数，matmul输入左矩阵，当前仅支持None，默认值None。
- **weight**（`Tensor`）：可选参数，matmul输入右矩阵，当前仅支持None，默认值None。
- **vocab\_parallel\_logits**（`Tensor`）：可选参数，matmul计算结果。传入时计算softmax输出，不传入时softmax输出为空Tensor。数据类型支持`float32`、`float16`、`bfloat16`，2维Tensor，shape第0维需与`logits_max`一致。默认值None。

## 返回值说明

返回2个Tensor：

- **loss**（`Tensor`）：交叉熵loss，数据类型`float32`，shape与`logits_max`一致。
- **softmax**（`Tensor`）：softmax结果。`vocab_parallel_logits`非None时，数据类型`float32`，shape与`vocab_parallel_logits`一致；`vocab_parallel_logits`为None时，返回空Tensor。

## 约束说明

- `logits_max`、`sum_exp_logits`、`predicted_logits`的shape需一致。
- `label_smoothing`当前仅支持0。
- 默认确定性实现。

## 调用示例

```python
import torch
import torch_npu

# 假设已通过fused_linear_online_max_sum获取输出并完成全局通信
batch = 128
vocab_size = 256

logits_max = torch.randn(batch, dtype=torch.float32).npu()
sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0
predicted_logits = torch.randn(batch, dtype=torch.float32).npu()

# 不返回softmax
loss, _ = torch_npu.fused_cross_entropy_loss_with_max_sum(
    logits_max, sum_exp_logits, predicted_logits,
    label_smoothing=0.0
)

# 返回softmax（传入vocab_parallel_logits）
vocab_parallel_logits = torch.randn(batch, vocab_size, dtype=torch.float16).npu()
loss, softmax = torch_npu.fused_cross_entropy_loss_with_max_sum(
    logits_max, sum_exp_logits, predicted_logits,
    label_smoothing=0.0, vocab_parallel_logits=vocab_parallel_logits
)
```
