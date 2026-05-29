# torch\_npu.fused\_linear\_cross\_entropy\_loss\_with\_max\_sum\_grad

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- API功能：词汇表并行场景下交叉熵损失计算的梯度算子。用于计算叶子节点`input`和`weight`的梯度。需要获得fused\_linear\_online\_max\_sum和fused\_cross\_entropy\_loss\_with\_max\_sum的相关输出作为本接口输入。支持高性能模式（传入softmax）和省显存模式（传入logits\_max和sum\_exp\_logits）。

- 计算公式：

    高性能模式（softmax非None）：

    $$softmax[arange, maskedTarget] \leftarrow softmax[arange, maskedTarget] - (1 - targetMask)$$

    $$softmax \leftarrow softmax \odot grad.unsqueeze(-1)$$

    $$gradInput = softmax \cdot weight^T$$

    $$gradWeight = softmax^T \cdot input$$

    省显存模式（softmax为None）：

    $$vocabParallelLogits = input \cdot weight^T$$

    $$expLogits = \exp(vocabParallelLogits - logitsMax.unsqueeze(-1))$$

    $$expLogits \leftarrow \frac{expLogits}{sumExpLogits.unsqueeze(-1)}$$

    后续同高性能模式处理。

## 函数原型

```python
torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad(grad, input, weight, target_mask, masked_target, label_smoothing=0.0, logits_max=None, sum_exp_logits=None, softmax=None) -> (Tensor, Tensor)
```

## 参数说明

- **grad**（`Tensor`）：必选参数，当前节点的梯度，1维Tensor。数据类型`float32`，数据格式ND。支持空Tensor。
- **input**（`Tensor`）：必选参数，矩阵乘的输入矩阵，2维Tensor。数据类型支持`float16`、`bfloat16`，数据格式ND。第0维长度需与`grad`一致。支持空Tensor。
- **weight**（`Tensor`）：必选参数，矩阵乘的权重矩阵，2维Tensor。数据类型需与`input`一致，数据格式ND。第0维长度不支持小于128，第1维长度需与`input`的第1维一致。
- **target\_mask**（`Tensor`）：必选参数，目标词ID是否在范围内的位掩码，1维Tensor。数据类型`uint8`，数据格式ND。每1bit代表1个布尔值，shape长度乘以8须不小于`grad`长度。
- **masked\_target**（`Tensor`）：必选参数，目标词ID映射到当前设备的局部索引，1维Tensor。数据类型支持`int32`、`int64`，数据格式ND。shape长度需与`grad`一致。
- **label\_smoothing**（`float`）：可选参数，标签平滑系数，当前仅支持0.0，默认值0.0。
- **logits\_max**（`Tensor`）：可选参数，全局logits最大值，1维Tensor。数据类型`float32`。`softmax`为None时必须提供。默认值None。
- **sum\_exp\_logits**（`Tensor`）：可选参数，处理后的logits，1维Tensor。数据类型`float32`。`softmax`为None时必须提供。默认值None。
- **softmax**（`Tensor`）：可选参数，softmax计算结果，2维Tensor。数据类型`float32`。传入时走高性能模式，不传时走省显存模式（需提供`logits_max`和`sum_exp_logits`）。默认值None。

## 返回值说明

返回2个Tensor：

- **input\_grad**（`Tensor`）：`input`的梯度，数据类型与`input`一致，shape为`[input.size(0), input.size(1)]`。
- **weight\_grad**（`Tensor`）：`weight`的梯度，数据类型与`input`一致，shape为`[weight.size(0), weight.size(1)]`。

## 约束说明

- `input`与`weight`数据类型必须一致。
- `label_smoothing`当前仅支持0。
- `softmax`为None时，`logits_max`和`sum_exp_logits`必须同时提供。
- `softmax`非None时，`logits_max`和`sum_exp_logits`无效。
- `weight`第0维长度不支持小于128。
- 默认确定性实现。

## 调用示例

- 高性能模式（传入softmax）：

```python
import torch
import torch_npu

batch = 128
hidden = 64
vocab_size = 256

grad = torch.randn(batch, dtype=torch.float32).npu()
input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()
softmax = torch.randn(batch, vocab_size, dtype=torch.float32).npu()

input_grad, weight_grad = torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad(
    grad, input_tensor, weight_tensor, target_mask, masked_target,
    label_smoothing=0.0, softmax=softmax
)
```

- 省显存模式（传入logits_max和sum_exp_logits）：

```python
import torch
import torch_npu

batch = 128
hidden = 64
vocab_size = 256

grad = torch.randn(batch, dtype=torch.float32).npu()
input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()
logits_max = torch.randn(batch, dtype=torch.float32).npu()
sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0

input_grad, weight_grad = torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad(
    grad, input_tensor, weight_tensor, target_mask, masked_target,
    label_smoothing=0.0, logits_max=logits_max, sum_exp_logits=sum_exp_logits, softmax=None
)
```
