# （beta）torch_npu.npu_layer_norm_eval

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.nn.functional.layer_norm`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

对层归一化（Layer Normalization）结果进行计算，语义与 `torch.nn.functional.layer_norm` 一致，由 NPU 侧实现加速。

## 计算公式与说明

### 归一化范围

设 `normalized_shape` 长度为 \(r\)，且须满足 `input.shape[-r:]` 与 `normalized_shape` **逐维相等**（与 `torch.nn.functional.layer_norm` 一致；仅各维长度乘积相同但形状不匹配时不会按该接口合法计算）。此时 `input` 的总维数为 \(k + r\)，其中 \(k\) 为 `input.dim()` 减去 \(r\)。前 \(k\) 个维度为“切片维”：对每一组固定的前 \(k\) 维下标，在末尾 \(r\) 维上共有

\[
N = \prod_{j=1}^{r} \text{normalized\_shape}[j]
\]

个标量参与一次归一化。将末尾 \(r\) 维按行主序拉平为长度 \(N\) 的向量 \(\mathbf{x} = (x_1, \ldots, x_N)\)，以下公式均在该向量上计算，且 \(\hat{x}_i\) 与 `input` 中同一位置的元素一一对应；不同前 \(k\) 维下标对应彼此独立的 \(\mathbf{x}\)。

### 均值与方差

在该切片内，**均值**为算术平均：

\[
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
\]

**方差**为减均值后的平方的算术平均（与 PyTorch `layer_norm` 一致，为总体方差，分母为 \(N\)，**不**使用无偏估计的分母 \(N-1\)）：

\[
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
\]

参数 **eps**（记为 \(\epsilon\)）加在方差上，避免除零并改善数值稳定性。

### 标准化与仿射

先对每个元素做标准化，再按 **weight**（\(\gamma\)，可选）、**bias**（\(\beta\)，可选）做逐元素仿射（\(\gamma_i\)、\(\beta_i\) 与 `normalized_shape` 同形，与 \(x_i\) 按广播对齐）：

\[
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
y_i = \gamma_i\,\hat{x}_i + \beta_i
\]

当 **weight** 或 **bias** 为 `None` 时，行为上等价于对应位置取 \(\gamma_i = 1\)、\(\beta_i = 0\)（与 `torch.nn.functional.layer_norm` 的默认一致）。

## 函数原型

```python
torch_npu.npu_layer_norm_eval(input, normalized_shape, weight=None, bias=None, eps=1e-05) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：输入张量。
- **normalized_shape**（`List[int]`）：size为预期输入的shape。
- **weight**（`Tensor`）：可选参数，gamma张量，默认值为None。
- **bias**（`Tensor`）：可选参数，beta张量，默认值为None。
- **eps**（`float`）：为保证数值稳定性添加到分母中的ε值，默认值为1e-5。

## 调用示例

```python
>>> import torch, torch_npu
>>> input = torch.tensor(
...     [[0.1863, 0.3755, 0.1115, 0.7308],
...      [0.6004, 0.6832, 0.8951, 0.2087],
...      [0.8548, 0.0176, 0.8498, 0.3703],
...      [0.5609, 0.0114, 0.5021, 0.1242],
...      [0.3966, 0.3022, 0.2323, 0.3914],
...      [0.1554, 0.0149, 0.1718, 0.4972]],
...     dtype=torch.float32,
... ).npu()
>>> input
tensor([[0.1863, 0.3755, 0.1115, 0.7308],
        [0.6004, 0.6832, 0.8951, 0.2087],
        [0.8548, 0.0176, 0.8498, 0.3703],
        [0.5609, 0.0114, 0.5021, 0.1242],
        [0.3966, 0.3022, 0.2323, 0.3914],
        [0.1554, 0.0149, 0.1718, 0.4972]], device='npu:0')
>>> normalized_shape = input.size()[1:]
>>> normalized_shape
torch.Size([4])
>>> weight = torch.ones(normalized_shape, dtype=input.dtype, device=input.device)
>>> bias = torch.zeros(normalized_shape, dtype=input.dtype, device=input.device)
>>> weight
tensor([1., 1., 1., 1.], device='npu:0')
>>> bias
tensor([0., 0., 0., 0.], device='npu:0')
>>> output = torch_npu.npu_layer_norm_eval(input, normalized_shape, weight, bias, 1e-5)
>>> output
tensor([[-0.6879,  0.1022, -1.0002,  1.5859],
        [ 0.0143,  0.3474,  1.1999, -1.5616],
        [ 0.9422, -1.4361,  0.9280, -0.4341],
        [ 1.1061, -1.2204,  0.8571, -0.7428],
        [ 0.9685, -0.4173, -1.4434,  0.8922],
        [-0.3078, -1.1025, -0.2151,  1.6255]], device='npu:0')
```
