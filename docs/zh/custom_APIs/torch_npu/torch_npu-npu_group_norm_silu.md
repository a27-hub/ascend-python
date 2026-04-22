# torch_npu.npu_group_norm_silu

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    | √  |
|<term>Atlas 推理系列产品</term>    | √  |

## 功能说明

- API功能：计算输入张量`input`按组归一化的结果，包括张量out、均值meanOut、标准差的倒数rstdOut以及silu的输出。
- 计算公式：
    - GroupNorm：$x$为输入`input`，$\gamma$和$\beta$分别代表输入`weight`和`bias`，$E[x] = \bar{x}$代表$x$的均值，$ Var[x]=\frac{1}{n}\sum_{i=1}^{n} (x_i - E[x])^2 $ 代表$x$的方差，则
    $$
    \begin{cases}
    \text{groupnormOut} = \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    \text{meanOut}  = E[x] \\
    \text{rstdOut}  = \frac{1}{\sqrt{Var[x] + eps}}
    \end{cases}
    $$
    - Silu：
    $$
    \text{out} = \frac{\text{groupnormOut}}{1 + e^{-\text{groupnormOut}}}
    $$

## 函数原型

```python
torch_npu.npu_group_norm_silu(input, weight, bias, group, eps=0.00001) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，源数据张量，维度需要为2~8维且第1维度能整除`group`。数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float32`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`float32`、`bfloat16`。

- **weight** (`Tensor`)：可选参数，索引张量，维度为1且元素数量需与输入`input`的第1维度保持相同，数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float32`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`float32`、`bfloat16`。

- **bias** (`Tensor`)：可选参数，更新数据张量，维度为1且元素数量需与输入`input`的第1维度保持相同，数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float32`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`float32`、`bfloat16`。

- **group** (`int`)：必选参数，表示将输入`input`的第1维度分为group组，group需大于0。

  > **group 参数选择指导：**
  > - `group=1`：等价于LayerNorm（层归一化），对整个输入进行归一化，适用于序列建模、全连接层后等场景。
  > - `group=C`（C为通道数）：等价于InstanceNorm（实例归一化），每个通道独立归一化，适用于风格迁移、图像生成等场景。
  > - `1 < group < C`：标准GroupNorm，在通道维度上划分若干组进行归一化，兼顾通道间信息聚合和组内归一化。适用于视觉模型（如ResNet、EfficientNet）的特征归一化，`group=32`是常用的默认值。
  >
  > 选择建议：
  > - 小group值（如group=1或较小的值）：更关注全局统计特性，适合需要捕获整体分布的任务。
  > - 大group值（如group=C或接近C的值）：更关注局部统计特性，适合需要保留通道独立性的任务。
  > - 中等group值（如group=32）：在全局和局部之间取得平衡，是视觉任务的常用选择。

- **eps** (`float`)：可选参数，数值稳定性而加到分母上的值，若保持精度，则eps需大于0。默认值为0.00001。

## 返回值说明

> **返回值用途说明：**
> - `out`：组归一化后经过SiLU激活的最终输出，用于网络的前向传播。
> - `meanOut`：每个分组的均值，用于反向传播时的梯度计算（需要与out一起保存）。
> - `rstdOut`：每个分组标准差的倒数，用于反向传播时的梯度计算（需要与out一起保存）。

- **out** (`Tensor`)：数据类型和shape与`input`相同，支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float32`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`float32`、`bfloat16`。

- **meanOut** (`Tensor`)：数据类型与`input`相同，shape为\(N, group\)，其中N为`input`第0维度值。数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float32`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`float32`、`bfloat16`。

- **rstdOut** (`Tensor`)：数据类型与`input`相同，shape为\(N, group\)，其中N为`input`第0维度值。数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float32`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`float32`、`bfloat16`。

  > **反向传播示例：**
  > ```python
  > # 保存out、meanOut、rstdOut用于反向传播
  > out, meanOut, rstdOut = torch_npu.npu_group_norm_silu(input, weight, bias, group=num_groups, eps=eps)
  > # 在自定义反向传播中，可使用meanOut和rstdOut计算梯度
  > ```

## 约束说明

- 该接口支持推理、训练场景下使用。
- `input`、`weight`、`bias`、`out`、`meanOut`、`rstdOut`数据类型必须在支持的范围之内。
- `out`、`meanOut`、`rstdOut`的数据类型与`input`相同；`weight`、`bias`与`input`可以不同。
- `weight`与`bias`的数据类型必须保持一致，且数据类型的精度不能低于`input`的数据类型。
- `weight`与`bias`的维度需为1且元素数量需与输入`input`的第1维度保持相同。
- `input`维度需大于一维且小于等于八维，且`input`第1维度能整除`group`。
- `input`任意维都需大于0。
- `out`的shape与`input`相同。
- `meanOut`与`rstdOut`的shape为\(N, group\)，其中N为`input`第0维度值。
- `eps`需大于0。
- `group`需大于0。

## 调用示例

```python
import torch
import numpy as np
import torch_npu
     
dtype = np.float32
shape_x = [24,320,48,48]
num_groups = 32
shape_c = [320]
eps = 0.00001
     
input_npu=torch.randn(shape_x,dtype=torch.float32).npu()
weight_npu=torch.randn(shape_c,dtype=torch.float32).npu()
bias_npu=torch.randn(shape_c,dtype=torch.float32).npu()
out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
     
     
input_npu=torch.randn(shape_x,dtype=torch.bfloat16).npu()
weight_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
bias_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
     
input_npu=torch.randn(shape_x,dtype=torch.float16).npu()
weight_npu=torch.randn(shape_c,dtype=torch.float16).npu()
bias_npu=torch.randn(shape_c,dtype=torch.float16).npu()
out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
```
