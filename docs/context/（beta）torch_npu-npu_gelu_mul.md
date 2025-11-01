# （beta）torch_npu.npu_gelu_mul

# 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>            |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>               | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

- API功能：将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行Gelu计算，将计算结果与x2相乘。
- 计算公式：

    给定输入张量 $\text{input}$（最后一维长度为 $2d$，$d$ 为正整数），函数 $\text{GeluMul}$ 计算流程如下：

    - 按最后一维将 $\text{input}$ 拆分为两个等长张量：
            $$x_1 = \text{input}[..., :d], \quad x_2 = \text{input}[..., d:]$$
            其中 $x_1$、$x_2$ 形状与 $\text{input}$ 除最后一维外一致，最后一维长度均为 $d$。

    - 对 $x_1$ 应用 GELU 激活（模式由 $\text{approximate}$ 控制），即 $x_1 = \text{GELU}(x_1)$：
        - "tanh" 近似模式（高效）：
                        $$\text{GELU}(x) = 0.5 \cdot x \cdot \left[ 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \cdot \left( x + 0.044715 \cdot x^3 \right) \right) \right]$$
        - "none" 精确模式（高精度）：
                        $$\text{GELU}(x) = 0.5 \cdot x \cdot \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$


    - 逐元素乘积输出：激活后的 $x_1$ 与 $x_2$ 逐元素相乘，得到最终结果：
            $$\text{out} = x_1 \cdot x_2$$
            其中 $\text{out}$ 形状与原始输入 $\text{input}$ 完全一致。


## 函数原型

```
torch_npu.npu_gelu_mul(input, *, approximate="none") -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选参数，输入张量，数据类型支持BFLOAT16、FLOAT16、FLOAT。支持非连续的Tensor，数据格式支持ND，shape维度2至8维，且shape满足如下要求：最后一维值为偶数且小于等于1024。其他维度的乘积小于等于20000。
- **approximate** (`String`)：可选参数，计算输入, Gelu计算的模式，只支持“none”和“tanh”，分别对应Gelu的erf模式和tanh模式，默认值为“none”。

## 返回值说明
- **out**（`Tensor`）：输出张量，数据类型支持BFLOAT16、FLOAT16、FLOAT。shape维度2至8维。支持非连续的Tensor，数据格式支持ND，输出的数据类型与输入保持一致，输出shape和输入shape其他维度一致，最后一维的值为输入shape最后一维值的二分之一。

## 约束说明

- 典型场景尾轴为16的倍数，当尾轴为非32B对齐时，建议走小算子拼接逻辑。

## 调用示例

```python
>>> import torch, torch_npu
>>> shape = [100, 400]
>>> input = torch.rand(shape, dtype=torch.float16).npu()
>>> output = torch_npu.npu_gelu_mul(input, approximate=mode)

```
