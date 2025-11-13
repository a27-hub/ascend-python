# torch_npu.npu_clipped_swiglu

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |

## 功能说明

-   API功能：带截断的Swish门控线性单元激活函数，实现x的变体SwiGlu计算。本接口相较于torch_npu.npu_swiglu，新增了部分输入参数：groupIndex、alpha、limit、bias、interleaved，用于支持
    GPT-OSS模型使用的变体SwiGlu以及MoE模型使用的分组场景。

-   计算公式：

    对给定的输入张量 x ，其维度为[a,b,c,d,e,f,g…]，aclnnClippedSwiglu对其进行以下计算：

    -   将 x 基于输入参数 dim 进行合轴，合轴后维度为[pre, cut, after]。其中 cut 轴为合轴之后需要切分为两个张量的轴，切分方式分为前后切分或者奇偶切分；pre，after 可以等于1。例如当 dim 为3，合轴后 x 的维度为[a * b * c, d, e * f * g * …]。此外，由于after轴的元素为连续存放，且计算操作为逐元素的，因此将cut轴与after轴合并，得到x的维度为[pre, cut * after]。

    -   根据输入参数 group_index, 对 x 的pre轴进行过滤处理，公式如下：
        $$
        sum = \text{Sum}(group\_index)
        $$

        $$
        x = x[ : sum, : ]
        $$
        其中sum表示group_index的所有元素之和。当不输入 group_index 时，跳过该步骤。

    -   根据输入参数 interleaved，对 x 进行切分，公式如下：

        当 interleaved 为 true 时，表示奇偶切分：
        $$
        A = x[ : , : : 2]
        $$

        $$
        B = x[ : , 1 : : 2]
        $$

        当 interleaved 为 false 时，表示前后切分：
        $$
        h = x.shape[1] // 2
        $$

        $$
        A = x[ : , : h]
        $$

        $$
        B = x[ : , h : ]
        $$

    -   根据输入参数 alpha、limit、bias 进行变体SwiGlu计算，公式如下：
        $$
        A = A.clamp(min=None, max=limit)
        $$
        
        $$
        B = B.clamp(min=-limit, max=limit)
        $$
        
        $$
        y\_glu = A * sigmoid(alpha * A)
        $$
        
        $$
        y = y\_glu * (B + bias)
        $$
    
    -   重塑输出张量y的维度数量与合轴前的x的维度数量一致，dim轴上的大小为x的一半，其他维度与x相同。
    
## 函数原型

```
torch_npu.npu_clipped_swiglu(Tensor x, *, Tensor? group_index=None, int dim=-1, float alpha=1.702, float limit=7.0, float bias=1.0, bool interleaved=True) -> Tensor
```

## 参数说明

-   **x** (`Tensor`)：必选参数，表示目标张量。数据类型支持`float16`、`bfloat16`、`float32`，不支持非连续的`Tensor`，数据格式为$ND$，`x`的维数必须大于1维且`dim`轴为偶数。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
-   **group_index** (`Tensor`)：可选参数，当前只支持count模式，表示该模式下指定分组的Tokens数（要求非负整数）。要求为1维张量，数据类型支持`int64`，数据格式$ND$。
-   **dim** (`int`)：可选参数，表示需要对x进行切分的维度序号，默认 -1。
-   **alpha** (`float`)：可选参数，表示glu激活函数系数，默认 1.702。
-   **limit** (`float`)：可选参数，表示变体swiglu输入门限，默认 7.0。
-   **bias** (`float`)：可选参数，表示变体swiglu计算中的偏差，默认 1.0。
-   **interleaved** (`bool`)：可选参数，表示输入`x`是否按奇偶方式切分，`true`表示为奇偶方式切分，`false`表示为前后方式切分，默认为`true`。

## 返回值说明

-   **y** (`Tensor`)：表示激活函数的输出，数据类型同输入`x`，在维度上，第dim维是输入`x`的`1/2`，其余维度与输入x相同，数据格式为$ND$。

## 调用示例
```python
import os
import shutil
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

class TestNPUClippedSwiglu(TestCase):
    def test_npu_clipped_swiglu(self, device="npu"):
        tokens_num = 4608
        hidden_size = 2048
        x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.float32)
        group_index = torch.randint(1, 101, (4, ), dtype=torch.int64)
        y = torch_npu.npu_clipped_swiglu(
            x.npu(),
            group_index=group_index.npu(),
            dim=-1,
            alpha=1.702,
            limit=7.0,
            bias=1.0,
            interleaved=True
        )

if __name__ == "__main__":
    run_tests()
```