# torch_npu.npu_add_rms_norm_quant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |


## 功能说明

- API功能：RmsNorm是大模型常用的标准化操作，相比LayerNorm其去掉了减去均值的部分。torch_npu.npu_add_rms_norm_quant算子将RmsNorm前的Add算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  x_i={x1}_{i1}+{x2}_{i2}
  $$

  $$
  y_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - divMod为True时：

    $$
    y1=round((y/scales1)+zero\_points1)
    $$

    $$
    y2=round((y/scales2)+zero\_points2)
    $$
  - divMod为False时：

    $$
    y1=round((y*scales1)+zero\_points1)
    $$

    $$
    y2=round((y*scales2)+zero\_points2)
    $$

## 函数原型

```
torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1, beta=None, scales2=None, zero_points2=None, *, axis=-1, epsilon=1e-06, div_mode=True) -> (y1Out, y2Out, xOut)
```

## 参数说明

- **x1** (`Tensor`)：必选参数，表示标准化过程中的源数据张量，公式中的`x1`。`shape`支持1-8维。数据格式支持`ND`，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持`float32`、`float16`、`bfloat16`。
- **x2** (`Tensor`)：必选参数，表示标准化过程中的源数据张量，公式中的`x2`。`shape`支持1-8维，`shape`和数据类型需要与`x1`保持一致。数据格式支持`ND`，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持`float32`、`float16`、`bfloat16`。
- **gamma** (`Tensor`)：必选参数，表示标准化过程中的权重张量，公式中的`g`。`shape`支持1-8维，`shape`需要与`x1`需要Norm的维度保持一致，数据类型需要与`x1`保持一致。数据格式支持`ND`，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、昇腾910_95I处理器、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持`float32`、`float16`、`bfloat16`。
- **scales1** (`Tensor`)：必选参数，表示量化过程中得到`y1`进行的`scales`张量，公式中的`scales1`。`shape`需要与`gamma`保持一致，或者最后一维和`gamma`保持一致，其他维度为1。数据格式支持`ND`，支持非连续的Tensor。当参数`div_mode`的值为`True`时，该参数的值不能为0。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float32`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`、`bfloat16`。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持`float32`、`float16`、`bfloat16`。
- **scales2** (`Tensor`)：可选参数，表示量化过程中得到`y2`进行的`scales`张量，公式中的`scales2`。可选参数，支持传入空指针。`shape`需要与`scales1`保持一致，数据类型需要与`scales1`保持一致。数据格式支持`ND`，支持非连续的Tensor。当参数`div_mode`的值为`True`时，该参数的值不能为0。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float32`。预留参数，实际未使用。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`、`bfloat16`。预留参数，实际未使用。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持`float32`、`float16`、`bfloat16`。
- **zero_points1** (`Tensor`)：可选参数，表示量化过程中得到`y1`进行的`offset`张量，公式中的`zero_points1`。可选参数，支持传入空指针。`shape`需要与`scales1`保持一致。数据格式支持`ND`，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`int32`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int32`、`bfloat16`。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持`int32`、`float32`、`float16`、`bfloat16`。
- **beta** (`Tensor`)：可选参数，表示标准化过程中的偏置张量，公式中的`beta`。可选参数，支持传入空指针。`shape`需要与`gamma`保持一致，数据类型需要与`gamma`保持一致。数据格式支持`ND`，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持`float32`、`float16`、`bfloat16`。
- **zero_points2** (`Tensor`)：可选参数，表示量化过程中得到`y2`进行的`offset`张量，公式中的`zero_points2`。可选参数，支持传入空指针。`shape`需要与`scales1`保持一致，数据类型需要与`zero_points1`保持一致。数据格式支持`ND`，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`int32`。预留参数，实际未使用。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int32`、`bfloat16`。预留参数，实际未使用。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持`int32`、`float32`、`float16`、`bfloat16`。
- **axis** (`int64_t`)：可选参数，Host侧的整型，表示需要进行量化的elewise轴，其他的轴做broadcast，指定的轴不能超过输入`x`的维度数。当前仅支持-1，传其他值均不生效。
- **epsilon** (`double`)：可选参数，公式中的输入`eps`，用于防止除0错误，数据类型为`double`。建议传较小的正数。
- **div_mode** (`bool`)：可选参数，公式中决定量化公式是否使用除法的参数，数据类型为`bool`。
    - <term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持`True`。
    - <term>昇腾910_95 AI处理器</term>：支持`True`和`False`。

## 返回值说明

- **y1Out** (`Tensor`)：表示量化输出Tensor，公式中的`y1`。`shape`支持1-8维度，`shape`需要与输入`x1`/`x2`一致，数据类型支持`int8`，数据格式支持`ND`，支持非连续的Tensor。
- **y2Out** (`Tensor`)：表示量化输出Tensor，公式中的`y2`。`shape`支持1-8维度，`shape`需要与输入`x1`/`x2`一致，数据类型支持`int8`，数据格式支持`ND`，支持非连续的Tensor。预留参数，实际未使用，输出为随机值。
- **xOut** (`Tensor`)：表示`x1`和`x2`的和，公式中的`x`。`shape`支持1-8维度，`shape`和数据类型需要与输入`x1`/`x2`一致，数据格式支持`ND`，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持`float32`、`float16`、`bfloat16`。

## 约束说明
- <term>Atlas 推理系列产品</term>：`x1`、`x2`、`y1Out`、`y2Out`、`xOut`的norm轴长度，以及`gamma`、`scales1`、`scales2`、`zero_points1`、`zero_points2`的长度必大于等于32Bytes。

- **支持类型说明**

  是否支持空Tensor：支持空进空出。

- **数据格式说明**
  
  所有输入输出Tensor的数据格式推荐使用`ND`格式，其他数据格式会由框架默认转换成`ND`格式进行处理。

- **各产品型号支持数据类型说明**

  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    | x1数据类型 | x2数据类型 | gamma数据类型 | beta数据类型 | scales1数据类型 | scales2数据类型 | zero_points1数据类型 | zero_points2数据类型 | y1Out数据类型 | y2Out数据类型 | xOut数据类型 |
    | ---------- | ---------- | ------------- | ------------ | --------------- | ----------------------- | --------------------------- | --------------------------- | ------------- | ------------- | ------------ |
    | float16    | float16    | float16       | float16      | float32         | float32                 | int32                       | int32                       | int8          | int8          | float16      |
    | bfloat16   | bfloat16   | bfloat16      | bfloat16     | bfloat16        | bfloat16                | bfloat16                    | bfloat16                    | int8          | int8          | bfloat16     |
  - <term>Atlas 推理系列产品</term>：
    | x1数据类型 | x2数据类型 | gamma数据类型 | beta数据类型 | scales1数据类型 | scales2数据类型 | zero_points1数据类型 | zero_points2数据类型 | y1Out数据类型 | y2Out数据类型 | xOut数据类型 |
    | ---------- | ---------- | ------------- | ------------ | --------------- | ----------------------- | --------------------------- | --------------------------- | ------------- | ------------- | ------------ |
    | float16    | float16    | float16       | float16      | float32         | float32                 | int32                       | int32                       | int8          | int8          | float16      |
  - <term>昇腾910_95 AI处理器</term>：
    | x1数据类型 | x2数据类型 | gamma数据类型 | beta数据类型 | scales1数据类型 | scales2数据类型 | zero_points1数据类型 | zero_points2数据类型 | y1Out数据类型 | y2Out数据类型 | xOut数据类型 |
    | ---------- | ---------- | ------------- | ------------ | --------------- | ----------------------- | --------------------------- | --------------------------- | ------------- | ------------- | ------------ |
    | float16    | float16    | float16       | float16      | float32         | float32                 | int32                       | int32                       | int8          | int8          | float16      |
    | bfloat16   | bfloat16   | bfloat16      | bfloat16     | float32         | float32                 | int32                       | int32                       | int8          | int8          | bfloat16     |
    | float32    | float32    | float32       | float32      | float32         | float32                 | float32                     | float32                     | int8          | int8          | float32      |
    | float16    | float16    | float16       | float16      | float16         | float16                 | float16                     | float16                     | int8          | int8          | float16      |
    | bfloat16   | bfloat16   | bfloat16      | bfloat16     | bfloat16        | bfloat16                | bfloat16                    | bfloat16                    | int8          | int8          | bfloat16     |
    | float16    | float16    | float16       | float16      | float32         | float32                 | float32                     | float32                     | int8          | int8          | float16      |
    | bfloat16   | bfloat16   | bfloat16      | bfloat16     | float32         | float32                 | float32                     | float32                     | int8          | int8          | bfloat16     |
    
## 调用示例

运行样例。

```python
import math

import numpy as np
import torch

import torch_npu

def test_npu_add_rms_norm_quant():
    shape_list = [[[16, ], [16, ]],
                  [[2, 16], [16, ]],
                  [[2, 16], [2, 16]],
                  [[16, 32], [16, 32]],
                  [[16, 32], [32, ]],
                  [[2, 2, 2, 8, 16, 32], [2, 2, 2, 8, 16, 32]],
                  [[2, 2, 2, 8, 16, 32], [16, 32]],
                  [[2, 2, 2, 8, 16, 32], [32, ]],
                  [[2, 2, 2, 2, 2, 16, 32], [2, 2, 2, 2, 2, 16, 32]],
                  [[2, 2, 2, 2, 2, 16, 32], [16, 32]],
                  [[2, 2, 2, 2, 2, 16, 32], [32, ]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [2, 2, 2, 2, 2, 8, 16, 32]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [16, 32]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [32, ]]]
    for item in shape_list:
        x_shape = item[0]
        quant_shape = item[1]
        x1 = torch.randn(x_shape, dtype=torch.float16)
        x2 = torch.randn(x_shape, dtype=torch.float16)
        gamma = torch.randn(quant_shape, dtype=torch.float16)
        beta = torch.randn(quant_shape, dtype=torch.float16)
        scales1 = torch.randn(quant_shape, dtype=torch.float32)
        zero_points1 = torch.randint(-10, 10, quant_shape, dtype=torch.int32)

        x1_npu = x1.npu()
        x2_npu = x2.npu()
        gamma_npu = gamma.npu()
        beta_npu = beta.npu()
        scales1_npu = scales1.npu()
        zero_points1_npu = zero_points1.npu()

        y1_v1, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu)
        y1_v2, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu, beta_npu) 

if __name__ == "__main__":
    test_npu_add_rms_norm_quant()
```