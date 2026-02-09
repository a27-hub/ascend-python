# （beta）torch_npu.npu_rms_norm_quant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 推理系列产品</term>   |     √    |

## 功能说明

- API功能：RmsNormQuant算子是大模型常用的标准化操作,相比LayerNorm算子,其去掉了减去均值的部分。AddRmsNormQuant算子将RmsNorm前的Add算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出的操作。
- 计算公式：
  
  $$
  x_i = x1_{i1} + x2_{i2}
  $$

  $$
  y_i = \frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, where \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$
  - divMod为True时: 
  $$
  y1 = round((y/scales1) + zero_points1)
  $$

  $$
  y2 = round((y/scales2) + zero_points2)
  $$
  - divMod为False时: 
  $$
  y1 = round((y * scales1) + zero_points1)
  $$

  $$
  y2 = round((y * scales2) + zero_points2)
  $$

## 函数原型

```
torch_npu.npu_rms_norm_quant(x, gamma, beta, scale, offset, epsilon=1e-06) -> Tensor
```

## 参数说明

- **x1**（`aclTensor*`，计算输入）：表示标准化过程中的源数据张量，对应公式中的`x1`。shape支持1-8维，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。<term>Atlas 推理系列产品</term>：数据类型支持`float16`。<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
- **x2**（`aclTensor*`，计算输入）：表示标准化过程中的源数据张量，对应公式中的`x2`。shape支持1-8维，shape和数据类型需与`x1`一致，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。<term>Atlas 推理系列产品</term>：数据类型支持`float16`。<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
- **gamma**（`aclTensor*`，计算输入）：表示标准化过程中的权重张量，对应公式中的`g`。shape支持1-8维，shape需与`x1`需要Norm的维度保持一致，数据类型需与`x1`一致，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。<term>Atlas 推理系列产品</term>：数据类型支持`float16`。<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
- **scales1**（`aclTensor*`，计算输入）：表示量化过程中对`y1`进行scale的张量，对应公式中的`scales1`。shape需与`gamma`一致，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。当参数`divMode`为True时，该参数值不能为0。<term>Atlas 推理系列产品</term>：数据类型支持`float32`。<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`、`bfloat16`。
- **scales2Optional**（`aclTensor*`，计算输入）：表示量化过程中对`y2`进行scale的张量，对应公式中的`scales2`。可选参数，支持传入空指针。shape需与`gamma`一致，数据类型需与`scales1`一致，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。当参数`divMode`为True时，该参数值不能为0。<term>Atlas 推理系列产品</term>：数据类型支持`float32`。<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`、`bfloat16`。预留参数，实际未使用。
- **zeroPoints1Optional**（`aclTensor*`，计算输入）：表示量化过程中对`y1`进行offset的张量，对应公式中的`zero_points1`。可选参数，支持传入空指针。shape需与`gamma`一致，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。<term>Atlas 推理系列产品</term>：数据类型支持`int32`。<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int32`、`bfloat16`。
- **zeroPoints2Optional**（`aclTensor*`，计算输入）：表示量化过程中对`y2`进行offset的张量，对应公式中的`zero_points2`。可选参数，支持传入空指针。shape需与`gamma`一致，数据类型需与`zeroPoints1Optional`一致，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。<term>Atlas 推理系列产品</term>：数据类型支持`int32`。<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int32`、`bfloat16`。预留参数，实际未使用。
- **axis**（`int64_t`，计算输入）：Host侧整型，表示需要进行量化的elewise轴，其他轴做broadcast，指定轴不能超过输入`x`的维度数。当前仅支持`-1`，传其他值均不生效。
- **epsilon**（`double`，计算输入）：公式中的输入`eps`，用于防止除0错误，建议传较小的正数。
- **divMode**（`bool`，计算输入）：决定量化公式是否使用除法的参数。<term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持`True`。

## 返回值说明

- **y1Out**（`aclTensor*`，计算输出）：量化输出Tensor，对应公式中的`y1`。shape支持1-8维，shape需与`x1`/`x2`一致，数据类型支持`int8`，数据格式支持$ND$，支持非连续Tensor。
- **y2Out**（`aclTensor*`，计算输出）：量化输出Tensor，对应公式中的`y2`。shape支持1-8维，shape需与`x1`/`x2`一致，数据类型支持`int8`，数据格式支持$ND$，支持非连续Tensor。预留参数，实际未使用，输出为随机值。
- **xOut**（`aclTensor*`，计算输出）：`x1`和`x2`的和，对应公式中的`x`。shape支持1-8维，shape和数据类型需与`x1`/`x2`一致，数据格式支持$ND$，支持非连续Tensor。<term>Atlas 推理系列产品</term>：数据类型支持`float16`。<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。

## 约束说明

- <term>Atlas 推理系列产品</term>：`x1`、`x2`、`y1Out`、`y2Out`、`xOut`的尾轴长度，以及`gamma`、`scales1`、`scales2Optional`、`zeroPoints1Optional`、`zeroPoints2Optional`的长度必须大于等于32 Bytes。
- 数据格式说明：所有输入输出Tensor的数据格式推荐使用$ND$格式，其他数据格式会由框架默认转换成$ND$格式进行处理。
- 各产品支持数据类型说明：
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    | 参数 | 支持数据类型 |
    | -------- | -------- |
    | `x1` | `float16` / `bfloat16` |
    | `x2` | `float16` / `bfloat16` |
    | `gamma` | `float16` / `bfloat16` |
    | `scales1` | `float32` / `bfloat16` |
    | `scales2Optional` | `float32` / `bfloat16` |
    | `zeroPoints1Optional` | `int32` / `bfloat16` |
    | `zeroPoints2Optional` | `int32` / `bfloat16` |
    | `y1Out` | `int8` |
    | `y2Out` | `int8` |
    | `xOut` | `float16` / `bfloat16` |
  - <term>Atlas 推理系列产品</term>：
    | 参数 | 支持数据类型 |
    | -------- | -------- |
    | `x1` | `float16` |
    | `x2` | `float16` |
    | `gamma` | `float16` |
    | `scales1` | `float32` |
    | `scales2Optional` | `float32` |
    | `zeroPoints1Optional` | `int32` |
    | `zeroPoints2Optional` | `int32` |
    | `y1Out` | `int8` |
    | `y2Out` | `int8` |
    | `xOut` | `float16` |

## 调用示例

```python
>>> eps = 1e-6
>>> x = torch.randn(16, dtype=torch.float16).npu()
>>> gamma = torch.randn(16, dtype=torch.float16).npu()
>>> beta = torch.zeros(16, dtype=torch.float16).npu()
>>> scale = torch.ones(1, dtype=torch.float16).npu()
>>> offset = torch.zeros(1, dtype=torch.int8).npu()
>>> y = torch_npu.npu_rms_norm_quant(x, gamma, beta, scale, offset, eps)
>>> y.cpu().numpy()
tensor([ 1, -1,  2,  0, -2,  1,  0,  1,  2,  0,  2,  0,  0,  0,  0,  0],
       device='npu:0', dtype=torch.int8)