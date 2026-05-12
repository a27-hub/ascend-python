# torch_npu.npu_rms_norm_quant_v2

> [!NOTICE]  
> 此接口为本版本新增功能，具体依赖要求请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

| 产品 | 是否支持 |
| :---------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                        |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                        |    x    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    x    |
| <term>Atlas 200I/500 A2 推理产品</term>                                         |    x    |
| <term>Atlas 推理系列产品</term>                                                |    x    |

## 功能说明

- API 功能：RmsNorm算子是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。RmsNormQuantV3算子将RmsNorm算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。同时在RmsNormQuantV2算子的基础上新增了Rstd的输出。
- 计算公式：
  
  $$
 	quant\_in_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i + beta_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^+epsilon}
 	$$

  - divMode为True时：
 	$$
 	y=round((quant\_in/scale)+offset)
 	$$

  - divMode为False时：
 	$$
 	y=round((quant\_in*scale)+offset)
 	$$

## 函数原型

```python
torch_npu.npu_rms_norm_quant_v2(x, beta, scale, *, offset, gamma, epsilon=1e-06, div_mode=True, dst_dtype=1) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，表示标准化过程中的源数据张量，对应公式中的$x$，数据格式支持 `ND`，shape支持 1-8 维，支持非连续的`Tensor`，不支持空`Tensor`。

- **gamma** (`Tensor`)：必选参数，表示标准化过程中的缩放张量，对应公式中的$g$，shape支持1-2维，若shape为1维，则需与`x`最后一维维度一致；若shape为2维，则第一维必须为1，第二维需与`x`最后一维维度一致。数据类型需与`x`保持一致，数据格式支持`ND`，支持非连续`Tensor`，不支持空`Tensor`。

- **scale** (`Tensor`)：必选参数，表示量化过程中得到`y`进行的`scale`张量，对应公式中的$scale$。shape为1，维度为1。数据格式支持`ND`，支持非连续的`Tensor`，不支持空`Tensor`。该参数的值不能为0。

- **offset** (`Tensor`)：可选参数，表示量化过程中的偏移张量，对应公式中的$offset$。shape需与 `scale` 保持一致，数据格式支持 `ND`，支持非连续 `Tensor`，不支持空 `Tensor`。

- **beta** (`Tensor`)：可选参数，表示标准化过程中的偏移张量，对应公式中的$b$。shape支持1-2维，规则同`gamma`。数据类型需与`x`保持一致，数据格式支持`ND`，支持非连续`Tensor`，不支持空`Tensor`。

- **epsilon** (`double`)：可选参数，对应公式中的$eps$，用于防止除零错误，默认值为 `1e-6`。建议传入较小的正数。

- **div_mode** (`bool`)：可选参数，对应公式中的$div_mode$，决定量化公式是否使用除法的参数，数据类型为bool，默认值为 `True`。

- **dst_dtype** (`int`): 可选参数，指定量化输出的类型，默认值为`1`。传`1`时当做int8处理。

## 返回值说明
  
  `Tensor`
  
  返回结果，对应公式中的$y$，即最终量化输出张量，数据类型由`dst_dtype`指定。当`dst_dtype`是`quint4x2`时，`y`的数据类型为`int32`，形状最后一维为`x`最后一维除以8，其余维度与`x`一致，每个`int32`元素包含8个`int4`结果。其他场景下`y`形状与输入`x`一致，数据类型由`dst_dtype`指定。

  `Tensor`
  
  返回结果，对应公式中的$rstd$，即RMS(x)的倒数。`rstd`形状与输入`x`一致，除开最后一维固定为`1`，数据类型为`float32`。当`x`不存在requires_grad属性时，该参数的最终输出无效。

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：当`y`的数据类型为INT4时，`x`、`gamma`以及`beta`的最后一维必须为偶数。
- <term>Ascend 950PR/Ascend 950DT</term>：当`y`的数据类型为INT32时，`y`的最后一维必须是`x`最后一维的1/8。
- 各产品型号支持数据类型说明：

  - <term>Ascend 950PR/Ascend 950DT</term>：

    | x数据类型 | gamma数据类型 | scale数据类型 | offsetOptional数据类型 | betaOptional数据类型 |epsilon数据类型 | y数据类型 | rstd数据类型 |
    | --------- | ------------- |  ------------- | -------------- |------------- | --------- |--------- |--------- |
    | FLOAT16   | FLOAT16       |  FLOAT16       | INT8           |FLOAT16       | DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  BFLOAT16       | INT8           | BFLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | FLOAT16   | FLOAT16       |  FLOAT16       | FLOAT16           | FLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  BFLOAT16       | BFLOAT16           |BFLOAT16       | DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | FLOAT32   | FLOAT32       |  FLOAT32       | FLOAT32           |FLOAT32       | DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | FLOAT16   | FLOAT16       |  FLOAT32       | INT32           |FLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  FLOAT32      | INT32           |BFLOAT16       | DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | FLOAT16   | FLOAT16       |  FLOAT32       | FLOAT32           | FLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  FLOAT32       | FLOAT32           | BFLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8     | FLOAT32       |

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> eps = 1e-6
>>> div_mode = True
>>> dst_dtype = 1
>>> x = torch.randn(16, dtype=torch.float16).npu()
>>> gamma = torch.randn(16, dtype=torch.float16).npu()
>>> scale = torch.ones(1, dtype=torch.float16).npu()
>>> offset = torch.zeros(1, dtype=torch.int8).npu()
>>> beta = torch.zeros(16, dtype=torch.float16).npu()
>>> y, rstd = torch_npu.npu_rms_norm_quant_v2(x, gamma, scale, offset=scale, beta=beta, eps=eps, div_mode=div_mode, dst_dtype=dst_dtype)
>>> y.cpu().numpy()
    tensor([ 1, -1,  2,  0, -2,  1,  0,  1,  2,  0,  2,  0,  0,  0,  0,  0],
        device='npu:0', dtype=torch.int8)
```
