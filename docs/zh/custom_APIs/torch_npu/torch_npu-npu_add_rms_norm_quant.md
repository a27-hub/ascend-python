# torch_npu.npu_add_rms_norm_quant

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
|<term>Atlas 推理系列产品</term>                               |    √     |

## 功能说明

- API功能：AddRmsNormQuant算子将RmsNorm前的Add操作、RmsNorm标准化操作以及RmsNorm后的Quantize量化操作融合为单个算子，减少搬入搬出操作。该接口在AddRmsNormQuant基础上增加了偏置项`beta`参数和RmsNorm归一化结果输出`rms_norm_out`。

- 计算公式：

  $$
  x_i={x1}_i+{x2}_i
  $$

  $$
  y_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i + beta_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  rmsNormOut_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i
  $$

  - div_mode为True时：

    $$
    y1Out=round((y/scales1)+zero\_points1)
    $$

    $$
    y2Out=round((y/scales2)+zero\_points2)
    $$

  - div_mode为False时：

    $$
    y1Out=round((y*scales1)+zero\_points1)
    $$

    $$
    y2Out=round((y*scales2)+zero\_points2)
    $$

> [!NOTE]
> `rms_norm_out`为RmsNorm归一化结果（不含beta），`beta`与`rms_norm_out`相互独立，可任意组合使用。

## 函数原型

```
torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1, *, scales2=None, zero_points1=None, zero_points2=None, beta=None, axis=-1, epsilon=1e-06, div_mode=True, output_res=False, dst_type=None) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **x1** (`Tensor`)：必选参数，标准化过程中的源数据张量。数据类型支持`float16`、`bfloat16`、`float32`。数据格式支持$ND$。维度为1-8。

- **x2** (`Tensor`)：必选参数，标准化过程中的源数据张量。shape和数据类型与`x1`保持一致。数据格式支持$ND$。维度为1-8。

- **gamma** (`Tensor`)：必选参数，标准化过程中的权重张量。shape与`x1`需要norm的维度保持一致，数据类型与`x1`保持一致。数据格式支持$ND$。

- **scales1** (`Tensor`)：必选参数，y1量化scale张量。shape与`gamma`保持一致。数据格式支持$ND$。

- <strong>*</strong>：位置参数与关键字参数的分隔符。

- **scales2** (`Tensor`)：可选参数，y2量化scale张量。默认值为`None`。传入时shape和数据类型与`scales1`保持一致。`div_mode`为True时值不能为0。

- **zero_points1** (`Tensor`)：可选参数，y1量化offset张量。默认值为`None`。传入时shape与`scales1`保持一致。数据类型支持`int32`、`float32`、`float16`、`bfloat16`。

- **zero_points2** (`Tensor`)：可选参数，y2量化offset张量。默认值为`None`。传入时shape与`scales1`保持一致，数据类型与`zero_points1`保持一致。

- **beta** (`Tensor`)：可选参数，标准化过程中的偏置项。默认值为`None`。传入时shape和数据类型与`gamma`保持一致。

- **axis** (`int`)：可选参数，需要进行量化的element-wise轴。当前仅支持`-1`。默认值为`-1`。

- **epsilon** (`float`)：可选参数，防除0常量。建议传较小的正数。默认值为`1e-6`。

- **div_mode** (`bool`)：可选参数，量化公式选择。`True`→y/scale+zp；`False`→y*scale+zp。默认值为`True`。

- **output_res** (`bool`)：可选参数，控制`rms_norm_out`是否包含有效数据。`True`时内核写入有效值；`False`时张量仍分配但内容无意义。默认值为`False`。

- **dst_type** (`int`)：可选参数，y1/y2量化输出数据类型。默认值为`None`（等效`torch.int8`）：
  - `1` (`torch.int8`)：INT8（默认）
  - `290` (`torch_npu.hifloat8`)：HIFLOAT8（仅950系列）
  - `291` (`torch_npu.float8_e5m2`)：FLOAT8_E5M2（仅950系列）
  - `292` (`torch_npu.float8_e4m3fn`)：FLOAT8_E4M3FN（仅950系列）

## 返回值说明

- **y1** (`Tensor`)：量化输出。shape与`x1`一致，dtype由`dst_type`决定（默认`torch.int8`）。
- **y2** (`Tensor`)：量化输出。shape和dtype与`y1`一致。
- **x_out** (`Tensor`)：x1+x2结果。shape和dtype与`x1`一致。
- **rms_norm_out** (`Tensor`)：RmsNorm结果（不含beta）。shape和dtype与`x1`一致。`output_res=False`时内容无意义。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 各参数shape每维大小不超过2147483647。
- `beta`与`rms_norm_out`相互独立，可任意组合使用。
- `axis`当前仅支持`-1`。
- 所有Tensor推荐使用ND格式。
- 各产品数据类型支持情况：

  - <term>Atlas A2/A3 训练/推理系列产品</term>：x1/x2/gamma/x_out/rms_norm_out: `float16`/`bfloat16`；scales1/scales2: `float32`/`bfloat16`；zero_points: `int32`/`bfloat16`；beta: `float16`/`bfloat16`；y1/y2: `int8`；div_mode仅True。

  - <term>Atlas 推理系列产品</term>：x1/x2/gamma/x_out/rms_norm_out仅`float16`；scales1/scales2仅`float32`；zero_points仅`int32`；beta仅`float16`；y1/y2仅`int8`；div_mode仅True。

  - <term>Ascend 950PR/950DT</term>：支持`float32`/`float16`/`bfloat16`/`int32`/`int8`/`hifloat8`/`float8_e5m2`/`float8_e4m3fn`全类型组合，div_mode支持True/False。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x1 = torch.randn((32, 256), dtype=torch.float16, device='npu')
    >>> x2 = torch.randn((32, 256), dtype=torch.float16, device='npu')
    >>> gamma = torch.randn((256,), dtype=torch.float16, device='npu') * 0.1
    >>> scales1 = torch.rand((256,), dtype=torch.float32, device='npu') * 0.01 + 0.01
    >>> y1, y2, x_out, rms_norm_out = torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1)
    >>> y1.shape
    torch.Size([32, 256])
    >>> y1.dtype
    torch.int8
    ```

- 带可选参数的调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x1 = torch.randn((32, 256), dtype=torch.float16, device='npu')
    >>> x2 = torch.randn((32, 256), dtype=torch.float16, device='npu')
    >>> gamma = torch.randn((256,), dtype=torch.float16, device='npu') * 0.1
    >>> scales1 = torch.rand((256,), dtype=torch.float32, device='npu') * 0.01 + 0.01
    >>> beta = torch.randn((256,), dtype=torch.float16, device='npu') * 0.1
    >>> y1, y2, x_out, rms_norm_out = torch_npu.npu_add_rms_norm_quant(
    ...     x1, x2, gamma, scales1, beta=beta, output_res=True)
    >>> rms_norm_out.shape
    torch.Size([32, 256])
    ```