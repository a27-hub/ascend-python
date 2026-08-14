# torch_npu.npu_add_rms_norm_dynamic_mx_quant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |

> 该算子仅支持 Ascend 950PR/Ascend 950DT（A5）及后续芯片，在 A2/A3 芯片上调用会在入口处直接校验拦截。可选参数 `x0` 额外要求 CANN >= 9.2.0。

## 功能说明

- API功能：先实现Add算子与RmsNorm算子融合，再与DynamicMxQuant算子融合。当残差`x0`存在时，先进行`x0`与`x1`残差的加法后再与`x1`相加。换言之，先对输入进行Add和RmsNorm归一化，将RmsNorm结果与beta相加后再进行动态MX量化，减少搬入搬出操作。其中RmsNorm是大模型常用的归一化操作，相比LayerNorm其去掉了减去均值的部分。注意，当传入可选参数`x0`时，支持三路残差加法（x = x0 + x1 + x2），适用于CANN >= 9.2.0版本。

- 计算公式：

  当`x0`为空时：
  $$
  x=x_{1}+x_{2}
  $$

  当`x0`不为空时：
  $$
  x=x_{0}+x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RMSNorm}(x)=\frac{x}{\operatorname{RMS}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  在尾轴维度上按blocksize=32分组，每组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale,\{P_i\}_{i=1}^{k}\}$：

  - 当`scale_alg`为0时（OCP实现）：
    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize
    $$

    其中emax为对应数据类型的最大正则数的指数位：

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

  - 当`scale_alg`为1时（cuBLAS实现，仅FP8类型）：
    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$

    其中$Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})$为该块中数值的最大绝对值，$Amax(DType)$为目标数据类型可表示的最大值。

## 函数原型

```python
torch_npu.npu_add_rms_norm_dynamic_mx_quant(x1, x2, gamma, *, beta=None, x0=None, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **x1**(`Tensor`)：必选参数，表示用于Add计算的第一个输入，对应公式中的$x1$。数据格式支持$ND$，支持空Tensor，支持非连续Tensor。数据类型支持`float16`、`bfloat16`。支持1-7维张量。
- **x2**(`Tensor`)：必选参数，表示用于Add计算的第二个输入，对应公式中的$x2$。数据格式支持$ND$，支持空Tensor，支持非连续Tensor。数据类型支持`float16`、`bfloat16`。shape与`x1`保持一致。
- **gamma**(`Tensor`)：必选参数，表示RMSNorm的缩放因子（权重），对应公式中的$gamma$。数据格式支持$ND$，支持空Tensor，支持非连续Tensor。数据类型支持`float16`、`bfloat16`、`float32`。shape为一维，元素数量与`x1`最后一维大小一致。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **beta**(`Tensor`)：可选参数，RMSNorm的偏置项，对应公式中的$beta$。数据类型支持`float16`、`bfloat16`、`float32`。shape为一维，元素数量与`x1`最后一维大小一致，且需与`gamma`的shape和数据类型保持一致。默认值为`None`，为`None`时不添加偏置。
- **x0**(`Tensor`)：可选参数，表示Add计算中的可选第三路输入，对应公式中的$x0$。当不为`None`时，计算公式变为$x=x_0+x_1+x_2$。数据类型支持`float16`、`bfloat16`，shape和数据类型需要与`x1`保持一致。支持非连续Tensor。默认值为`None`。需要CANN >= 9.2.0版本支持。
- **epsilon**(`float`)：可选参数，表示添加到分母中的值，以确保数值稳定，对应公式中的$epsilon$。建议传入较小正数。默认值为`1e-6`。
- **scale_alg**(`int`)：可选参数，表示mxscale的计算方法，对应公式中的$scale\_alg$。取值为`0`表示Open Compute Project(OCP)实现，取值为`1`表示cuBLAS实现。当`dst_type`为FP4类型（`float4_e2m1fn_x2`或`float4_e1m2`）时仅支持取值为`0`。默认值为`0`。
- **round_mode**(`str`)：可选参数，表示数据转换的模式，对应公式中的$round\_mode$。当`dst_type`为FP4类型时支持`"rint"`、`"floor"`、`"round"`；当`dst_type`为FP8类型时仅支持`"rint"`。默认值为`"rint"`。
- **dst_type**(`ScalarType`)：可选参数，表示量化输出y的数据类型，对应公式中的$DType$。支持以下类型：

    | dst_type值 | 对应数据类型 | 说明 |
    | :---: | :---: | :--- |
    | `torch_npu.float8_e5m2`（或`23`） | FLOAT8_E5M2 | FP8类型 |
    | `torch_npu.float8_e4m3fn`（或`24`） | FLOAT8_E4M3FN | FP8类型 |
    | `torch_npu.float4_e2m1fn_x2`（或`296`） | FLOAT4_E2M1 | FP4类型，x1最后一维必须为偶数 |
    | `torch_npu.float4_e1m2`（或`297`） | FLOAT4_E1M2 | FP4类型，x1最后一维必须为偶数 |

    默认值为`torch_npu.float4_e2m1fn_x2`。

## 返回值说明

- **y**(`Tensor`)：归一化并量化后的结果，对应公式中的$P_i$和$d_i$。数据类型由`dst_type`决定（FP8或FP4），shape与`x1`一致。
- **x_out**(`Tensor`)：Add计算的结果，对应公式中的$x$。数据类型和shape与输入`x1`保持一致。
- **mxscale**(`Tensor`)：每个分组对应的量化尺度，对应公式中的$mxscale$和$S^b$。数据类型为`float8_e8m0`（以`uint8`存储）。shape约束：rank(mxscale) = rank(x1) + 1，mxscale.shape[-2] = ceil(x1.shape[-1] / 32) 向上取整并偶数pad，mxscale.shape[-1] = 2，其余维度与`x1`一致。
- **rstd**(`Tensor`)：归一化后的标准差的倒数，对应公式中$Rms(x)$的倒数。当`x1`、`x2`（或`x0`）的`requires_grad`为`True`时，数据类型为`float32`，shape为`x1`的shape剔除最后一维（即不需要norm的维度）；否则返回空Tensor。

## 约束说明

- **芯片型号约束**：该算子仅支持 Ascend 950PR/Ascend 950DT（A5）及后续芯片，在 A2/A3 芯片上调用会在接口入口处直接校验拦截并报错。
- 当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2`时，`x1`最后一维的值必须为偶数。
- `gamma`、`beta`的数据类型只能与`x1`的数据类型保持一致或者为`float32`。
- `x1`和`x2`的`requires_grad`必须一致。
- 当传入`x0`时，`x0`的shape和数据类型必须与`x1`保持一致，且需要CANN >= 9.2.0版本。
- 当`dst_type`为FP4类型时，`scale_alg`只能为`0`。
- 当`dst_type`为FP8类型时，`round_mode`只能为`"rint"`。
- **边界值场景说明**：
  - 当输入是Inf时：输出y为0，x_out为Inf，mxscale为255（偶数pad填充值为0），rstd为0。
  - 当输入是NaN时：输出y为0，x_out为NaN，mxscale为255（偶数pad填充值为0），rstd为NaN。

## 调用示例

```python
import torch
import torch_npu

# 基本调用（两路加法，不传x0）
x1 = torch.randn(2, 64, dtype=torch.bfloat16, device='npu')
x2 = torch.randn(2, 64, dtype=torch.bfloat16, device='npu')
gamma = torch.ones(64, dtype=torch.float32, device='npu')
beta = torch.zeros(64, dtype=torch.float32, device='npu')
epsilon = 1e-6

y, x_out, mxscale, rstd = torch_npu.npu_add_rms_norm_dynamic_mx_quant(
    x1, x2, gamma,
    beta=beta,
    epsilon=epsilon,
    scale_alg=0,
    round_mode="rint",
    dst_type=torch_npu.float8_e5m2,
)

print("y.shape:", y.shape)
print("x_out.shape:", x_out.shape)
print("mxscale.shape:", mxscale.shape)
print("rstd.shape:", rstd.shape)

# 三路加法调用（传入x0，需要CANN >= 9.2.0）
x0 = torch.randn(2, 64, dtype=torch.bfloat16, device='npu')
y, x_out, mxscale, rstd = torch_npu.npu_add_rms_norm_dynamic_mx_quant(
    x1, x2, gamma,
    beta=beta,
    x0=x0,
    epsilon=epsilon,
    scale_alg=0,
    round_mode="rint",
    dst_type=torch_npu.float8_e5m2,
)

print("y.shape:", y.shape)
print("x_out.shape:", x_out.shape)
print("mxscale.shape:", mxscale.shape)
print("rstd.shape:", rstd.shape)
```
