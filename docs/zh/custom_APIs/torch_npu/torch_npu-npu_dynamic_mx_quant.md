# torch_npu.npu_dynamic_mx_quant

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- API功能：目的数据类型为FLOAT4类、FLOAT8类的MX量化。在给定的轴axis上，每block_size个数为一组，计算出这组数对应的量化尺度mxscale，然后对组内每一个数除以mxscale，根据round_mode转换到对应的dst_type，得到量化结果y。

- 计算公式：
  - 场景1，当scale_alg为0时，即OCP Microscaling Formats (Mx) Specification实现：
    - 将输入x在axis维度上按k = block_size个数分组，一组k个数$\{V_i\}_{i=1}^{k}$动态量化为$\{mxscale, \{P_i\}_{i=1}^{k}\}$：

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
      mxscale = 2^{shared\_exp} \\
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space block\_size
      $$

    - 量化后的$P_i$按对应的$V_i$的位置组成输出y，mxscale按对应的axis维度上的分组组成输出mxscale。
    - emax表示对应数据类型的最大正则数的指数位，对应关系如下：

      | DataType | emax |
      | :------: | :--: |
      | FLOAT4_E2M1 | 2 |
      | FLOAT4_E1M2 | 0 |
      | FLOAT8_E4M3FN | 8 |
      | FLOAT8_E5M2 | 15 |

  - 场景2，当scale_alg为1时，只涉及FP8类型（CuBALS Scale计算算法）：
    - 将长向量按块分，每块长度为block_size，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足block_size个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值：

      $$
      Amax(D_{fp32}^b)=max(\{|d_i|\}_{i=1}^{k})
      $$

    - 当max_low_bound大于0时，对最大绝对值进行下界钳位：

      $$
      Amax(D_{fp32}^b)=max(Amax(D_{fp32}^b), max\_low\_bound)
      $$

    - 将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值：

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

    - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$，并提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
      $$

    - 计算块缩放因子$S_{ue8m0}^b=2^{E_{int}^b}$和块转换因子$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$，对每个块内元素应用$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$。最终输出的量化结果是$(S^b, [d^i]_{i=1}^k)$，其中$S^b$代表块的缩放因子，即$S_{ue8m0}^b$。

  - 场景3，当scale_alg为2时，只涉及FP4_E2M1类型：
    - 当dst_type_max为0.0、6.0或7.0时：
      - 将输入x在axis维度上按k = block_size个数分组，一组k个数$\{V_i\}_{i=1}^{k}$动态量化为$\{mxscale, \{P_i\}_{i=1}^{k}\}$：

        $$
        shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果} 尾数位的高比特前一/两位 \text{为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases} \\
        P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space block\_size
        $$

      - 量化后的$P_i$按对应的$V_i$的位置组成输出y，mxscale按对应的axis维度上的分组组成输出mxscale。
    - 当dst_type_max不为0.0、6.0或7.0时：与场景2流程相同，其中当dst_type_max为0时，$Amax(DType)$是目标精度能表示的最大值；当dst_type_max不为0时，$Amax(DType)$是dst_type_max传入值。

> [!NOTE]
>
> - 本接口为确定性计算。
> - 本接口在CANN算子库中优先调用aclnnDynamicMxQuantV3内核；当运行环境不支持V3内核时，自动回退到aclnnDynamicMxQuantV2内核，此时`max_low_bound`参数不生效、`scale_alg`不支持取值2。

## 函数原型

```python
torch_npu.npu_dynamic_mx_quant(input, *, axis=-1, round_mode="rint", dst_type=296, block_size=32, scale_alg=0, dst_type_max=0.0, max_low_bound=0.0) -> (Tensor, Tensor)
```

## 参数说明

- **input**（`Tensor`）：必选参数，表示待量化的数据，对应公式中的$V_i$和$d_i$。支持1-7维。数据格式支持ND，支持非连续Tensor。数据类型支持`float16`、`bfloat16`、`float32`。
  - 当dst_type为FP4类时，input的最后一维必须是偶数。
  - 当input的数据类型为`float32`时，block_size必须为32，且量化轴的维度不能小于32。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **axis**（`int`）：可选参数，表示量化发生的轴，对应公式中的axis，默认值为`-1`。取值范围为[-D, D-1]，D为input的维数。

- **round_mode**（`str`）：可选参数，表示数据转换的模式，对应公式中的round_mode，默认值为`"rint"`。
  - 当dst_type为FP4类时，支持`"rint"`、`"floor"`、`"round"`。
  - 当dst_type为FP8类时，仅支持`"rint"`。

- **dst_type**（`ScalarType`）：可选参数，表示指定数据转换后y的数据类型，默认值为`torch_npu.float4_e2m1fn_x2`（296）。支持取值：

  | 取值 | 数据类型 |
  | --- | --- |
  | `torch_npu.float8_e5m2`（291） | FLOAT8_E5M2 |
  | `torch_npu.float8_e4m3fn`（292） | FLOAT8_E4M3FN |
  | `torch_npu.float4_e2m1fn_x2`（296） | FLOAT4_E2M1 |
  | `torch_npu.float4_e1m2fn_x2`（297） | FLOAT4_E1M2 |

- **block_size**（`int`）：可选参数，表示指定每次量化的元素个数，对应公式中的block_size，默认值为`32`。仅支持32的倍数，不能为0，且不能超过1024。当scale_alg为2时，block_size必须为32；当input的数据类型为`float32`时，block_size必须为32。

- **scale_alg**（`int`）：可选参数，表示mxscale的计算方法，对应公式中的scale_alg，默认值为`0`。
  - 取值为0代表场景1，为1代表场景2，为2代表场景3。
  - 当dst_type为`torch_npu.float4_e1m2fn_x2`时仅支持取值为0；当dst_type为`torch_npu.float4_e2m1fn_x2`时仅支持取值为0和2；当dst_type为FP8类时仅支持取值为0和1。

- **dst_type_max**（`float`）：可选参数，表示Amax(DType)的取值，对应公式中的Amax(DType)，默认值为`0.0`。
  - 支持取值0.0和6.0-12.0。取值为0.0代表Amax(DType)为量化结果数据类型的最大值；取值为6.0-12.0代表Amax(DType)为传入值。
  - 仅支持在dst_type为`torch_npu.float4_e2m1fn_x2`且block_size为32时设置。

- **max_low_bound**（`float`）：可选参数，表示每个block计算出的最大绝对值的下界钳位值，对应公式中的max_low_bound，默认值为`0.0`。
  - 仅当scale_alg为1时生效，scale_alg不为1时必须为0.0。
  - 取值为非负数。当max_low_bound大于0时，每个block的最大绝对值将与max_low_bound取较大值，即$Amax = max(Amax, max\_low\_bound)$。

## 返回值说明

- **y**（`Tensor`）：表示输入x量化后的对应结果，对应公式中的$P_i$和$d_i$。shape和输入input一致。数据类型由dst_type指定：dst_type为FP8类时，数据类型支持`float8_e4m3fn`、`float8_e5m2`；dst_type为FP4类时，实际返回的数据类型为uint8，且y的最后一维为input最后一维的一半（两个FP4元素打包为一个uint8），查看具体值需自行解包。数据格式支持ND，不支持非连续Tensor。
- **mxscale**（`Tensor`）：表示每个分组对应的量化尺度，对应公式中的mxscale和$S^b$。数据类型支持`float8_e8m0fnu`，实际返回的数据类型为uint8，查看具体值需自行转换。shape的秩为input的秩加1，最后一维固定为2；量化轴axis上的值为$ceil(input.shape[axis] / block\_size)$进行偶数对齐后除以2；当axis为非尾轴时，mxscale需要对每两行数据进行交织处理；其他维度与input一致。数据格式支持ND，不支持非连续Tensor。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 该接口为确定性计算。
- 关于input、mxscale的shape约束关系：
  - rank(mxscale) = rank(input) + 1
  - axis_change = axis if axis >= 0 else axis + rank(input)
  - mxscale.shape[axis_change] = (ceil(input.shape[axis] / block_size) + 2 - 1) / 2，即向上取整后偶数对齐再除以2
  - mxscale.shape[-1] = 2
  - 其他维度与输入input一致
- 当input的数据类型为`float32`时：block_size必须为32，量化轴的维度不能小于32。
- max_low_bound约束：
  - 仅当scale_alg为1时生效，scale_alg不为1时max_low_bound必须为0.0。
  - max_low_bound必须为非负数。
  - 当max_low_bound大于0时，每个block计算出的最大绝对值将与max_low_bound取较大值后再用于scale计算。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> input = torch.randn((1, 512), dtype=torch.bfloat16, device='npu')
    >>> y, mxscale = torch_npu.npu_dynamic_mx_quant(input)
    >>> y.shape
    torch.Size([1, 256])
    >>> y.dtype
    torch.uint8
    >>> mxscale.shape
    torch.Size([1, 8, 2])
    >>> mxscale.dtype
    torch.uint8
    ```

    指定FP8类目的数据类型进行量化：

    ```python
    >>> y, mxscale = torch_npu.npu_dynamic_mx_quant(input, dst_type=torch.float8_e4m3fn, scale_alg=1)
    >>> y.shape, y.dtype
    (torch.Size([1, 512]), torch.float8_e4m3fn)
    >>> mxscale.shape
    torch.Size([1, 8, 2])
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class DynamicMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, axis=-1, round_mode="rint", dst_type=296, block_size=32,
                    scale_alg=0, dst_type_max=0.0, max_low_bound=0.0):
            return torch_npu.npu_dynamic_mx_quant(x, axis=axis, round_mode=round_mode, dst_type=dst_type,
                                                  block_size=block_size, scale_alg=scale_alg,
                                                  dst_type_max=dst_type_max, max_low_bound=max_low_bound)
    def dynamic_mx_quant_test():
        input = torch.randn((1, 512), dtype=torch.bfloat16).npu()
        model = DynamicMxQuantModel()
        model.to('npu')

        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)

        y, mxscale = model(input)
    if __name__ == "__main__":
        dynamic_mx_quant_test()
    ```
