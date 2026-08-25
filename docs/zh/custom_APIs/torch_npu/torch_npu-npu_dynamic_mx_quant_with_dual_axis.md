# torch\_npu.npu\_dynamic\_mx\_quant\_with\_dual\_axis

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- API功能：在-1轴和-2轴上同时进行目的数据类型为FLOAT4类、FLOAT8类的MX量化。在给定的-1轴和-2轴上，每32个数，计算出这两组数对应的量化尺度`mxscale1`、`mxscale2`，然后分别对两组数所有元素除以对应的`mxscale1`或`mxscale2`，根据`round_mode`转换到对应的`dst_type`，得到量化结果`y1`和`y2`。

- 合轴说明：算子实现时，会对-2轴（不包含）之前的所有轴进行合轴处理。即对于输入shape为$(d_0, d_1, ..., d_{n-3}, d_{n-2}, d_{n-1})$的张量，-2轴之前的维度$(d_0, d_1, ..., d_{n-3})$会被合并为一个维度，等效于将输入reshape为$(d_0 \times d_1 \times ... \times d_{n-3}, d_{n-2}, d_{n-1})$后再进行量化计算。

- 计算公式：

  - 将输入x在-1轴上按照32个数进行分组，一组32个数 $\{\{V_i\}_{i=1}^{32}\}$ 量化为 $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale1 = 2^{shared\_exp} \\
    P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
    $$

  - 同时，将输入x在-2轴上按照32个数进行分组，一组32个数 $\{\{V_j\}_{j=1}^{32}\}$ 量化为 $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax \\
    mxscale2 = 2^{shared\_exp} \\
    P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
    $$

  - -1轴量化后的$P_i$按对应的$V_i$的位置组成输出y1，mxscale1按对应的-1轴维度上的分组组成输出mxscale1。-2轴量化后的$P_j$按对应的$V_j$的位置组成输出y2，mxscale2按对应的-2轴维度上的分组组成输出mxscale2。

  - emax：对应数据类型的最大正则数的指数位。

    | dst_type | emax |
    | :---: | :---: |
    | float4_e2m1fn_x2 | 2 |
    | float4_e1m2fn_x2 | 0 |
    | float8_e4m3fn | 8 |
    | float8_e5m2 | 15 |

## 函数原型

```python
torch_npu.npu_dynamic_mx_quant_with_dual_axis(input, *, round_mode="rint", dst_type=296, scale_alg=0, dst_type_max=0.0) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，待量化的输入张量，数据类型支持`float16`、`bfloat16`，shape支持2-7维度，支持非连续的Tensor，数据格式支持ND。输入input的维度必须大于等于2。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，input的最后一维必须是偶数。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **round_mode** (`str`)：可选参数，数据转换的模式。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，支持取值`"rint"`、`"floor"`、`"round"`；当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时，仅支持取值`"rint"`。默认值为`"rint"`。
- **dst_type** (`int`)：可选参数，指定量化后输出y1和y2的数据类型。支持取值291（`float8_e5m2`）、292（`float8_e4m3fn`）、296（`float4_e2m1fn_x2`）、297（`float4_e1m2fn_x2`），默认值为296。
- **scale_alg** (`int`)：可选参数，指定计算`mxscale`的算法，支持取值0和1。取值为0时采用基于shared_exp的MX量化算法；取值为1时仅适用于FLOAT8类型，采用基于FP32块缩放因子的算法。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时仅支持取值0。默认值为0。
- **dst_type_max** (`float`)：可选参数，指定目标数据类型的最大表示值，取值为0.0时使用目标精度能表示的最大值，默认值为0.0。

## 返回值说明

- **y1** (`Tensor`)：-1轴量化后的输出张量。当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时，y1的数据类型与`dst_type`对应，shape与输入x一致。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，y1的实际数据类型为uint8，shape的最后一维为x最后一维的一半。
- **mxscale1** (`Tensor`)：-1轴每个分组对应的量化尺度，数据类型为`float8_e8m0fnu`，实际返回的数据类型为uint8。shape比输入x多一维，最后一维为2。
- **y2** (`Tensor`)：-2轴量化后的输出张量。数据类型和shape规则与y1一致。
- **mxscale2** (`Tensor`)：-2轴每个分组对应的量化尺度，数据类型为`float8_e8m0fnu`，实际返回的数据类型为uint8。shape比输入x多一维，最后一维为2，且输出需要对每两行数据进行交织处理。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输入x的维度必须大于等于2。
- 输入`input`和输出`mxscale1`、`mxscale2`的shape约束关系：
  - rank(mxscale1) = rank(x) + 1
  - rank(mxscale2) = rank(x) + 1
  - mxscale1.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2
  - mxscale2.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2
  - mxscale1.shape[-1] = 2
  - mxscale2.shape[-1] = 2
  - 其他维度与输入x一致
  - 举例：输入x的shape为[B, M, N]，对应的y1和y2的shape为[B, M, N]，mxscale1的shape为[B, M, (ceil(N/32)+2-1)/2, 2]，mxscale2的shape为[B, (ceil(M/32)+2-1)/2, N, 2]。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn((2, 32), dtype=torch.bfloat16).npu()
    >>> y1, mxscale1, y2, mxscale2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(x, round_mode="rint", dst_type=292, scale_alg=0)
    >>> print(y1)
    tensor([[  0,  18,  -2, ..., -16,   0,  24],
            [  0,  -8,  32, ...,   0, -16,   0]], device='npu:0',
           dtype=torch.float8_e4m3fn)
    >>> print(mxscale1)
    tensor([[[  0, 128]],
            [[  0, 128]]], device='npu:0', dtype=torch.uint8)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class DynamicMxQuantWithDualAxisModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return torch_npu.npu_dynamic_mx_quant_with_dual_axis(x, round_mode="rint", dst_type=292, scale_alg=0)

    x = torch.randn((2, 32), dtype=torch.bfloat16).npu()
    model = DynamicMxQuantWithDualAxisModel()
    model.to('npu')

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)

    y1, mxscale1, y2, mxscale2 = model(x)
    print(y1)
    print(mxscale1)
    ```
