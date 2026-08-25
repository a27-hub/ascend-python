# torch\_npu.npu\_dynamic\_mx\_quant

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- API功能：目的数据类型为FLOAT4类、FLOAT8类的动态对称MX量化。在给定的轴`axis`上，根据每`block_size`个数，计算出这组数对应的量化尺度`mxscale`，然后对这组数每一个除以`mxscale`，根据`round_mode`转换到对应的`dst_type`，得到量化结果`y`。

- 计算公式：

  - 将输入x在axis维度上按k = block_size个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = block_size
  $$
  shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
  mxscale = 2^{shared\_exp}\\
  P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space block\_size\\
  $$

  量化后的$P_i$按对应的$V_i$的位置组成输出y，mxscale按对应的axis维度上的分组组成输出mxscale。

  - emax：对应数据类型的最大正则数的指数位。

    | dst_type | emax |
    | :---: | :---: |
    | float4_e2m1fn_x2 | 2 |
    | float4_e1m2fn_x2 | 0 |
    | float8_e4m3fn | 8 |
    | float8_e5m2 | 15 |

## 函数原型

```python
torch_npu.npu_dynamic_mx_quant(input, *, axis=-1, round_mode="rint", dst_type=296, block_size=32, scale_alg=0, dst_type_max=0.0, max_low_bound=0.0) -> (Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，待量化的输入张量，数据类型支持`float16`、`bfloat16`，shape支持1-7维度，支持非连续的Tensor，数据格式支持ND。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，input的最后一维必须是偶数。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **axis** (`int`)：可选参数，量化发生的轴，取值范围为[-D, D-1]，D为x的维数，默认值为-1。
- **round_mode** (`str`)：可选参数，数据转换的模式。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，支持取值`"rint"`、`"floor"`、`"round"`；当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时，仅支持取值`"rint"`。默认值为`"rint"`。
- **dst_type** (`int`)：可选参数，指定量化后输出y的数据类型。支持取值291（`float8_e5m2`）、292（`float8_e4m3fn`）、296（`float4_e2m1fn_x2`）、297（`float4_e1m2fn_x2`），默认值为296。
- **block_size** (`int`)：可选参数，指定每次量化的元素个数，必须是32的倍数，取值范围为(0, 1024]，默认值为32。
- **scale_alg** (`int`)：可选参数，指定计算`mxscale`的算法，支持取值0、1、2。取值为0时采用基于shared_exp的MX量化算法；取值为1时仅适用于FLOAT8类型，采用基于FP32块缩放因子的算法；取值为2时仅适用于`float4_e2m1fn_x2`类型，采用改进的shared_exp算法。默认值为0。
- **dst_type_max** (`float`)：可选参数，指定目标数据类型的最大表示值，取值为0.0时使用目标精度能表示的最大值，默认值为0.0。
- **max_low_bound** (`float`)：可选参数，每个block计算出的最大绝对值的下界钳位值，取值大于0时生效，默认值为0.0。

## 返回值说明

- **y** (`Tensor`)：量化后的输出张量。当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时，y的数据类型与`dst_type`对应，shape与输入x一致。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，y的实际数据类型为uint8，shape的最后一维为x最后一维的一半（每两个FP4数据打包为一个uint8），查看具体值需自行解包。
- **mxscale** (`Tensor`)：每个分组对应的量化尺度，数据类型为`float8_e8m0fnu`，实际返回的数据类型为uint8。shape比输入x多一维，最后一维为2。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输入`x`和输出`mxscale`的shape约束关系：
  - rank(mxscale) = rank(x) + 1
  - axis_change = axis if axis >= 0 else axis + rank(x)
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / block_size) + 2 - 1) / 2
  - mxscale.shape[-1] = 2
  - 其他维度与输入x一致

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn((1, 32), dtype=torch.bfloat16).npu()
    >>> y, mxscale = torch_npu.npu_dynamic_mx_quant(x, axis=-1, round_mode="rint", dst_type=292, block_size=32)
    >>> print(y)
    tensor([[  0,  -2,  18, ..., -16,  -8,   0]], device='npu:0',
           dtype=torch.float8_e4m3fn)
    >>> print(mxscale)
    tensor([[[  0, 128]]], device='npu:0', dtype=torch.uint8)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class DynamicMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return torch_npu.npu_dynamic_mx_quant(x, axis=-1, round_mode="rint", dst_type=292, block_size=32)

    x = torch.randn((1, 32), dtype=torch.bfloat16).npu()
    model = DynamicMxQuantModel()
    model.to('npu')

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)

    y, mxscale = model(x)
    print(y)
    print(mxscale)
    ```
