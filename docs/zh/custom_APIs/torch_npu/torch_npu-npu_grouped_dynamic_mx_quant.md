# torch\_npu.npu\_grouped\_dynamic\_mx\_quant

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- API功能：根据传入的分组索引起始值（`group_index`），对输入的各个分组以基本块（`blocksize`）的粒度进行目标数据类型为float4/float8的动态MX量化。

- 计算公式：

  - 将输入x在第0维上先按照`group_index`进行分组，每个group内按k = blocksize个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 计算出这组数对应的量化尺度mxscale_pre。

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale\_pre = 2^{shared\_exp}
    $$

  - 这组数每一个除以mxscale_pre，根据`round_mode`转换到对应的`dst_type`，得到量化结果y。

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale\_pre, round\_mode), \space i\space from\space 1\space to\space blocksize
    $$

  - 量化后的$P_i$按对应的$V_i$的位置组成输出y，mxscale_pre按对应的`group_index`分组，分组内第一个维度pad为偶数，组成输出mxscale。

  - emax：对应数据类型的最大正则数的指数位。

    | dst_type | emax |
    | :---: | :---: |
    | float4_e2m1fn_x2 | 2 |
    | float4_e1m2fn_x2 | 0 |
    | float8_e4m3fn | 8 |
    | float8_e5m2 | 15 |

## 函数原型

```python
torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, *, round_mode="rint", dst_type=23, blocksize=32, scale_alg=0, dst_type_max=0.0) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，待量化的输入张量，数据类型支持`float16`、`bfloat16`，shape仅支持2维，支持非连续的Tensor，数据格式支持ND。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，x的最后一维必须是偶数。
- **group_index** (`Tensor`)：必选参数，量化分组的起始索引，数据类型支持`int32`，shape仅支持1维，数据格式支持ND。索引要求大于等于0，且非递减，并且最后一个数需要与x的第一个维度大小相等。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **round_mode** (`str`)：可选参数，数据转换的模式，仅支持取值`"rint"`，默认值为`"rint"`。
- **dst_type** (`int`)：可选参数，指定量化后输出y的数据类型。支持取值23（`float8_e5m2`）、24（`float8_e4m3fn`）、296（`float4_e2m1fn_x2`）、297（`float4_e1m2fn_x2`），默认值为23。
- **blocksize** (`int`)：可选参数，指定每次量化的元素个数，仅支持取值32，默认值为32。
- **scale_alg** (`int`)：可选参数，指定计算`mxscale`的算法，支持取值0、1、2。取值为0时采用基于shared_exp的MX量化算法；取值为1时仅适用于FLOAT8类型，采用基于FP32块缩放因子的算法；取值为2时仅适用于`float4_e2m1fn_x2`类型。默认值为0。
- **dst_type_max** (`float`)：可选参数，指定目标数据类型的最大表示值，取值为0.0时使用目标精度能表示的最大值，默认值为0.0。

## 返回值说明

- **y** (`Tensor`)：量化后的输出张量。当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时，y的数据类型与`dst_type`对应，shape与输入x一致。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，y的实际数据类型为uint8，shape的最后一维为x最后一维的一半。
- **mxscale** (`Tensor`)：每个分组对应的量化尺度，数据类型为`float8_e8m0fnu`，实际返回的数据类型为uint8。假设x的shape为[m, n]，group_index的shape为[g]，则mxscale的shape为[m / (blocksize * 2) + g, n, 2]。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输入x的维度必须为2，group_index的维度必须为1。
- group_index的元素要求大于等于0，且非递减，并且最后一个元素需要与x的第一个维度大小相等。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn((8, 32), dtype=torch.bfloat16).npu()
    >>> group_index = torch.tensor([4, 8], dtype=torch.int32).npu()
    >>> y, mxscale = torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, round_mode="rint", dst_type=24, blocksize=32)
    >>> print(y)
    tensor([[  0,  18,  -2, ..., -16,   0,  24],
            [  0,  -8,  32, ...,   0, -16,   0],
            ...,
            [  0,  24,  16, ...,  -8,  32,   0],
            [  0, -16,   0, ...,  24,   0,  18]], device='npu:0',
           dtype=torch.float8_e4m3fn)
    >>> print(mxscale)
    tensor([[[  0, 128],
             [  0, 128],
             ...,
             [  0, 128],
             [  0, 128]]], device='npu:0', dtype=torch.uint8)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class GroupedDynamicMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, group_index):
            return torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, round_mode="rint", dst_type=24, blocksize=32)

    x = torch.randn((8, 32), dtype=torch.bfloat16).npu()
    group_index = torch.tensor([4, 8], dtype=torch.int32).npu()
    model = GroupedDynamicMxQuantModel()
    model.to('npu')

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)

    y, mxscale = model(x, group_index)
    print(y)
    print(mxscale)
    ```
