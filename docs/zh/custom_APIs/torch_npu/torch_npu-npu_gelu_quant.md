# torch\_npu.npu\_gelu\_quant

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- API功能：对张量进行GELU（Gaussian Error Linear Unit，高斯误差线性单元）激活操作，再对结果进行静态/动态量化。

- 计算公式：

  1. 先计算Gelu得到geluOut

     - approximate = "tanh"

       $$
       geluOut = 0.5 * self * (1 + Tanh(\sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
       $$

     - approximate = "none"

       $$
       geluOut = 0.5 * self * [1 + erf(self/\sqrt{2})]
       $$

  2. 再对geluOut进行量化操作

     - quant_mode = "static"

       $$
       y = round\_to\_dstType(geluOut * input\_scale + input\_offset, round\_mode)
       $$

     - quant_mode = "dynamic"

       $$
       geluOut = geluOut * input\_scale \\
       Max = max(abs(geluOut)) \\
       out\_scale = Max / maxValue \\
       y = round\_to\_dstType(geluOut / out\_scale, round\_mode)
       $$

  - maxValue：对应数据类型的最大值。

    | 数据类型 | maxValue |
    | :---: | :---: |
    | int8 | 127 |
    | float8_e4m3fn | 448 |
    | float8_e5m2 | 57344 |
    | hifloat8 | 32768 |

## 函数原型

```python
torch_npu.npu_gelu_quant(self, *, input_scale=None, input_offset=None, approximate="none", quant_mode="dynamic", dst_type=None, round_mode="rint") -> (Tensor, Tensor)
```

## 参数说明

- **self** (`Tensor`)：必选参数，待计算的输入张量，数据类型支持`float16`、`bfloat16`、`float32`，`quant_mode`为`"dynamic"`时shape支持2-8维度，`quant_mode`为`"static"`时shape支持1-8维度，支持非连续的Tensor，数据格式支持$ND$。不支持空Tensor。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **input_scale** (`Tensor`)：可选参数，量化的输入scale，数据类型支持`float16`、`bfloat16`、`float32`，shape仅支持1维，大小只能是`self`的尾轴维度大小或1，支持非连续的Tensor，数据格式支持$ND$。当`quant_mode`为`"static"`时为必选输入，为`"dynamic"`时为可选输入。
- **input_offset** (`Tensor`)：可选参数，量化的输入offset，数据类型和shape与`input_scale`保持一致。当`quant_mode`为`"dynamic"`且`input_scale`不输入时，`input_offset`也不输入。
- **approximate** (`str`)：可选参数，Gelu激活函数的模式，支持取值`"none"`、`"tanh"`，默认值为`"none"`。
- **quant_mode** (`str`)：可选参数，量化的模式，支持取值`"static"`（静态量化）、`"dynamic"`（动态量化），默认值为`"dynamic"`。
- **round_mode** (`str`)：可选参数，数据转换的模式，默认值为`"rint"`。
  - 当`dst_type`为`int8`、`float8_e4m3fn`、`float8_e5m2`时，仅支持取值`"rint"`。
  - 当`dst_type`为`hifloat8`时，支持取值`"round"`、`"hybrid"`。
- **dst_type** (`int`)：可选参数，指定量化后输出y的数据类型。数据类型支持`int8`、`hifloat8`、`float8_e5m2`、`float8_e4m3fn`，默认值为`None`，表示`int8`。

## 返回值说明

- **y** (`Tensor`)：Gelu激活后量化输出的结果，数据类型由`dst_type`指定，shape与输入`self`一致。当`dst_type`为`hifloat8`时，`y`的数据类型为`uint8`（实际承载`hifloat8`类型）。
- **out_scale** (`Tensor`)：动态量化时计算出的量化尺度，数据类型为`float32`，shape为`self`的shape剔除最后一维。当`quant_mode`为`"static"`时，`out_scale`为空Tensor。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 当`dst_type`为`hifloat8`时，必须传入`round_mode`为`"round"`或`"hybrid"`，不支持默认值`"rint"`。
- `input_scale`的数据类型与`self`的类型一致，或者在类型不一致时采用精度更高的类型。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    x = torch.rand((4, 8), dtype=torch.float16).npu() * 10
    input_scale = torch.ones((8,), dtype=torch.float16).npu()
    # 静态量化
    y, out_scale = torch_npu.npu_gelu_quant(x, input_scale=input_scale, approximate="tanh", quant_mode="static", dst_type=1, round_mode="rint")
    print(y)
    print(out_scale)

    # 动态量化
    y, out_scale = torch_npu.npu_gelu_quant(x, approximate="tanh", quant_mode="dynamic", dst_type=1, round_mode="rint")
    print(y)
    print(out_scale)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class GeluQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, input_scale=None):
            return torch_npu.npu_gelu_quant(x, input_scale=input_scale, approximate="tanh", quant_mode="static", dst_type=1, round_mode="rint")

    x = torch.rand((4, 8), dtype=torch.float16).npu() * 10
    input_scale = torch.ones((8,), dtype=torch.float16).npu()
    model = GeluQuantModel()
    model.to('npu')

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)

    y, out_scale = model(x, input_scale)
    print(y)
    print(out_scale)
    ```
