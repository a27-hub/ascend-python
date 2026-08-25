# torch_npu.npu_quantize

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>            |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    √     |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

- API功能：对输入的张量进行量化处理。
- 计算公式：
    - 若`div_mode`为`True`：
        $$
        result=(input/scales)+zero\_points
        $$

    - 若`div_mode`为`False`：

        $$
        result=(input*scales)+zero\_points
        $$

## 函数原型

```python
torch_npu.npu_quantize(self, scales, zero_points, dtype, axis=1, div_mode=True) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，需要进行量化的源数据张量，对应公式中的`self`。数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。`div_mode`为`False`且`dtype`为`quint4x2`时，最后一维需要能被8整除。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float`、`float16`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float`、`float16`、`bfloat16`。`div_mode`为`False`时，且当数据类型为`float`时，数据格式支持$NZ$。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`float`、`float16`、`bfloat16`。

- **scales** (`Tensor`)：必选参数，对`self`进行缩放的张量，对应公式中的`scales`。支持空Tensor，支持非连续的Tensor。
    - `div_mode`为`True`时：
        - <term>Atlas 推理系列产品</term>：数据类型支持`float`。
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float`、`bfloat16`。

    - `div_mode`为`False`时，数据格式支持$ND$。支持1维或多维（1维时，对应轴的大小需要与`self`中第`axis`维相等或等于1；多维时，`scales`的shape需要与`self`的shape维度相等，除`axis`指定的维度，其他维度为1，`axis`指定的维度必须和`self`对应的维度相等或等于1）。数据类型、数据格式需要和`self`的数据类型和数据格式一致。
        - <term>Atlas 推理系列产品</term>：数据类型支持`float`、`float16`。
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float`、`float16`、`bfloat16`。当数据格式为$NZ$时，`scales`的所有元素值为1。

- **zero_points** (`Tensor`)：必选参数，允许为None，对`self`进行偏移的张量，对应公式中的`zero_points`。支持空Tensor，支持非连续的Tensor。
    - `div_mode`为`True`时：
        - <term>Atlas 推理系列产品</term>：数据类型支持`int8`、`uint8`、`int32`。
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`、`uint8`、`int32`、`bfloat16`。

    - `div_mode`为`False`时，数据格式支持$ND$。支持1维或多维（1维时，对应轴的大小需要与`self`中第`axis`维相等或等于1；多维时，`scales`的shape需要与`self`维度相等，除`axis`指定的维度，其他维度为1，`axis`指定的维度必须和`self`对应的维度相等）。`zero_points`的shape和dtype需要和`scales`一致。
        - <term>Atlas 推理系列产品</term>：数据类型支持`float`、`float16`。
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float`、`float16`、`bfloat16`。当`self`的数据格式为$NZ$时，值为空。

- **dtype** (`int`)：必选参数，指定输出参数的类型。
    - `div_mode`为`True`时：
        - <term>Atlas 推理系列产品</term>：类型支持`qint8`、`quint8`、`int32`。
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：类型支持`qint8`、`quint8`、`int32`。
        - <term>Ascend 950PR/Ascend 950DT</term>：类型支持`qint8`、`quint8`、`int32`、`hifloat8`、`float8_e4m3fn`、`float8_e5m2`。

    - `div_mode`为`False`时，类型支持`qint8`、`quint4x2`。当`dtype`为`quint4x2`时，输出tensor类型为`int32`，由8个`int4`拼接。

- **axis** (`int`)：可选参数，量化的element-wise轴，其他的轴做broadcast，默认值为`1`。

    `div_mode` 为`False`时，`axis`取值范围是[-2, +∞)且指定的轴不能超过输入`self`的维度数。如果`axis`为-2，代表量化的element-wise轴是输入`self`的倒数第二根轴；如果`axis`大于-2，量化的element-wise轴是输入的最后一根轴。

- **div_mode** (`bool`)：可选参数，表示计算`scales`模式，对应公式中的`div_mode`。当`div_mode`为`True`时，表示用除法计算`scales`；`div_mode`为`False`时，表示用乘法计算`scales`，默认值为`True`。

## 返回值说明

`Tensor`

对应公式中的`result`。数据类型由参数`dtype`指定，如果参数`dtype`为`quint4x2`，输出的`dtype`是`int32`，shape的最后一维是`self`的shape最后一维的1/8，shape其他维度和`self`的shape其他维度保持一致；如果参数`dtype`不为`quint4x2`时，shape与输入`self`的shape保持一致。输出的数据格式与输入`self`的数据格式保持一致，且当数据格式为$NZ$时，数据类型仅支持int32。支持空Tensor，支持非连续的Tensor。
- <term>Ascend 950PR/Ascend 950DT</term>：当`dtype`为`hifloat8`时，单算子模式下输出`y`的数据类型为`uint8`（实际承载`hifloat8`类型），图模式下为`torch.bits8`（实际承载`hifloat8`类型）。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- `self`数据格式为$NZ$时，`self`输入shape支持3维，形如\(e，k，n\)，k为256的倍数，n为8的倍数，`scales`输入shape支持1维或3维，`zero_points`输入为None，`dtype`为`quint4x2`。
- `div_mode`为`False`时：
    - 支持<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>。
    - 当`dtype`为`quint4x2`或者`axis`为-2时，不支持<term>Atlas 推理系列产品</term>。

## 调用示例

- 单算子模式调用
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>

        ```python
        import torch
        import torch_npu
        x = torch.randn(1, 1, 12).bfloat16().npu()
        scale = torch.tensor([0.1] * 12).bfloat16().npu()
        out = torch_npu.npu_quantize(x, scale, None, torch.qint8, -1, False)
        print(x)
            dtype=torch.bfloat16)
        print(out)
            dtype=torch.int8)
        ```

    - <term>Atlas 推理系列产品</term>

        ```python
        import torch
        import torch_npu
        x = torch.randn((2, 3, 12), dtype=torch.float).npu()
        scale = torch.tensor(([3] * 12),dtype=torch.float).npu()
        out = torch_npu.npu_quantize(x, scale, None, torch.qint8, -1, False)
        print(x)

        print(out)

            dtype=torch.int8)
        ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    x = torch.randn((2, 3, 12), dtype=torch.float16).npu()
    scale = torch.tensor(([3] * 12), dtype=torch.float16).npu()
    axis = 1

    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, x, scale, zero_points, dst_type, div_mode):
            return torch_npu.npu_quantize(x, scale, zero_points=zero_points, dtype=dst_type, div_mode=div_mode)

    model = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    config.debug.graph_dump.type = 'pbtxt'
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
    output_data = model(x, scale, None, dst_type=torch.qint8, div_mode=div_mode)
    print("shape of x:", x.shape)
    print("shape of output_data:", output_data.shape)
    print("x:", x)
    print("output_data:", output_data)

    output_data: tensor([[[ 1,  2,  2,  3,  2,  2,  5, -1,  2, -3, -2, -2],

        dtype=torch.int8)

    ```
