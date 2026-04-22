# torch_npu.npu_quant_max

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |

## 功能说明

- API功能：对输入张量进行量化操作，输出量化后的张量及输入张量的绝对值最大值。

- 计算公式：
    $$
    \mathbf{y} = \text{quantize}(\mathbf{x} \times \text{scale}) \\
    \text{amax} = \max(|\mathbf{x}|)
    $$

    其中，quantize表示根据`round_mode`指定的舍入模式，将结果转换为`dst_dtype`指定的数据类型。

## 函数原型

```python
torch_npu.npu_quant_max(x, scale, *, round_mode="rint", dst_dtype=291) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，支持非连续的Tensor。
- **scale** (`Tensor`)：必选参数，缩放因子张量，数据类型支持`float32`，数据格式支持$ND$, 一维Tensor。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **round_mode** (`str`)：可选参数，舍入模式。支持取值：
    - `"rint"`：使用四舍五入到最近偶数的舍入方式。
    - `"round"`：使用四舍五入的舍入方式。
    - `"hybrid"`：混合舍入方式。
    默认值为`"rint"`。
- **dst_dtype** (`int`)：可选参数，指定量化输出的数据类型。支持取值：
    - `290`：输出类型为`torch_npu.hifloat8`。
    - `291`：输出类型为`torch.float8_e5m2`。
    - `292`：输出类型为`torch.float8_e4m3fn`。
    默认值为`291`。

## 返回值说明

- **y** (`Tensor`)：量化后的输出Tensor，数据类型由`dst_dtype`指定，形状与输入`x`一致。
- **amax** (`Tensor`)：输入`x`的绝对值最大值，一维Tensor，数据类型与输入`x`一致。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式。
- `x`、`scale`这两个输入中不能为None。

## 调用示例

- 单算子模式调用
    - 基本用法

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((4, 16), dtype=torch.float16).to("npu")
        >>> scale = torch.rand((16,), dtype=torch.float32).to("npu")
        >>>
        >>> # 使用默认参数：round_mode="rint", dst_dtype=291(float8_e5m2)
        >>> y, amax = torch_npu.npu_quant_max(x, scale)
        >>> y.shape
        torch.Size([4, 16])
        >>> y.dtype
        torch.float8_e5m2
        >>> amax
        tensor(0.9876, device='npu:0', dtype=torch.float16)
        ```

    - 指定舍入模式和输出类型

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((8, 32, 64), dtype=torch.bfloat16).to("npu")
        >>> scale = torch.rand((64,), dtype=torch.float32).to("npu")
        >>>
        >>> # 指定round_mode为"round"，dst_dtype为292(float8_e4m3fn)
        >>> y, amax = torch_npu.npu_quant_max(x, scale, round_mode="round", dst_dtype=292)
        >>> y.shape
        torch.Size([8, 32, 64])
        >>> y.dtype
        torch.float8_e4m3fn
        >>> amax
        tensor(0.9980, device='npu:0', dtype=torch.bfloat16)
        ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    device = torch.device(f'npu:0')
    torch_npu.npu.set_device(device)

    class QuantMaxModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, scale, round_mode="rint", dst_dtype=291):
            y, amax = torch_npu.npu_quant_max(x, scale, round_mode=round_mode, dst_dtype=dst_dtype)
            return y, amax

    x = torch.randn((2, 4, 16), device='npu', dtype=torch.float16)
    scale = torch.randn((16,), device='npu', dtype=torch.float32)
    quant_max_model = QuantMaxModel().npu()
    quant_max_model = torch.compile(quant_max_model, backend=npu_backend, dynamic=True)
    y, amax = quant_max_model(x, scale)
    print(y)
    print(amax)

    # 执行上述代码的输出类似如下
    tensor([[[0.9375, 0.8750, 1.0000, ..., 0.7500, 0.6250, 0.8750],
             [0.5000, 0.7500, 0.6250, ..., 0.8750, 1.0000, 0.5625],
             [0.8125, 0.9375, 0.7500, ..., 0.6250, 0.5000, 0.7500],
             [0.7500, 0.5625, 0.8750, ..., 0.9375, 0.7500, 0.6250]],

            [[0.6250, 0.8125, 0.7500, ..., 0.5000, 0.8750, 0.9375],
             [0.9375, 0.7500, 0.5625, ..., 0.8125, 0.6250, 0.7500],
             [0.7500, 0.8750, 0.9375, ..., 0.5625, 0.7500, 0.5000],
             [0.5625, 0.6250, 0.7500, ..., 0.7500, 0.8750, 0.8125]]], device='npu:0',
        dtype=torch.float8_e5m2)
    tensor(0.9844, device='npu:0', dtype=torch.float16)
    ```
    