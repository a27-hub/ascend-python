# 文档结构模板

## 完整模板

```markdown
# torch_npu.npu_{operator_name}

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |

## 功能说明

- API功能：{一句话描述算子功能}
- 计算公式：（无公式时省略此项）

  $$
  {LaTeX 公式}
  $$

> [!NOTE]
> {重要使用说明或性能注意事项}（无注意事项时省略此块）

## 函数原型

​```
torch_npu.npu_{operator_name}(required_param1, required_param2, *, optional_param1=None, optional_param2=0) -> Tensor
​```

## 参数说明

- **required_param1** (`Tensor`)：必选参数，{描述}，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，支持非连续的Tensor。
- **required_param2** (`Tensor`)：必选参数，{描述}，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$。
- <strong>*</strong>：位置参数与关键字参数的分隔符。其之前的参数为位置参数，需按顺序传入；其之后的参数为关键字参数，需通过键值对方式传入，未赋值时使用默认值。
- **optional_param1** (`Tensor`)：可选参数，{描述}，默认值为`None`。
- **optional_param2** (`int`)：可选参数，{描述}，默认值为`0`。

## 返回值说明

`Tensor`

{描述输出的 shape、数据类型和格式}

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式。
- {其他约束条件}

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn((2, 4), dtype=torch.float16, device='npu')
    >>> result = torch_npu.npu_{operator_name}(x)
    >>> result
    tensor([[...]], device='npu:0', dtype=torch.float16)
    >>> result.shape
    torch.Size([2, 4])
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch_npu.npu_{operator_name}(x)

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    output = model(x)
    print(output)
    ```
```

## 产品支持表可选行

按需选用，仅列出实际支持的产品：

```
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
|<term>Atlas 推理系列产品</term>                               |    √     |
|<term>Atlas 训练系列产品</term>                               |    √     |
```

## 参数说明特殊情况

**不同产品支持不同数据类型时**，按产品分条列出：

```markdown
- **x** (`Tensor`)：必选参数，{描述}，数据格式支持$ND$。
  - <term>Atlas 推理系列产品</term>：数据类型支持`int8`。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`、`int32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`、`int32`。
```

**多返回值时**，在"返回值说明"中逐一描述：

```markdown
## 返回值说明

- **yOut** (`Tensor`)：{描述}，数据类型和shape与输入`x`保持一致。
- **rstdOut** (`Tensor`)：{描述}，数据类型支持`float32`。
- **xOut** (`Tensor`)：{描述}，shape与输入`x`保持一致。
```
