# torch_npu.npu_gelu_mul

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |

## 功能说明

- API功能：将输入Tensor按最后一维分成左右两半，对左半部分做GELU激活后与右半部分逐元素相乘。
- 计算公式：

  给定输入张量`input`，最后一维的长度为`2d`，`npu_gelu_mul`进行以下计算：

  1. 将`input`分割为两部分：
     $$
     x_1 = \text{input}[..., :d], \quad x_2 = \text{input}[..., d:]
     $$
  2. 对$x_1$应用GELU激活函数，`approximate="none"`时使用erf模式：
     $$
     \text{GELU}(x_1) = 0.5 \cdot x_1 \left( 1 + \text{erf}\left( \frac{x_1}{\sqrt{2}} \right) \right)
     $$
     `approximate="tanh"`时使用tanh模式：
     $$
     \text{GELU}(x_1) = 0.5 \cdot x_1 \cdot \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \cdot \left( x_1 + 0.044715 x_1^3 \right) \right) \right)
     $$
  3. 最终输出为$\text{GELU}(x_1)$与$x_2$的逐元素乘积：
     $$
     \text{out} = \text{GELU}(x_1) \times x_2
     $$

> [!NOTE]
> 典型场景尾轴为16的倍数，当尾轴为非32B对齐时，建议走小算子拼接逻辑。

## 函数原型

```
torch_npu.npu_gelu_mul(input, *, approximate='none') -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选参数，对应公式中的`input`，待进行`npu_gelu_mul`计算的入参，数据格式支持$ND$，支持非连续的Tensor，支持空Tensor。维度支持2-8维，最后一维的值必须为偶数且小于等于1024，其他维度的乘积须小于等于200000。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`bfloat16`、`float16`、`float32`。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`bfloat16`、`float16`、`float32`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`bfloat16`、`float16`、`float32`。
- <strong>*</strong>：位置参数与关键字参数的分隔符。其之前的参数为位置参数，需按顺序传入；其之后的参数为关键字参数，需通过键值对方式传入，未赋值时使用默认值。
- **approximate** (`str`)：可选参数，GELU计算的模式，可配置为`none`或者`tanh`。其中`none`代表使用erf模式，`tanh`代表使用tanh模式。默认值为`none`。

## 返回值说明

`Tensor`

数据类型与`input`保持一致，数据格式支持$ND$，shape的最后一维为`input`最后一维的二分之一，其他维度与`input`保持一致，支持非连续的Tensor。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式。
- `input`输入不能为None。
- `input`最后一维的值必须为偶数且小于等于1024，其他维度的乘积须小于等于200000。
- `approximate`仅支持`none`和`tanh`两种取值。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> input_tensor = torch.randn(2, 4, dtype=torch.float32, device='npu')
    >>> output_tensor = torch_npu.npu_gelu_mul(input_tensor, approximate='tanh')
    >>> print(output_tensor)
    tensor([[...]], device='npu:0')
    >>> print(output_tensor.shape)
    torch.Size([2, 2])
    >>> print(output_tensor.dtype)
    torch.float32
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch_npu.npu.set_compile_mode(jit_compile=True)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return torch_npu.npu_gelu_mul(x, approximate="none")

    x = torch.randn(2, 4, dtype=torch.float32, device='npu')
    model = Net()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    npu_out = model(x)
    print(npu_out.shape, npu_out.dtype)

    # 执行上述代码的输出类似如下
    torch.Size([2, 2]) torch.float32
    ```
