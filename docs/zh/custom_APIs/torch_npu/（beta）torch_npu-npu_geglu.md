# （beta）torch\_npu.npu\_geglu

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- API功能：高斯误差线性单元激活函数。

- 计算公式：

  若`activate_left`为`True`，表示对`self`的左半部分做激活：

  $$
  out = GeGlu(self) = Gelu(A) \cdot B
  $$

  若`activate_left`为`False`，表示对`self`的右半部分做激活：

  $$
  out = GeGlu(self) = A \cdot Gelu(B)
  $$

  其中，$A$表示`self`的左半部分，$B$表示`self`的右半部分。

  - 当`approximate`为0（`none`）时：

    $$
    Gelu(x) = 0.5 * x * [1 + erf(x/\sqrt{2})]
    $$

  - 当`approximate`为1（`tanh`）时：

    $$
    Gelu(x) = 0.5 * x * (1 + Tanh(\sqrt{2 / \pi} * (x + 0.044715 * x^{3})))
    $$

## 函数原型

```python
torch_npu.npu_geglu(self, dim=-1, approximate=1, activate_left=False) -> (Tensor, Tensor)
```

## 参数说明

- **self** (`Tensor`)：必选参数，待进行GeGlu计算的输入张量，对应公式中的`self`。shape支持1-8维，且在`dim`指定的维度上必须能被2整除。支持非连续的Tensor，支持空Tensor，数据格式支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持`float32`、`float16`。
    - <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float32`、`float16`、`bfloat16`。
- **dim** (`int`)：可选参数，设定的切分轴，对`self`对应的轴进行对半切，取值范围为[-D, D-1]，D为`self`的维数，默认值为-1。
- **approximate** (`int`)：可选参数，GeGlu计算使用的Gelu激活函数模式，0表示使用`none`（基于erf），1表示使用`tanh`，默认值为1。
- **activate_left** (`bool`)：可选参数，表示激活函数操作数据块的方向。`False`表示对右半部分做激活，`True`表示对左半部分做激活，默认值为`False`。

## 返回值说明

- **out** (`Tensor`)：GeGlu计算的输出结果。shape与`self`的shape除了`dim`指定的轴外保持一致，`dim`指定的轴为`self`对应轴的一半。数据类型与`self`一致。
- **out_gelu** (`Tensor`)：Gelu激活的中间输出结果。shape和数据类型与`out`一致。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- `self`在`dim`指定的维度上必须能被2整除。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float16).npu()
    >>> out, out_gelu = torch_npu.npu_geglu(x, dim=-1, approximate=1, activate_left=False)
    >>> print(out)
    tensor([[0.0000, 1.6107]], device='npu:0', dtype=torch.float16)
    >>> print(out_gelu)
    tensor([[0.5000, 0.9540]], device='npu:0', dtype=torch.float16)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class GeGluModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return torch_npu.npu_geglu(x, dim=-1, approximate=1, activate_left=False)

    x = torch.randn((2, 32), dtype=torch.float16).npu()
    model = GeGluModel()
    model.to('npu')

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)

    out, out_gelu = model(x)
    print(out)
    print(out_gelu)
    ```
