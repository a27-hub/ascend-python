# torch_npu.npu_swiglu_quant

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |

## 功能说明

- API功能：在SwiGlu激活函数后添加quant操作，实现输入`x`的SwiGluQuant融合计算，支持int8或int4量化输出。支持MoE场景（传入`group_index`）与非MoE场景（`group_index`传空），支持动态量化与静态量化（per_tensor/per_channel）。
- 计算公式（MoE场景动态量化，其中A表示输入`x`的前半部分，B表示后半部分，g表示`group_index`，G为分组数量；int8量化时 $dstTypeScale=127$，int4量化时 $dstTypeScale=7$）：

  $$
  Act = SwiGlu(x) = Swish(A) * B
  $$

  $$
  scale = dstTypeScale / row\_max(abs(Y_{tmp})),\quad Y = Cast(Mul(Y_{tmp}, scale))
  $$

> [!NOTE]
> - 量化场景说明：`x`的输入来自GroupedMatMul算子和MoeInitRouting的输出，通过`group_index`实现MoE分组动态量化、静态per_tensor量化、静态per_channel量化。
> - `dst_type`控制量化输出数据类型：`2`表示int8量化输出，`29`表示int4量化输出。int4量化时`x`的最后一维需为4的倍数。
> - `offsets`仅在静态量化场景生效；动态量化场景下传`None`即可。

## 函数原型

```
torch_npu.npu_swiglu_quant(x, *, smooth_scales=None, offsets=None, group_index=None, activate_left=False, quant_mode="dynamic", group_list_type=0, dst_type=2) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入待处理的数据，公式中的x，数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$。`x`的维数必须大于1维，最后一维需为2的倍数且长度不超过8192；当`dst_type`为`29`（int4量化）时，最后一维还需为4的倍数。
- <strong>*</strong>：位置参数与关键字参数的分隔符。其之前的参数为位置参数，需按顺序传入；其之后的参数为关键字参数，需通过键值对方式传入，未赋值时使用默认值。
- **smooth_scales** (`Tensor`)：可选参数，量化的smooth_scales，公式中的smooth_scales，数据类型支持`float32`，数据格式支持$ND$。不支持空Tensor；shape支持`[G, N]`或`[G, ]`，其中G为`group_index`分组数量，N为`x`最后一维大小的二分之一。默认值为`None`。
- **offsets** (`Tensor`)：可选参数，公式中的offsets，数据类型支持`float32`，数据格式支持$ND$。该参数在动态量化场景下不生效；静态量化场景下，per_channel模式shape支持`[G, N]`，per_tensor模式shape支持`[G, ]`且数据类型与shape需与`smooth_scales`保持一致。默认值为`None`。
- **group_index** (`Tensor`)：可选参数，MoE分组需要的group_index，公式中的group_index，数据类型支持`int32`，数据格式支持$ND$。shape支持`[G, ]`，元素要求非递减，且最大值不超过`x`除最后一维之外所有维度大小之积。默认值为`None`（非MoE场景）。
- **activate_left** (`bool`)：可选参数，表示左矩阵是否参与运算。默认值为`False`。
- **quant_mode** (`str`)：可选参数，量化模式。当前支持取值：
  - `"dynamic"`：动态量化（默认）。
  - `"static"`：静态量化，仅支持per_tensor量化和per_channel量化。

  默认值为`"dynamic"`。
- **group_list_type** (`int`)：可选参数，`group_index`的组织模式。取值：
  - `0`：cumsum模式（累计边界，默认）。
  - `1`：count模式（每组计数）。

  默认值为`0`。
- **dst_type** (`int`)：可选参数，量化输出数据类型。取值：
  - `2`：`y`输出为int8量化（默认）。
  - `29`：`y`输出为int4量化。

  默认值为`2`。

## 返回值说明

- **y** (`Tensor`)：量化输出张量，shape最后一维为`x`最后一维的二分之一，其余维度与`x`保持一致。数据类型由`dst_type`决定：`dst_type=2`时为`int8`，`dst_type=29`时为int4量化输出。
- **scale** (`Tensor`)：量化scale输出张量，数据类型为`float32`，shape与`x`相比无最后一维，其余维度与`x`保持一致。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式。
- `x`维数必须大于1维，最后一维需为2的倍数且不超过8192。
- 当`dst_type`为`29`（int4量化）时，`x`的最后一维需为4的倍数。
- `smooth_scales`不支持空Tensor。
- `group_index`内元素要求非递减。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn((3, 32), dtype=torch.bfloat16, device='npu')
    >>> smooth_scales = torch.ones((2, 16), dtype=torch.float32, device='npu')
    >>> group_index = torch.tensor([1, 3], dtype=torch.int32, device='npu')
    >>> y, scale = torch_npu.npu_swiglu_quant(x, smooth_scales=smooth_scales, group_index=group_index)
    >>> y.shape
    torch.Size([3, 16])
    >>> scale.shape
    torch.Size([3])
    >>> y.dtype
    torch.int8
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
        def forward(self, x, smooth_scales, group_index):
            return torch_npu.npu_swiglu_quant(x, smooth_scales=smooth_scales, group_index=group_index)

    x = torch.randn((3, 32), dtype=torch.bfloat16, device='npu')
    smooth_scales = torch.ones((2, 16), dtype=torch.float32, device='npu')
    group_index = torch.tensor([1, 3], dtype=torch.int32, device='npu')

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    y, scale = model(x, smooth_scales, group_index)
    print(y, scale)
    ```
