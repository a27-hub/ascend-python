# torch_npu.npu_swi_glu_quant_v2

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- API功能：在SwiGlu激活函数后添加量化（quant）操作，实现输入`x`的SwiGluQuant计算，支持`int8`或`int4`量化输出。
- 接口功能差异点：相比于`aclnnSwiGluQuant`接口，本接口（对应aclnn `aclnnSwiGluQuantV2`）新增支持`group_index`传入cumsum模式和count模式（通过`group_list_type`控制）；新增支持非MoE场景（`group_index`传空）；新增支持`int8`或`int4`量化输出（通过`dst_type`控制）。
- 算子支持范围：支持MoE场景（传入`group_index`）和非MoE场景（`group_index`传空）。
- 计算公式（动态量化）：

  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp} = Act * smooth\_scales \\
    scale = dstTypeScale / row\_max(abs(Y_{tmp})) \\
    Y = Cast(Mul(Y_{tmp}, Scale))
  $$

  其中`A`为输入`x`的前半部分，`B`为后半部分；`int8`量化时 $dstTypeScale = 127$，`int4`量化时 $dstTypeScale = 7$。

> [!NOTE]
> 动态量化时返回的`scale`为量化scale（$dstTypeScale/row\_max$），非反量化scale。`group_index`在cumsum模式（`group_list_type=0`）下为累计行边界（如`[2, 4]`表示行`0:2`与`2:4`两组）。

## 函数原型

```
torch_npu.npu_swi_glu_quant_v2(x, *, smooth_scales=None, offsets=None, group_index=None,
                               activate_left=False, quant_mode="dynamic", group_list_type=0, dst_type=2)
                               -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入待处理数据，公式中的`x`。数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$。
- **\***：位置参数与关键字参数的分隔符。其之前的参数为位置参数，需按顺序传入；其之后的参数为关键字参数，需通过键值对方式传入，未赋值时使用默认值。
- **smooth_scales** (`Tensor`)：可选参数，量化的smooth_scales，公式中的`smooth_scales`。数据类型支持`float32`，数据格式支持$ND$。shape支持`[G, N]`或`[G, ]`，其中`G`代表`group_index`分组数量，`N`为`x`最后一维大小的二分之一。默认值为`None`。
- **offsets** (`Tensor`)：可选参数，公式中的`offsets`，静态量化场景下生效。数据类型支持`float32`，数据格式支持$ND$。per_channel模式下shape支持`[G, N]`，per_tensor模式下shape支持`[G, ]`。默认值为`None`。
- **group_index** (`Tensor`)：可选参数，MoE分组需要的`group_index`，公式中的`group_index`。数据类型支持`int32`，数据格式支持$ND$，shape支持`[G, ]`。默认值为`None`（表示非MoE场景）。
- **activate_left** (`bool`)：可选参数，表示左矩阵是否参与运算，`True`表示对`x`前半部分做Swish激活，`False`表示对后半部分做Swish激活。默认值为`False`。
- **quant_mode** (`str`)：可选参数，量化模式。`"dynamic"`表示动态量化，`"static"`表示静态量化（静态量化仅支持per_tensor和per_channel量化，且需传入`offsets`）。默认值为`"dynamic"`。
- **group_list_type** (`int`)：可选参数，`group_index`的解释模式，`0`表示cumsum模式（累计边界），`1`表示count模式（每组长度）。默认值为`0`。
- **dst_type** (`int`)：可选参数，量化输出数据类型，`2`表示`int8`量化输出，`29`表示`int4`量化输出（2-per-byte打包为`uint8`）。默认值为`2`。

## 返回值说明

- **y** (`Tensor`)：量化输出。`dst_type=2`时数据类型为`int8`，shape最后一维为`x`最后一维的二分之一，其余维度与`x`一致；`dst_type=29`时数据类型为`uint8`（`int4`按2-per-byte打包），shape最后一维为`x`最后一维的四分之一。
- **scale** (`Tensor`)：量化scale，数据类型为`float32`，shape为`x`去掉最后一维。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式。
- `x`的维数必须大于1维；`x`最后一维需要为2的倍数；当前仅支持`x`最后一维长度不超过8192。
- 当`dst_type`传入`29`（int4量化）时，`x`最后一维需要为4的倍数。
- `quant_mode`仅支持`"dynamic"`与`"static"`；`group_list_type`仅支持`0`与`1`；`dst_type`仅支持`2`与`29`。
- 静态量化（`quant_mode="static"`）场景下需传入`offsets`。
- 该接口默认确定性实现。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn((32, 256), dtype=torch.float32, device='npu')
    >>> smooth_scales = torch.ones((1, 128), dtype=torch.float32, device='npu')
    >>> y, scale = torch_npu.npu_swi_glu_quant_v2(x, smooth_scales=smooth_scales, dst_type=2)
    >>> y.shape
    torch.Size([32, 128])
    >>> y.dtype
    torch.int8
    >>> scale.shape
    torch.Size([32])
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu

    def fn(x, smooth_scales):
        return torch_npu.npu_swi_glu_quant_v2(x, smooth_scales=smooth_scales, dst_type=2)

    x = torch.randn((32, 256), dtype=torch.float32, device='npu')
    smooth_scales = torch.ones((1, 128), dtype=torch.float32, device='npu')
    compiled = torch.compile(fn, backend="npugraph_ex")
    y, scale = compiled(x, smooth_scales)
    print(y.shape, scale.shape)
    ```
