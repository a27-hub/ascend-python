# torch\_npu.npu\_swi\_glu\_quant\_v2

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- API功能：在SwiGlu激活函数后添加quant操作，实现输入`x`的SwiGluQuant计算，支持int8或int4量化输出。
- 接口功能差异点：相比`aclnnSwiGluQuant`，本接口（V2）新增支持`group_index`的cumsum模式与count模式（通过`group_list_type`控制）；新增支持非MoE场景（`group_index`传空）；新增支持int8或int4量化输出`y`，通过`dst_type`控制。
- 支持场景：MoE场景（传入`group_index`）与非MoE场景（`group_index`传空），通过`group_index`实现MoE分组动态量化、静态per_tensor量化、静态per_channel量化。
- 计算公式（MoE场景动态量化）：

  $$
    Act = SwiGlu(x) = Swish(A)*B
    $$
    $$
    Y_{tmp}^i = Act[g[i]\colon g[i+1], \colon] * smooth\_scales[g[i]\colon g[i+1], \colon]
    $$
    $$
    scale = dstTypeScale / row\_max(abs(Y_{tmp}))
    $$
    $$
    Y = Cast(Mul(Y_{tmp}, Scale))
    $$

  其中`A`表示输入`x`的前半部分，`B`表示后半部分，`g`表示`group_index`，`G`为分组数量。int8量化时`dstTypeScale = 127`，int4量化时`dstTypeScale = 7`。

- 计算公式（非MoE场景动态量化，`group_index`传空）：

  $$
    Act = SwiGLU(x) = Swish(A)*B
    $$
    $$
    Y_{tmp} = Act * smooth\_scales
    $$
    $$
    scale = dstTypeScale / row\_max(abs(Y_{tmp}))
    $$
    $$
    Y = Cast(Mul(Y_{tmp}, Scale))
    $$

  静态量化场景在`Y_tmp`上叠加`offsets`，且输出`Y = Cast(Y_tmp)`（不乘scale）。

> [!NOTE]
> - 通过`activate_left`控制SwiGlu的左右激活：`True`时对前半部分`A`做Swish，`False`时对后半部分`B`做Swish。
> - `group_index`在`group_list_type=0`（cumsum）模式下为累计边界（如`[2,4]`表示两组：rows 0:2 与 rows 2:4）。
> - 动态量化时`offsets`不生效，可传入空；静态量化时`offsets`生效。

## 函数原型

```
torch_npu.npu_swi_glu_quant_v2(x, *, smooth_scales=None, offsets=None, group_index=None,
                              activate_left=False, quant_mode="dynamic", group_list_type=0,
                              dst_type=2) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入待处理的数据。数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$，不支持非连续Tensor。
- *：位置参数与关键字参数的分隔符。其之前的参数为位置参数，需按顺序传入；其之后的参数为关键字参数，需通过键值对方式传入，未赋值时使用默认值。
- **smooth_scales** (`Tensor`)：可选参数，量化的smooth_scales。不支持空Tensor；shape支持`[G, N]`或`[G, ]`，其中`G`为`group_index`分组数量，`N`为输入`x`最后一维大小的二分之一；数据类型支持`float32`，数据格式支持$ND$。默认值为`None`。
- **offsets** (`Tensor`)：可选参数，公式中的offsets。动态量化场景下不生效（传入空即可）；静态量化场景下数据类型支持`float32`，per_channel模式下shape支持`[G, N]`，per_tensor模式下shape支持`[G, ]`且需与`smooth_scales`的数据类型和shape保持一致。默认值为`None`。
- **group_index** (`Tensor`)：可选参数，MoE分组的group_index。shape支持`[G, ]`，元素要求非递减，且最大值不得超过输入`x`除最后一维外所有维度大小之积，`G`值不得超过同一上限；数据类型支持`int32`。传`None`表示非MoE场景。默认值为`None`。
- **activate_left** (`bool`)：可选参数，表示左矩阵是否参与SwiGlu运算（`True`对前半部分做Swish，`False`对后半部分做Swish）。默认值为`False`。
- **quant_mode** (`str`)：可选参数，量化模式。支持取值：
  - `"dynamic"`：动态量化
  - `"static"`：静态量化（仅支持per_tensor与per_channel）
  
  默认值为`"dynamic"`。
- **group_list_type** (`int`)：可选参数，`group_index`的解析模式。支持取值：
  - `0`：cumsum模式（`group_index`为累计边界）
  - `1`：count模式（`group_index`为每组元素数）
  
  默认值为`0`。
- **dst_type** (`int`)：可选参数，量化输出数据类型。支持取值：
  - `2`：int8量化输出（默认）
  - `29`：int4量化输出（int4采用2-per-byte打包，输出`y`的dtype为`uint8`，物理最后一维 = 输入`x`最后一维 / 4）
  
  默认值为`2`。

## 返回值说明

- **y** (`Tensor`)：量化输出张量。shape除最后一维为输入`x`最后一维的二分之一外，其余维度与`x`保持一致；int4（`dst_type=29`）因2-per-byte打包，物理最后一维为`x`最后一维的四分之一。数据类型：int8（`dst_type=2`）或`uint8`（`dst_type=29`）。
- **scale** (`Tensor`)：量化scale张量。shape为输入`x`去掉最后一维后的形状，其余维度与`x`保持一致；数据类型为`float32`。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式（`torch.compile`）。
- 输入`x`的维数必须大于1维；`x`的最后一维需为2的倍数，且不超过8192。
- 当`dst_type=29`（int4）时，`x`的最后一维需为4的倍数。
- `quant_mode`当前仅支持`"dynamic"`与`"static"`；`static`仅支持per_tensor与per_channel。
- `group_list_type`仅支持`0`（cumsum）与`1`（count）。
- `dst_type`仅支持`2`（int8）与`29`（int4）。
- `group_index`元素需非递减，最大值与`G`均不得超过`x`除最后一维外所有维度大小之积。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn((3, 32), dtype=torch.float16, device='npu')
    >>> smooth_scales = torch.ones((2, 16), dtype=torch.float32, device='npu')
    >>> group_index = torch.tensor([1, 3], dtype=torch.int32, device='npu')
    >>> y, scale = torch_npu.npu_swi_glu_quant_v2(
    ...     x, smooth_scales=smooth_scales, group_index=group_index,
    ...     activate_left=False, quant_mode="dynamic", group_list_type=0, dst_type=2)
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

    x = torch.randn((3, 32), dtype=torch.float16, device='npu')
    smooth_scales = torch.ones((2, 16), dtype=torch.float32, device='npu')
    group_index = torch.tensor([1, 3], dtype=torch.int32, device='npu')

    class Model(torch.nn.Module):
        def forward(self, x, smooth_scales, group_index):
            return torch_npu.npu_swi_glu_quant_v2(
                x, smooth_scales=smooth_scales, group_index=group_index, dst_type=2)

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    y, scale = model(x, smooth_scales, group_index)
    print(y, scale)
    ```
