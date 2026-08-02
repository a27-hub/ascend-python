# torch\_npu.npu\_dynamic\_mx\_quant\_with\_dual\_axis\_out

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

对输入张量的最后两轴分别进行MX量化，并将两路量化结果及对应的scale写入调用方预先分配的四个输出张量。

该接口不会在算子内部为`y1_out`、`mxscale1_out`、`y2_out`、`mxscale2_out`分配payload显存，适用于调用方已有可复用输出缓冲区的场景。该接口不是原地接口：`input`与任一输出、任意两个输出之间均不得有显存重叠。

## 函数原型

```python
torch_npu.npu_dynamic_mx_quant_with_dual_axis_out(
    input,
    y1_out,
    mxscale1_out,
    y2_out,
    mxscale2_out,
    *,
    round_mode="rint",
    dst_type=296,
    scale_alg=0,
    dst_type_max=0.0
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **input**（`Tensor`）：必选参数，待量化数据。至少为2维，数据类型支持`float16`、`bfloat16`。
- **y1_out**（`Tensor`）：必选参数，最后一轴量化结果的预分配输出。
- **mxscale1_out**（`Tensor`）：必选参数，最后一轴量化scale的预分配输出，使用`uint8`存储`float8_e8m0fnu`数据。
- **y2_out**（`Tensor`）：必选参数，倒数第二轴量化结果的预分配输出。
- **mxscale2_out**（`Tensor`）：必选参数，倒数第二轴量化scale的预分配输出，使用`uint8`存储`float8_e8m0fnu`数据。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，需使用键值对赋值。
- **round\_mode**（`str`）：可选参数，数据转换模式，默认值为`"rint"`。
- **dst\_type**（`int`）：可选参数，量化结果的数据类型，与`torch_npu.npu_dynamic_mx_quant_with_dual_axis`一致，默认值为`296`。
- **scale\_alg**（`int`）：可选参数，scale计算模式，默认值为`0`。
- **dst\_type\_max**（`float`）：可选参数，量化类型的最大值，默认值为`0.0`。旧版CANN会忽略非默认值并给出告警。

## 输出张量要求

设`input`的shape为`[d0, ..., d(n-2), d(n-1)]`：

- FP8输出时，`y1_out`和`y2_out`的shape与`input`一致，数据类型必须与`dst_type`一致。
- FLOAT4输出时，`y1_out`和`y2_out`使用`uint8`存储，最后一维为`d(n-1) / 2`，因此输入最后一维必须是偶数。
- `mxscale1_out`的shape为`[d0, ..., d(n-2), (ceil(d(n-1) / 32) + 1) // 2, 2]`。
- `mxscale2_out`的shape为`[d0, ..., (ceil(d(n-2) / 32) + 1) // 2, d(n-1), 2]`。

## 返回值说明

返回`(y1_out, mxscale1_out, y2_out, mxscale2_out)`。返回张量与传入的四个输出张量分别指向同一数据地址，不会替换或重新分配其payload。

## 约束说明

- `input`及四个输出必须位于同一NPU设备，且均为连续Tensor。
- `input`与每个输出不得重叠，四个输出之间也必须两两不重叠；违反约束时接口直接报错，不执行量化。
- `y1_out`和`y2_out`允许是同一底层storage中地址区间不重叠的两个连续view，便于复用一块已有payload缓冲区。
- 四个输出的shape和数据类型必须严格匹配上述要求，接口不会自动调整shape、数据类型或重新分配输出。
- 该接口支持训练、推理场景下使用。

## 调用示例

```python
import torch
import torch_npu

input = torch.randn((1, 64, 64), dtype=torch.bfloat16, device="npu")
y1_out = torch.empty_like(input, dtype=torch.float8_e5m2)
y2_out = torch.empty_like(input, dtype=torch.float8_e5m2)
mxscale1_out = torch.empty((1, 64, 1, 2), dtype=torch.uint8, device="npu")
mxscale2_out = torch.empty((1, 1, 64, 2), dtype=torch.uint8, device="npu")

y1, mxscale1, y2, mxscale2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis_out(
    input,
    y1_out,
    mxscale1_out,
    y2_out,
    mxscale2_out,
    dst_type=23,
)

assert y1.data_ptr() == y1_out.data_ptr()
assert mxscale1.data_ptr() == mxscale1_out.data_ptr()
assert y2.data_ptr() == y2_out.data_ptr()
assert mxscale2.data_ptr() == mxscale2_out.data_ptr()
```
