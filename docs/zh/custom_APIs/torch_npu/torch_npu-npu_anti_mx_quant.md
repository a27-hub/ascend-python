# torch_npu.npu_dynamic_block_quant

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |

## 功能说明

- API功能：将调用`npu_dynamic_mx_quant`量化得到的`float4/float8`的Tensor反量化为`float16/bfloat16/float32`格式。

- 计算公式：
  $$
  X_{dq} = X_q \times 2^{sf - bias}
  $$

  其中 $sf$ 是缩放因子，由输入`mxscale`提供；$bias$ 是指数位的偏移，对于`float8_e8m0`格式，$bias=127$；$X_q$ 是量化得到的`float4/float8`张量；$X_{dq}$ 是反量化得到的`float16/bfloat16/float32`张量。

## 函数原型

```python
torch_npu.npu_anti_mx_quant(x, mxscale, *, axis=-1, dst_type=15, src_type=292) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，待反量化的Tensor。数据类型支持`float4_e2m1`、`float4_e1m2`、`float8_e5m2`、`float8_e4m3fn`，`float8`支持非连续的Tensor，`float4`不支持非连续的Tensor，数据格式支持ND。当前shape支持1-7维。
- **mxscale** (`Tensor`)：必选参数，输入张量，参与反量化计算的量化尺度。数据类型支持`float8_e8m0`，数据格式支持ND。当前shape支持2-8维。
- **axis** (`int`)：可选参数，指定反量化轴。当前支持取值范围为`[-D, D-1]`，其中`D`为输入x的维度数。
- **dst_type** (`int`)：可选参数，指定输出`y`的数据类型。当前支持取值范围为`{0, 1, 27}`，分别对应输出`y`的数据类型为`{0: float32, 1: float16, 27: bfloat16}`。
- **src_type** (`int`)：可选参数，指定输入`x`的数据类型。当前支持取值范围为`{291, 292, 296, 297}`，分别对应输入`x`的数据类型为`{291: float8_e5m2, 292: float8_e4m3fn, 296: float4_e2m1, 297: float4_e1m2}`。

## 返回值说明

- **y** (`Tensor`)：反量化结果。数据类型支持`float16/bfloat16/float32`，`float8`支持非连续的Tensor，`float4`不支持非连续的Tensor，数据格式支持ND。shape需与输入`x`保持一致。

## 调用示例

  ```python
  import torch
  import torch_npu

  # method 1: generate inputs manually
  x = torch.randn((16, 128), dtype=torch.float32).to(dtype=torch.float8_e5m2)
  x = x.npu()
  mxscale = torch.randn((16, 4, 2), dtype=torch.float32).to(dtype=torch.float8_e8m0fnu)
  mxscale = mxscale.npu()
  # run api
  y = torch_npu.npu_anti_mx_quant(x, mxscale, axis=-1, dst_type=torch.bfloat16, src_type=torch.float8_e5m2)
  print("x:============", x.shape(), x.cpu())
  print("mxscale:============", mxscale.shape(), mxscale.cpu())
  print("y:============", y.shape(), y.cpu())

  # method 2: generate inputs by npu_dynamic_mx_quant
  x_tmp = torch.randn((16, 128), dtype=torch.bfloat16).npu()
  x, mxscale = torch_npu.npu_dynamic_mx_quant(x_tmp, axis=-1, round_mode="rint", dst_type=torch.float8_e5m2, block_size=32, scale_alg=0)
  # run api
  y = torch_npu.npu_anti_mx_quant(x, mxscale, axis=-1, dst_type=torch.bfloat16, src_type=torch.float8_e5m2)
  print("x:============", x.shape(), x.cpu())
  print("mxscale:============", mxscale.shape(), mxscale.cpu())
  print("y:============", y.shape(), y.cpu())
  ```
