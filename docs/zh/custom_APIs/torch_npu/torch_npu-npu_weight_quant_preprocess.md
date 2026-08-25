# torch\_npu.npu\_weight\_quant\_preprocess

## 产品支持情况


| 产品                                   | 是否支持 |
| -------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> |    √    |

## 功能说明

该接口用于伪量化场景下Matmul类算子的权重预处理，当前主要实现`weight`张量从$ND$格式转换为$FRACTAL\_NZ$格式。涉及的Matmul类算子包括QuantBatchMatmulV4、GroupedMatmul。

## 函数原型

```python
torch_npu.npu_weight_quant_preprocess(weight, weight_scale, x_dtype, weight_dtype, weight_scale_dtype, weight_offset=None, bias=None, x_scale_dtype=None, k_group_size=0) -> Tuple[Tensor, Tensor, Tensor, Tensor]
```

## 参数说明

- **weight**（`Tensor`）：**必选参数**，Matmul的权重矩阵，支持非连续`Tensor`。
  - <term>Ascend 950PR/Ascend 950DT</term>：逻辑数据类型仅支持`float4_e2m1fn_x2`（使用`uint8`承载），数据格式支持$ND$。支持2维或3维输入，逻辑shape分别为$(K, N)$、$(G, K, N)$，其中$G$表示group数量。1个`uint8`元素打包2个FP4数据，因此`weight`的物理K维长度为$K/2$。
- **weight\_scale**（`Tensor`）：**必选参数**，权重的反量化scale参数，支持非连续`Tensor`。
  - <term>Ascend 950PR/Ascend 950DT</term>：逻辑数据类型仅支持`float8_e8m0fnu`（使用`uint8`承载），数据格式支持$ND$、$NCL$。支持3维或4维输入，shape分别为$(ceil\_div(K, 64), N, 2)$、$(G, ceil\_div(K, 64), N, 2)$。
- **x\_dtype**（`int`）：**必选参数**，Matmul的激活矩阵的数据类型。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持`torch.float8_e4m3fn`。
- **weight\_dtype**（`int`）：**必选参数**，用于指定`weight`中实际承载的数据类型。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持`torch.float4_e2m1fn_x2`。
- **weight\_scale\_dtype**（`int`）：**必选参数**，用于指定`weight_scale`中实际承载的数据类型。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持`torch.float8_e8m0fnu`。
- **weight\_offset**（`Tensor`）：**可选参数**，权重的反量化offset参数，默认值为`None`。
  - <term>Ascend 950PR/Ascend 950DT</term>：当前版本不支持此参数，请传入`None`。
- **bias**（`Tensor`）：**可选参数**，Matmul的偏置矩阵，必须为连续的`Tensor`。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$。支持1维或2维输入，shape为$(N)$、$(1, N)$或$(G, N)$。
- **x\_scale\_dtype**（`int`）：**可选参数**，激活的量化scale参数的数据类型，默认值为`None`。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持`torch.float8_e8m0fnu`。
- **k\_group\_size**（`int`）：**可选参数**，权重在per-group量化时K维度的group大小，默认值为`0`。
  - <term>Ascend 950PR/Ascend 950DT</term>：当前仅支持`32`。

## 返回值说明

`Tuple[Tensor, Tensor, Tensor, Tensor]`

返回四个`Tensor`：

- **out\_weight**（`Tensor`）：预处理后的`weight`，数据类型与输入`weight`相同，数据格式为$FRACTAL\_NZ\_C0\_16$。
- **out\_weight\_scale**（`Tensor`）：预处理后的`weight_scale`，数据类型与输入`weight_scale`相同。
- **out\_weight\_offset**（`Tensor`）：预处理后的`weight_offset`，当前版本返回空`Tensor`。
- **out\_bias**（`Tensor`）：预处理后的`bias`，数据类型与输入`bias`相同（若提供了`bias`）。

## 约束说明

当前支持如下参数组合：

- **MM_MX_A8W4数据流**（仅支持<term>Ascend 950PR/Ascend 950DT</term>）：
  - `weight`数据类型必须为`FLOAT4_E2M1`，数据格式为$ND$，K必须满足$K \% k\_group\_size = 0$。
  - `weight`的逻辑shape为$\{K, N\}$；使用`uint8`承载FP4数据时，view shape为$\{K/2, N\}$，storage shape为$\{N, K/2\}$（transposed），stride为$[1, K/2]$。
  - `weight_scale`数据类型必须为`FLOAT8_E8M0`，数据格式为$ND$/$NCL$。
  - `weight_scale`的view shape为$\{ceil\_div(K, 64), N, 2\}$，storage shape为$\{N, ceil\_div(K, 64), 2\}$（transposed）。
  - `k_group_size`必须等于`32`。
  - `x_dtype`必须为`FLOAT8_E4M3FN`。
  - `x_scale_dtype`必须为`FLOAT8_E8M0`。
  - 当前不支持`weight_offset`，必须传入`None`。
- **GMM_MX_A8W4数据流**（仅支持<term>Ascend 950PR/Ascend 950DT</term>）：
  - `weight`数据类型必须为`FLOAT4_E2M1`，数据格式为$ND$，K必须满足$K \% k\_group\_size = 0$。
  - `weight`的逻辑shape为$\{G, K, N\}$；使用`uint8`承载FP4数据时，view shape为$\{G, K/2, N\}$，storage shape为$\{G, N, K/2\}$（transposed），stride为$[K * N/2, 1, K/2]$。
  - `weight_scale`数据类型必须为`FLOAT8_E8M0`，数据格式为$ND$/$NCL$。
  - `weight_scale`的view shape为$\{G, ceil\_div(K, 64), N, 2\}$，storage shape为$\{G, N, ceil\_div(K, 64), 2\}$（transposed），stride为$[N * ceil\_div(K, 64) * 2, 2, ceil\_div(K, 64) * 2, 1]$。
  - `k_group_size`必须等于`32`。
  - `x_dtype`必须为`FLOAT8_E4M3FN`。
  - `x_scale_dtype`必须为`FLOAT8_E8M0`。
  - 当前不支持`weight_offset`，必须传入`None`。
- 不支持空`Tensor`输入。
- 其余数据类型与shape组合为预留接口，当前调用将报错。

## 调用示例

- MM_MX_A8W4数据流场景

  ```python
  import torch
  import torch_npu

  # MM_MX_A8W4 数据流示例
  k = 64  # 逻辑K维长度
  n = 128
  packed_k = k // 2
  k_scale = (k + 63) // 64

  # weight: FLOAT4_E2M1，每个 uint8 打包2个FP4数据
  # logical shape: {K, N}, view: {K/2, N}, storage: {N, K/2}
  cpu_weight = torch.randint(0, 255, (n, packed_k), dtype=torch.uint8)
  weight = cpu_weight.npu().transpose(0, 1)

  # weight_scale: FLOAT8_E8M0, 3-D transposed
  # view: {ceil_div(K, 64), N, 2}, storage: {N, ceil_div(K, 64), 2}
  cpu_scale = torch.randint(0, 255, (n, k_scale, 2), dtype=torch.uint8)
  weight_scale = cpu_scale.npu().transpose(0, 1)

  out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
      weight,
      weight_scale,
      x_dtype=torch.float8_e4m3fn,
      weight_dtype=torch.float4_e2m1fn_x2,
      weight_scale_dtype=torch.float8_e8m0fnu,
      weight_offset=None,
      bias=None,
      x_scale_dtype=torch.float8_e8m0fnu,
      k_group_size=32
  )
  ```
- GMM_MX_A8W4数据流场景

  ```python
  import torch
  import torch_npu

  # GMM_MX_A8W4 数据流示例
  g = 4
  k = 64  # 逻辑K维长度
  n = 128
  packed_k = k // 2
  k_scale = (k + 63) // 64

  # weight: FLOAT4_E2M1，每个 uint8 打包2个FP4数据
  # logical shape: {G, K, N}, view: {G, K/2, N}, storage: {G, N, K/2}
  cpu_weight = torch.randint(0, 255, (g, n, packed_k), dtype=torch.uint8)
  weight = cpu_weight.npu().transpose(1, 2)

  # weight_scale: FLOAT8_E8M0, 4-D transposed
  # view: {G, ceil_div(K, 64), N, 2}, storage: {G, N, ceil_div(K, 64), 2}
  cpu_scale = torch.randint(0, 255, (g, n, k_scale, 2), dtype=torch.uint8)
  weight_scale = cpu_scale.npu().transpose(1, 2)

  out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
      weight,
      weight_scale,
      x_dtype=torch.float8_e4m3fn,
      weight_dtype=torch.float4_e2m1fn_x2,
      weight_scale_dtype=torch.float8_e8m0fnu,
      weight_offset=None,
      bias=None,
      x_scale_dtype=torch.float8_e8m0fnu,
      k_group_size=32
  )
  ```
