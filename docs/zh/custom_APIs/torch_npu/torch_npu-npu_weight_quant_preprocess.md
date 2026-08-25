# torch\_npu.npu\_weight\_quant\_preprocess

## 产品支持情况

| 产品                                   | 是否支持 |
| -------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> |    √    |

## 功能说明

该接口用于伪量化场景下Matmul类算子的伪量化参数的预处理。涉及的Matmul类算子包括QuantBatchMatmulV4、GroupedMatmul、WeightQuantBatchMatmulV2。

## 函数原型

```python
torch_npu.npu_weight_quant_preprocess(weight, weight_scale, x_dtype, weight_dtype, weight_scale_dtype, weight_offset=None, bias=None, x_scale_dtype=None, k_group_size=0) -> Tuple[Tensor, Tensor, Tensor, Tensor]
```

## 参数说明

- **weight**（`Tensor`）：**必选参数**，Matmul的权重矩阵，支持非连续`Tensor`。
  - <term>Ascend 950PR/Ascend 950DT</term>：逻辑数据类型支持`int4`（使用`uint8`承载）、`float4_e2m1fn_x2`（使用`uint8`承载），数据格式支持$ND$。支持2维或3维输入，逻辑shape分别为$(K, N)$、$(G, K, N)$，其中$G$表示group数量。1个`uint8`元素打包2个4-bit数据，要求打包维度的元素个数为偶数：沿K维打包时`weight`的物理K维长度为$K/2$，沿N维打包时物理N维长度为$N/2$。
- **weight\_scale**（`Tensor`）：**必选参数**，权重的反量化scale参数，支持非连续`Tensor`。
  - <term>Ascend 950PR/Ascend 950DT</term>：逻辑数据类型支持`float8_e8m0fnu`（使用`uint8`承载）、`float16`、`bfloat16`，数据格式支持$ND$、$NCL$。
    - MX格式scale：支持3维或4维输入，shape分别为$(ceil\_div(K, 64), N, 2)$、$(G, ceil\_div(K, 64), N, 2)$。
    - per-tensor场景：shape为$(1)$、$(1, 1)$。
    - per-channel场景：shape为$(N)$、$(1, N)$。
    - per-group场景：shape为$(G, N)$，其中$G$表示group数量。
- **x\_dtype**（`int`）：**必选参数**，Matmul的激活矩阵的数据类型。
  - <term>Ascend 950PR/Ascend 950DT</term>：A16S4数据流支持`torch.float16`、`torch.bfloat16`；A8W4数据流支持`torch.float8_e4m3fn`。
- **weight\_dtype**（`int`）：**必选参数**，用于指定`weight`中实际承载的数据类型。
  - <term>Ascend 950PR/Ascend 950DT</term>：A16S4数据流取值为`torch_npu.int4`；其余数据流取值为`torch.float4_e2m1fn_x2`。
- **weight\_scale\_dtype**（`int`）：**必选参数**，用于指定`weight_scale`中实际承载的数据类型，取值需与`weight_scale`的实际数据类型一致。
  - <term>Ascend 950PR/Ascend 950DT</term>：支持`torch.float8_e8m0fnu`、`torch.float16`、`torch.bfloat16`。
- **weight\_offset**（`Tensor`）：**可选参数**，权重的反量化offset参数，默认值为`None`。
  - <term>Ascend 950PR/Ascend 950DT</term>：A16S4各数据流支持透传，其shape和数据类型需与`weight_scale`保持一致；其余数据流请传入`None`。
- **bias**（`Tensor`）：**可选参数**，Matmul的偏置矩阵，必须为连续的`Tensor`。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$。支持1维或2维输入，shape为$(N)$、$(1, N)$或$(G, N)$。
- **x\_scale\_dtype**（`int`）：**可选参数**，激活的量化scale参数的数据类型，默认值为`None`。
  - <term>Ascend 950PR/Ascend 950DT</term>：A8W4数据流仅支持`torch.float8_e8m0fnu`；A16S4数据流请传入`None`。
- **k\_group\_size**（`int`）：**可选参数**，权重在per-group量化时K维度的group大小，默认值为`0`。
  - <term>Ascend 950PR/Ascend 950DT</term>：A8W4数据流（MX格式scale）取值为`32`；A16S4 per-group场景取值为大于`0`的整数，表示K维per-group的group大小；其余场景使用默认值`0`。

## 返回值说明

`Tuple[Tensor, Tensor, Tensor, Tensor]`

返回四个`Tensor`：

- **out\_weight**（`Tensor`）：预处理后的`weight`，数据类型与输入`weight`相同。数据格式按数据流区分：$ND$直拷场景（A16S4 per-tensor、A16S4 per-channel/per-group转置输入）输出为$ND$，与输入`weight`保持相同的sizes/strides（属纯透传）；A16S4 per-channel/per-group非转置输入场景输出为$FRACTAL\_NZ\_C0\_8$，view shape与输入`weight`相同（uint8打包视图），storage shape为$\{ceil\_div(N, 16), ceil\_div(K, 16), 16, 8\}$；A8W4数据流输出为$FRACTAL\_NZ\_C0\_16$。
- **out\_weight\_scale**（`Tensor`）：预处理后的`weight_scale`，数据类型与输入`weight_scale`相同。
- **out\_weight\_offset**（`Tensor`）：预处理后的`weight_offset`，未传入`weight_offset`时返回空`Tensor`；A16S4数据流传入时透传，与输入保持相同的sizes/strides。
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
- **MM_A16S4数据流**（仅支持<term>Ascend 950PR/Ascend 950DT</term>）：
  - `weight`数据类型必须为`INT4`，使用`uint8`承载（1个`uint8`元素打包2个INT4数据），数据格式为$ND$，支持2维输入，转置与非转置均支持。
  - 非转置时`weight`的shape为$\{K, N/2\}$（沿N维打包），转置时shape为$\{K/2, N\}$（沿K维打包），打包维度的元素个数须为偶数。
  - `weight_scale`数据类型为`float16`或`bfloat16`，其shape决定量化粒度。
  - `x_dtype`必须为`float16`或`bfloat16`。
  - `x_scale_dtype`必须传入`None`。
  - `weight_offset`支持透传，shape和数据类型需与`weight_scale`保持一致。
  - 根据`weight_scale`的shape分为如下三种场景：
    - per-tensor：`weight_scale`仅含单个元素，shape为$\{1\}$或$\{1, 1\}$；`k_group_size`使用默认值`0`；`out_weight`为$ND$直拷输出，与输入`weight`保持相同的sizes/strides。
    - per-channel：`weight_scale`的shape为$\{N\}$或$\{1, N\}$；`k_group_size`使用默认值`0`；转置输入时`out_weight`为$ND$直拷输出，非转置输入时`out_weight`为$FRACTAL\_NZ\_C0\_8$输出，storage shape为$\{ceil\_div(N, 16), ceil\_div(K, 16), 16, 8\}$（N块在前）。
    - per-group：`weight_scale`的shape为$\{G, N\}$，其中$G$为group数量且$G > 1$；`k_group_size`必须大于`0`，表示K维per-group的group大小；输出路由与per-channel场景一致。

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

- MM_A16S4数据流场景（per-channel，非转置输入，输出$FRACTAL\_NZ\_C0\_8$）

  ```python
  import torch
  import torch_npu
  import numpy as np

  # MM_A16S4 数据流示例（per-channel，非转置输入，输出 NZ 格式）
  k, n = 256, 128

  # weight: INT4，每个 uint8 打包2个INT4数据，沿N维打包
  # logical shape: {K, N}, packed shape: {K, N/2}
  weight_int4 = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int8)
  w = (weight_int4.numpy() & 0xF).astype(np.uint8)
  weight = torch.from_numpy((w[:, 1::2] << 4) | w[:, 0::2]).npu()

  # weight_scale: float16, per-channel, shape: {N}
  weight_scale = torch.randn((n,), dtype=torch.float16).npu()

  out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
      weight,
      weight_scale,
      x_dtype=torch.float16,
      weight_dtype=torch_npu.int4,
      weight_scale_dtype=torch.float16
  )

  # out_weight 可直接输入 npu_weight_quant_batchmatmul 计算
  x = torch.randn((16, k), dtype=torch.float16).npu()
  npu_out = torch_npu.npu_weight_quant_batchmatmul(x, out_weight, out_weight_scale, weight_dtype=torch_npu.int4)
  ```
