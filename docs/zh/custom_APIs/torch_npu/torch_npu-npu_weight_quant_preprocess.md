# torch\_npu.npu\_weight\_quant\_preprocess

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- API功能：该接口用于完成伪量化Matmul（包括WeightQuantBatchMatmulV2、QuantBatchMatmulV4、GroupedMatmul-伪量化）的权重预处理。主要将`weight`从$ND$格式转换为$FRACTAL\_NZ$格式，并在需要时对`weight_scale`、`weight_offset`、`bias`进行同步处理。
- 计算公式：

    本接口为图外前处理逻辑，不涉及具体计算公式，也不进入模型计算图。需在进入图模式前完成权重预处理，预处理后的输出张量用于后续的量化Matmul计算。

    支持2种数据流：

    - **MM_MX_A8W4**：Matmul MxA8W4，适用于QuantBatchMatmulV4。
    - **GMM_MX_A8W4**：GroupedMatmul MxA8W4，适用于GroupedMatmul-伪量化。

## 函数原型

```python
torch_npu.npu_weight_quant_preprocess(
    weight,
    weight_scale,
    x_dtype,
    weight_dtype,
    weight_scale_dtype,
    weight_offset=None,
    bias=None,
    x_scale_dtype=None,
    k_group_size=0
) -> Tuple[Tensor, Tensor, Tensor, Tensor]
```

## 参数说明

- **weight**（`Tensor`）：**必选参数**，Matmul的权重矩阵，支持非连续`Tensor`（transposed）。
  - <term>Ascend 950PR/Ascend 950DT</term>：逻辑数据类型仅支持`float4_e2m1fn_x2`（对应`ACL_FLOAT4_E2M1`，使用`uint8`承载），数据格式支持$ND$。支持2维或3维输入，逻辑shape分别为$(K, N)$、$(G, K, N)$，其中$G$表示group数量。1个`uint8`元素打包2个FP4数据，因此`weight`的物理K维长度为$K/2$。
- **weight\_scale**（`Tensor`）：**必选参数**，权重的反量化scale参数，支持非连续`Tensor`（transposed）。
  - <term>Ascend 950PR/Ascend 950DT</term>：逻辑数据类型仅支持`float8_e8m0fnu`（对应`ACL_FLOAT8_E8M0`，使用`uint8`承载），数据格式支持$ND$、$NCL$。支持3维或4维输入，shape分别为$(ceil\_div(K, 64), N, 2)$、$(G, ceil\_div(K, 64), N, 2)$。
- **x\_dtype**（`int`）：**必选参数**，Matmul的激活矩阵的数据类型，对应aclDataType的整数值。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持`torch.float8_e4m3fn`（对应`ACL_FLOAT8_E4M3FN`）。
- **weight\_dtype**（`int`）：**必选参数**，用于指定`weight`中物理承载数据的逻辑数据类型，对应aclDataType的整数值。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持`torch.float4_e2m1fn_x2`（对应`ACL_FLOAT4_E2M1`）。
- **weight\_scale\_dtype**（`int`）：**必选参数**，用于指定`weight_scale`中物理承载数据的逻辑数据类型，对应aclDataType的整数值。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持`torch.float8_e8m0fnu`（对应`ACL_FLOAT8_E8M0`）。
- **weight\_offset**（`Tensor`）：**可选参数**，权重的反量化offset参数，默认值为`None`。
  - <term>Ascend 950PR/Ascend 950DT</term>：当前版本不支持此参数，请传入`None`。
- **bias**（`Tensor`）：**可选参数**，Matmul的偏置矩阵，必须为连续的`Tensor`。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$。支持1维或2维输入，shape为$(N)$、$(1, N)$或$(G, N)$。
- **x\_scale\_dtype**（`int`）：**可选参数**，激活的量化scale参数的数据类型，对应aclDataType的整数值，默认值为`None`。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持`torch.float8_e8m0fnu`（对应`ACL_FLOAT8_E8M0`）。
- **k\_group\_size**（`int`）：**可选参数**，权重在per-group量化时K维度的group大小，默认值为`0`。
  - <term>Ascend 950PR/Ascend 950DT</term>：当前仅支持`32`。

## 返回值说明

`Tuple[Tensor, Tensor, Tensor, Tensor]`

返回四个`Tensor`：

- **out\_weight**（`Tensor`）：预处理后的`weight`，数据类型与输入`weight`相同，数据格式为$FRACTAL\_NZ\_C0\_16$。FP4的逻辑C0长度为32，PyTorch层使用1个`int8`或`uint8`元素打包2个FP4数据，因此物理存储的C0长度为16。
- **out\_weight\_scale**（`Tensor`）：预处理后的`weight_scale`，数据类型与输入`weight_scale`相同。
- **out\_weight\_offset**（`Tensor`）：预处理后的`weight_offset`，当前版本返回空`Tensor`。
- **out\_bias**（`Tensor`）：预处理后的`bias`，数据类型与输入`bias`相同（若提供了`bias`）。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口仅支持单算子模式，需在进入图模式前调用，不进入模型计算图。
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

- 单算子模式调用
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
