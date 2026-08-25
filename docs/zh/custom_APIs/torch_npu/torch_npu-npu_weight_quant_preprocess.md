# torch_npu.npu_weight_quant_preprocess

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950DT</term>     | √  |

## 功能说明

- API功能：对量化矩阵乘的`weight`及配套量化参数进行预处理，根据输入的数据类型、`weight_scale`的shape和`weight`的转置状态，在内部完成数据流分流，将`weight`转换为[torch\_npu.npu\_weight\_quant\_batchmatmul](torch_npu-npu_weight_quant_batchmatmul.md)计算所需的排布格式（$ND$直拷或$FRACTAL\_NZ$转换），并同步输出处理后的`weight_scale`、`weight_offset`、`bias`。
- 支持的数据流如下：
    - A16W4数据流：`x`数据类型为`float16`/`bfloat16`，`weight`以`uint8`类型承载4-bit紧凑排布的`int4`（A16S4）或`float4_e2m1`（A16F4/MXFP4）数据，每字节打包2个4-bit元素。
    - A8W4数据流：`x`数据类型为`float8_e4m3fn`，`weight`以`uint8`类型承载MXFP4（`float4_e2m1`）数据，scale为MX格式。

## 函数原型

```python
torch_npu.npu_weight_quant_preprocess(weight, weight_scale, x_dtype, weight_dtype, weight_scale_dtype, weight_offset=None, bias=None, x_scale_dtype=None, k_group_size=0) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **weight** (`Tensor`)：必选参数。量化矩阵乘中的右矩阵。数据格式支持$ND$，数据类型支持`uint8`，用于承载4-bit紧凑排布的`int4`/`float4_e2m1`数据，每字节打包2个4-bit元素（低4位为偶数下标元素），要求打包维度的元素个数为偶数。
    - A16W4数据流：支持两维输入，非转置时shape为\(K, N/2\)（沿N维打包），转置时shape为\(K/2, N\)（沿K维打包）。
    - A8W4数据流：矩阵乘场景支持两维转置输入，GroupedMatmul场景支持三维转置输入。

- **weight\_scale** (`Tensor`)：必选参数。`weight`的反量化缩放因子。数据格式支持$ND`。其shape和`weight_scale_dtype`共同决定数据流分流：

    - A16S4 INT4（`weight_dtype`为`torch_npu.int4`）：
        - pertensor模式：shape为\(1,\)或\(1, 1\)，即仅含单个元素。
        - perchannel模式：shape为\(N,\)或\(1, N\)。
        - pergroup模式：shape为\(G, N\)，其中G为group个数且G > 1。
    - A16F4 FP4 pergroup（`weight_dtype`为`torch_npu.float4_e2m1fn_x2`）：shape为\(G, N\)，G > 1，`weight_scale_dtype`为`float16`或`bfloat16`。
    - A16 MXFP4（`weight_dtype`为`torch_npu.float4_e2m1fn_x2`）：`weight_scale_dtype`为`torch_npu.float8_e8m0fnu`，shape为两维\(ceil\_div\(K, 32\), N\)，需为连续Tensor。
    - A8W4 MXFP4：`weight_scale_dtype`为`torch_npu.float8_e8m0fnu`，矩阵乘场景shape为三维转置输入，GroupedMatmul场景shape为四维转置输入。

- **x\_dtype** (`int`)：必选参数。指定后续矩阵乘左矩阵`x`的数据类型。A16W4数据流支持`torch.float16`、`torch.bfloat16`；A8W4数据流支持`torch.float8_e4m3fn`。

- **weight\_dtype** (`int`)：必选参数。指定`weight`的逻辑数据类型。A16S4数据流取值为`torch_npu.int4`；A16F4/MXFP4及A8W4数据流取值为`torch_npu.float4_e2m1fn_x2`。

- **weight\_scale\_dtype** (`int`)：必选参数。指定`weight_scale`的数据类型，取值需与`weight_scale`的实际数据类型一致。

- **weight\_offset** (`Tensor`)：可选参数。`weight`的反量化偏移量，默认值为None。A16S4 INT4各数据流支持透传，其shape需与`weight_scale`保持一致；A16F4/MXFP4及A8W4数据流要求传入None。

- **bias** (`Tensor`)：可选参数。矩阵乘中的偏置项，默认值为None，传入时原样透传至输出。

- **x\_scale\_dtype** (`int`)：可选参数。指定后续矩阵乘左矩阵`x`的量化缩放因子数据类型，默认值为None。A8W4数据流取值为`torch_npu.float8_e8m0fnu`；A16W4数据流要求传入None。

- **k\_group\_size** (`int`)：可选参数。K维的分组大小，默认值为0。MX格式scale场景（A8W4、A16 MXFP4）取值为32。

## 返回值说明

返回4个`Tensor`的元组`(out_weight, out_weight_scale, out_weight_offset, out_bias)`：

- **out\_weight**：预处理后的`weight`。A16W4数据流下，perchannel/pergroup非转置输入及FP4/MXFP4输入输出为$FRACTAL\_NZ$格式（NZ\_C0\_16分形，storage shape为\(ceil\_div\(N, 16\), ceil\_div\(K, 16\), 16, 16\)）；其余场景（pertensor、perchannel/pergroup转置输入）输出为$ND$直拷，与输入`weight`保持相同的sizes/strides，属纯透传。
- **out\_weight\_scale**：预处理后的`weight_scale`，与输入`weight_scale`保持相同的sizes/strides。
- **out\_weight\_offset**：预处理后的`weight_offset`，未传入`weight_offset`时输出空Tensor。
- **out\_bias**：预处理后的`bias`，未传入`bias`时输出空Tensor。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 不支持空Tensor输入。
- 仅支持<term>Ascend 950DT</term>，其他产品调用会报错"Unsupported NPU architecture"。
- A16W4数据流的`weight`只接受`uint8`载体（每字节2个4-bit元素），其他数据类型（如`int8`）会被拒绝。
- 输入的数据类型、`weight_scale` shape与`weight`转置状态不满足任一数据流的要求时，接口报错"Unsupported data flow combination"。
- A16W4数据流下ND直拷输出为纯透传，调用方也可以不经本接口预处理，直接将$ND$格式的`uint8`载体`weight`输入[torch\_npu.npu\_weight\_quant\_batchmatmul](torch_npu-npu_weight_quant_batchmatmul.md)（单算子模式和图模式均支持）。

## 调用示例

- 单算子模式调用
    - A16W4 perchannel场景（INT4，非转置输入，输出NZ格式），仅支持<term>Ascend 950DT</term>。

        ```python
        import torch
        import torch_npu
        import numpy as np

        k, n = 256, 128
        # 逻辑shape (k, n) 的int4数据沿N维打包为uint8载体 (k, n//2)
        weight_int4 = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int8)
        w = (weight_int4.numpy() & 0xF).astype(np.uint8)
        weight = torch.from_numpy((w[:, 1::2] << 4) | w[:, 0::2]).npu()
        weight_scale = torch.randn((n,), dtype=torch.float16).npu()

        out_weight, out_weight_scale, _, _ = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch_npu.int4,
            weight_scale_dtype=torch.float16)

        x = torch.randn((16, k), dtype=torch.float16).npu()
        npu_out = torch_npu.npu_weight_quant_batchmatmul(x, out_weight, out_weight_scale, weight_dtype=torch_npu.int4)
        ```

    - A8W4 MXFP4场景，仅支持<term>Ascend 950DT</term>。

        ```python
        import torch
        import torch_npu

        k, n = 128, 64
        weight = torch.zeros((n, k), dtype=torch.uint8).npu().transpose(0, 1)
        weight_scale = torch.zeros((n, k // 64, 2), dtype=torch.uint8).npu().transpose(0, 1)

        out_weight, out_weight_scale, _, _ = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu,
            k_group_size=32)
        ```
