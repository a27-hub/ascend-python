# torch\_npu.npu\_weight\_quant\_preprocess<a name="ZH-CN_TOPIC_0000002231202137"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>    |    √     |

## 功能说明

- API功能：该接口用于完成伪量化 Matmul（包括 WeightQuantBatchMatmulV2、QuantBatchMatmulV4、GroupedMatmul-伪量化）的权重预处理。主要将 `weight` 从 ND 格式转换为 FRACTAL_NZ 格式，并在需要时对 `weight_scale`、`weight_offset`、`bias` 进行同步处理。
- 计算公式：

    本接口为预处理算子，不涉及具体计算公式。预处理后的输出张量将用于后续的量化 Matmul 计算。

    支持1种数据流：
    - **MM_MX_A8W4**：Matmul MxA8W4，适用于 QuantBatchMatmulV4
    
    **GMM_MX_A8W4**（GroupedMatmul MxA8W4）为预留接口，当前暂不支持。

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
    k_group_size=None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]
```

## 参数说明

- **weight** (`Tensor`)：必选参数。Matmul 的权重矩阵。支持非连续 Tensor（transposed）。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 `int8`、`int4`、`fp8_e4m3`、`hif8`、`fp4_e2m1`。数据格式支持 `ND`。支持输入维度为2维 \(K, N\)（MM_MX_A8W4）。3维 \(G, K, N\)（GMM_MX_A8W4）为预留接口，当前暂不支持。

- **weight\_scale** (`Tensor`)：必选参数。权重的反量化 scale 参数。支持非连续 Tensor（transposed）。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 `float16`、`bfloat16`、`fp8_e8m0`。数据格式支持 `ND`、`NCL`。支持输入维度为3维 \(ceil\_div(K, 64), N, 2\)（MM_MX_A8W4）。4维 \(G, ceil\_div(K, 64), N, 2\)（GMM_MX_A8W4）为预留接口，当前暂不支持。

- **x\_dtype** (`int`)：必选参数。Matmul 的激活矩阵的数据类型，对应 aclDataType 的整数值。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持 `ACL_FLOAT8_E4M3FN`（对应数值请参考 CANN 文档）。

- **weight\_dtype** (`int`)：必选参数。权重矩阵的数据类型，对应 aclDataType 的整数值。需与 `weight` tensor 的实际数据类型一致。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持 `ACL_INT8`、`ACL_INT4`、`ACL_FLOAT8_E4M3FN`、`ACL_HIF8`、`ACL_FLOAT4_E2M1`。

- **weight\_scale\_dtype** (`int`)：必选参数。权重 scale 参数的数据类型，对应 aclDataType 的整数值。需与 `weight_scale` tensor 的实际数据类型一致。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持 `ACL_FLOAT16`、`ACL_BFLOAT16`、`ACL_FLOAT8_E8M0`。

- **weight\_offset** (`Tensor`)：可选参数。权重的反量化 offset 参数。当前 MM_MX_A8W4/GMM_MX_A8W4 数据流不支持，默认值为 None。
    - <term>Ascend 950PR/Ascend 950DT</term>：当前版本不支持此参数，请传入 None。

- **bias** (`Tensor`)：可选参数。Matmul 的偏置矩阵。必须为 contiguous Tensor。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 `float16`、`bfloat16`、`float32`。数据格式支持 `ND`。支持输入维度为1维 \(N\) 或2维 \(1, N\)。

- **x\_scale\_dtype** (`int`)：可选参数。激活的量化 scale 参数的数据类型，对应 aclDataType 的整数值。默认值为 None。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持 `ACL_FLOAT8_E8M0`。

- **k\_group\_size** (`int`)：可选参数。权重在 per-group 量化时 K 维度的 group 的大小。默认值为 0。
    - <term>Ascend 950PR/Ascend 950DT</term>：MM_MX_A8W4/GMM_MX_A8W4 数据流要求 `k_group_size` 为 32。

## 返回值说明<a name="zh-cn_topic_0000001771071862_section22231435517"></a>

`Tuple[Tensor, Tensor, Tensor, Tensor]`

返回四个 Tensor：

- **out\_weight**：预处理后的 weight，数据类型与输入 `weight` 相同，格式为 `FRACTAL_NZ_C0_32`（FP4 场景）或 `FRACTAL_NZ_C0_16`。
- **out\_weight\_scale**：预处理后的 weight\_scale，数据类型与输入 `weight_scale` 相同。
- **out\_weight\_offset**：预处理后的 weight\_offset，当前版本返回空 Tensor。
- **out\_bias**：预处理后的 bias，数据类型与输入 `bias` 相同（若提供了 bias）。

## 约束说明<a name="zh-cn_topic_0000001771071862_section12345537164214"></a>

- 该接口支持推理场景下使用。
- 该接口支持单算子模式、静态图模式、动态图模式、aclgraph 模式。
- **MM_MX_A8W4 数据流**（仅支持<term>Ascend 950PR/Ascend 950DT</term>）：
    - `weight` 数据类型必须为 `FLOAT4_E2M1`，格式为 `ND`，K 必须满足 `K % k_group_size == 0`。
    - `weight` 的 view shape 为 \{K, N\}，storage shape 为 \{N, K\}（transposed），stride 为 \[1, K\]。
    - `weight_scale` 数据类型必须为 `FLOAT8_E8M0`，格式为 `ND`/`NCL`。
    - `weight_scale` 的 view shape 为 \{ceil\_div(K, 64), N, 2\}，storage shape 为 \{N, ceil\_div(K, 64), 2\}（transposed）。
    - `k_group_size` 必须等于 32。
    - `x_dtype` 必须为 `FLOAT8_E4M3FN`。
    - `x_scale_dtype` 必须为 `FLOAT8_E8M0`。
    - 当前不支持 `weight_offset`，必须传入 None。
- **GMM_MX_A8W4 数据流**：预留接口，当前暂不支持。
- 不支持空 Tensor 输入。
- 其余数据类型与 shape 组合为预留接口，当前调用将报错。

## 调用示例<a name="zh-cn_topic_0000001771071862_section14459801435"></a>

- 图模式调用（MM_MX_A8W4 数据流）

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)

    # MM_MX_A8W4 数据流示例
    k = 64
    n = 128

    # weight: FLOAT4_E2M1, transposed (view: {K, N}, storage: {N, K})
    cpu_weight = torch.randint(0, 255, (n, k), dtype=torch.uint8)  # FP4 用 uint8 存储，实际 4 bits
    weight = cpu_weight.npu().as_strided((k, n), (1, k))  # 构造 transposed view

    # weight_scale: FLOAT8_E8M0, 3-D transposed (view: {ceil_div(K,64), N, 2}, storage: {N, ceil_div(K,64), 2})
    cpu_scale = torch.randint(0, 255, (n, k // 64, 2), dtype=torch.uint8)
    weight_scale = cpu_scale.npu().as_strided((k // 64, n, 2), (2, 2, 1))

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, weight, weight_scale):
            return torch_npu.npu_weight_quant_preprocess(
                weight, weight_scale,
                x_dtype=292,  # ACL_FLOAT8_E4M3FN
                weight_dtype=296,  # ACL_FLOAT4_E2M1
                weight_scale_dtype=293,  # ACL_FLOAT8_E8M0
                weight_offset=None,
                bias=None,
                x_scale_dtype=293,  # ACL_FLOAT8_E8M0
                k_group_size=32
            )

    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    out_weight, out_weight_scale, out_weight_offset, out_bias = model(weight, weight_scale)
    ```

**注意**：GMM_MX_A8W4 数据流为预留接口，当前暂不支持，请勿使用3维 weight 输入。

## 父主题

[torch_npu API](./torch_npu_list.md)
