# torch_npu.npu_grouped_matmul_swiglu_quant_v2

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    | √  |

## 功能说明

`npu_grouped_matmul_swiglu_quant_v2`是一种融合分组矩阵乘法（GroupedMatmul）、SwiGLU混合激活函数、量化（quant）的计算方法。该方法适用于需要对矩阵乘法结果进行SwiGLU激活函数激活的场景，融合算子在底层能够对部分过程并行，达到性能优化的效果。

## 函数原型

```
torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, x_scale, group_list, *, smooth_scale=None, weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0, group_list_type=0, tuning_config=None, x_dtype=None, weight_dtype=None, weight_scale_dtype=None, x_scale_dtype=None) -> (Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选输入，矩阵乘法的左矩阵，对应公式中的`X`。shape支持2维[m,k]，数据类型支持`int8`、`int32`，数据格式支持ND，支持非连续的Tensor。当`x_dtype`指定时，`x`中存储的实际数据将按照指定类型解释。
- **weight**（`TensorList`）：必选输入，权重矩阵(矩阵乘法右矩阵)，对应公式中的`W`。目前仅支持TensorList长度为1。shape支持3维[e,k,n]（ND格式）或5维NZ格式，数据类型支持`int8`、`int32`，数据格式支持ND和FRACTAL_NZ(通过接口npu\_format\_cast，可实现格式转换)，支持非连续的Tensor。当`weight_dtype`指定时，`weight`中存储的实际数据将按照指定类型解释。
- **weight\_scale**（`TensorList`）：必选输入，右矩阵的量化因子，对应公式中的`w_scale`。目前仅支持TensorList长度为1。`weight`数据类型为`int8`时，`weight_scale`的shape支持2维；`weight`数据类型为`int32`时，`weight_scale`的shape支持2维和3维；MX量化场景下支持4维。数据类型支持`float32`、`float16`、`bfloat16`，MX量化场景下通过`weight_scale_dtype`指定为`float8_e8m0`。数据格式支持ND，支持非连续的Tensor。
- **x\_scale**（`Tensor`）：必选输入，左矩阵的量化因子，对应公式中的`x_scale`。非MX场景下shape支持1维[m]，MX量化场景下shape支持3维[m, ceil(k/64), 2]。数据类型支持`float32`，MX量化场景下通过`x_scale_dtype`指定为`float8_e8m0`。数据格式支持ND，支持非连续的Tensor。
- **group\_list**（`Tensor`）：必选输入，指示每个分组参与计算的Token个数，对应公式中的`grouplist`。shape支持1维[e]，长度需与weight的首轴维度相等。数据类型支持`int64`，数据格式支持ND，支持非连续的Tensor。
- **smooth\_scale**（`Tensor`）：可选输入，平滑缩放因子。数据类型为`float32`，数据格式支持ND。仅A4W4场景下需传入，首轴长度需与weight的首轴维度相等，支持两种shape：(E, N/2)或(E,)，当使用(E,)时会进行广播乘法。其他场景传入默认值None。
- **weight\_assist\_matrix**（`TensorList`）：可选输入，右矩阵的辅助矩阵，对应公式中的`weightAssistMatrix`。数据类型支持`float32`，数据格式支持ND，shape支持2维。仅A8W4场景下需传入，首轴长度需与weight的首轴维度相等，尾轴长度需要与weight还原为ND格式的尾轴相同。其他场景传入默认值None。
- **bias**（`Tensor`）：可选输入，矩阵乘计算的偏移值，公式中的bias，shape支持2维，数据类型支持`int32`，预留输入，当前仅支持传入默认值None。
- **dequant\_mode**（`int`）：可选输入，表示反量化模式，数据类型为`int32`，默认值为0。
    - 取值为0时，表示激活矩阵per-token，权重矩阵per-channel。A8W8和A4W4场景下支持。
    - 取值为1时，表示激活矩阵per-token，权重矩阵per-group。A8W4场景下支持。
    - 取值为2时，表示MX量化。MX量化场景下支持。
- **dequant\_dtype**（`int`）：可选输入，表示中间GroupedMatmul结果的数据类型，数据类型为`int32`，默认值为0。
    - A8W8/A8W4/A4W4场景下：默认值0即可，不支持该参数。
    - MX量化场景下：支持传入0（表示FLOAT）。
    - Pertoken量化场景下：支持传入0（FLOAT）、1（FLOAT16）、27（BFLOAT16）。
- **quant\_mode**（`int`）：可选输入，参数表示SwiGLU后的量化模式。数据类型为`int32`，默认值为0。
    - 取值为0时，表示per-token。
    - 取值为1时，表示per-group。
    - 取值为2时，表示MX量化。
- **quant\_dtype**（`int`）：可选输入，参数表示量化后低比特数据类型。数据类型为`int32`，默认值为0（表示`int8`）。MX量化场景或Pertoken量化场景下支持传入`float8_e5m2`、`float8_e4m3`、`hifloat8`等类型对应的枚举值。
- **group\_list\_type**（`int`）：可选输入，参数表示grouplist的输入类型，数据类型为`int32`，默认值为0。
    - 取值为0时，表示cumsum模式，groupList中的每个元素代表当前分组的累计长度。
    - 取值为1时，表示count模式，groupList中的每个元素代表该分组包含多少元素。
- **tuning\_config**（`List[int]`）：可选输入，参数数组中的第一个元素表示各个专家处理的token数的预期值。从第二个元素开始预留，用户无须填写，未来会进行扩展。默认设置为None。
- **x\_dtype**（`int`）：可选输入，表示`x`的实际数据类型，默认值为None。支持`torch_npu.float4_e2m1fn_x2`（MXFP4场景）和`torch_npu.hifloat8`。`x_dtype`和`weight_dtype`必须同时指定或同时为None。
- **weight\_dtype**（`int`）：可选输入，表示`weight`的实际数据类型，默认值为None。支持`torch_npu.float4_e2m1fn_x2`（MXFP4场景）和`torch_npu.hifloat8`。`x_dtype`和`weight_dtype`必须同时指定或同时为None。
- **weight\_scale\_dtype**（`int`）：可选输入，表示`weight_scale`的实际数据类型，默认值为None。仅支持`float8_e8m0`（MX量化场景）。当指定该参数时，算子进入MX量化路径。
- **x\_scale\_dtype**（`int`）：可选输入，表示`x_scale`的实际数据类型，默认值为None。仅支持`float8_e8m0`（MX量化场景）。

## 返回值说明

- **output**（`Tensor`）：输出的量化结果，对应公式中的`Q`。A8W8/A8W4/A4W4场景下数据类型为`int8`，shape支持2维[m, n/2]；MX量化或Pertoken量化场景下数据类型支持`float8_e4m3`、`float8_e5m2`、`float4_e2m1`、`hifloat8`，shape支持2维[m, n/2]。数据格式支持ND，支持非连续的Tensor。
- **output\_scale**（`Tensor`）：输出的量化因子，对应公式中的`Q_scale`。A8W8/A8W4/A4W4及Pertoken量化场景下数据类型为`float`，shape支持1维[m]；MX量化场景下数据类型为`float8_e8m0`，shape支持3维[m, ceil((n/2)/64), 2]。数据格式支持ND，支持非连续的Tensor。

## 约束说明

-   该接口支持推理和训练场景下使用。
-   该接口支持图模式。
-   确定性计算：该接口默认为确定性实现。
-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    -   支持A8W8、A8W4、A4W4量化场景，输入和输出Tensor支持的数据类型组合如下：

        |量化场景|x|weight|weight\_scale|x\_scale|smooth\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|--------|--------|
        |A8W8|`int8`|`int8`|`float32`、`float16`、`bfloat16`|`float32`|-|`int8`|`float`|
        |A8W4|`int8`|`int4`、`int32`|`uint64`|`float32`|-|`int8`|`float`|
        |A4W4|`int4`、`int32`|`int4`、`int32`|`float32`|`float32`|`float32`|`int8`|`float`|

    -   shape约束如下：

        |量化场景|x|weight|weight\_scale|x\_scale|smooth\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|--------|--------|
        |A8W8|(M, K)|NZ格式shape形如{(E, N/32, K/16, 16, 32)}|{(E, N)}|(M,)|-|(M, N/2)|(M,)|
        |A8W4|(M, K)|ND格式{(E, K, N)}或NZ格式|per-channel:{(E, N)}; per-group:{(E, K\_group\_num, N)}|(M,)|-|(M, N/2)|(M,)|
        |A4W4|(M, K)|ND格式{(E, K, N)}或NZ格式|{(E, N)}|(M,)|(E, N/2)或(E,)|(M, N/2)|(M,)|

    -   A8W8场景下，weight仅支持NZ格式（FRACTAL\_NZ），不支持ND数据格式。INT32为A8W4和A4W4场景下的适配用途，实际1个INT32会被解释为8个INT4数据。
    -   A8W4场景下，dequant\_mode支持取值0和1；A8W8和A4W4场景下，dequant\_mode仅支持取值0。
    -   不支持dequant\_dtype和quant\_mode参数（使用默认值即可）。
    -   A8W8场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于65536。
    -   A8W4场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于20000。
    -   A4W4场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于20000。

-   MX量化场景（需指定`weight_scale_dtype`为`float8_e8m0`）：
    -   支持MXFP8和MXFP4量化场景，输入和输出Tensor支持的数据类型组合如下：

        |MX量化场景|x|weight|weight\_scale|x\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|--------|
        |MXFP8|`float8_e4m3`、`float8_e5m2`|`float8_e4m3`、`float8_e5m2`|`float8_e8m0`|`float8_e8m0`|`float8_e4m3`、`float8_e5m2`|`float8_e8m0`|
        |MXFP4|`float4_e2m1`|`float4_e2m1`|`float8_e8m0`|`float8_e8m0`|`float4_e2m1`、`float8_e4m3`、`float8_e5m2`|`float8_e8m0`|

    -   MX量化场景下shape约束如下：

        |x|weight|weight\_scale|x\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|
        |(M, K)|非转置{(E, K, N)}或转置{(E, N, K)}|非转置{(E, ceil(K/64), N, 2)}或转置{(E, N, ceil(K/64), 2)}|(M, ceil(K/64), 2)|(M, N/2)|(M, ceil((N/2)/64), 2)|

    -   MXFP4场景下，由于`float4`无原生PyTorch类型，需要通过`x_dtype`和`weight_dtype`参数指定实际数据类型为`torch_npu.float4_e2m1fn_x2`。MXFP8场景下若tensor已为float8类型则无需指定。`x_dtype`和`weight_dtype`必须同时指定或同时为None。
    -   weight\_scale转置属性需要与weight保持一致。
    -   weight支持转置，仅支持ND格式。
    -   dequant\_mode支持取值2，quant\_mode支持取值2，dequant\_dtype支持取值0。
    -   N需满足128对齐。
    -   MXFP4场景不支持K=2。
    -   MXFP4场景需满足K为偶数；当output的数据类型为`float4_e2m1`时，需满足N为大于等于4的偶数。

-   Pertoken量化场景（MX量化路径下，dequant\_mode=0且quant\_mode=0）：
    -   输入和输出Tensor支持的数据类型组合如下：

        |x|weight|weight\_scale|x\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|
        |`float8_e4m3`、`float8_e5m2`|`float8_e4m3`、`float8_e5m2`|`float32`、`bfloat16`|`float32`|`float8_e4m3`、`float8_e5m2`|`float`|
        |`int8`|`int8`|`float32`、`bfloat16`、`float16`|`float32`|`int8`|`float`|
        |`hifloat8`|`hifloat8`|`float32`、`bfloat16`|`float32`|`hifloat8`|`float`|

    -   Pertoken量化场景下shape约束如下：

        |x|weight|weight\_scale|x\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|
        |(M, K)|非转置{(E, K, N)}或转置{(E, N, K)}|{(E, N)}|(M,)|(M, N/2)|(M,)|

    -   weight支持转置，仅支持ND格式。
    -   dequant\_mode支持取值0，quant\_mode支持取值0。
    -   dequant\_dtype支持取值0（FLOAT）、1（FLOAT16）、27（BFLOAT16）。

## 调用示例

-   单算子模式调用

    ```python
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax
    
    torch.npu.config.allow_internal_format = True
    
    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList    
    E = 2
    M = 512
    K = 7168
    N = 4096
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
    output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu())
    ```

-   图模式调用：

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from scipy.special import softmax
    from torchair.configs.compiler_config import CompilerConfig
    
    torch.npu.config.allow_internal_format = True
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
     
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, weightscale, xscale, group_list, quant_dtype):
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weightscale, xscale, group_list, quant_dtype=quant_dtype)
            return output    
     
    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList    
    E = 2
    M = 512
    K = 7168
    N = 4096
    quant_dtype = 2
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
     
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu(), quant_dtype)
    ```

