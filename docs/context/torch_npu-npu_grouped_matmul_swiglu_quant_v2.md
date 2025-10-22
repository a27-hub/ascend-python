# torch_npu.npu_grouped_matmul_finalize_routing

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>    | √  |

## 功能说明<a name="zh-cn_topic_0000002259402011_section14441123234220"></a>

aclnnGroupedMatmulV4、aclnnDynamicDequant、aclnnSwigluQuant融合, deepseek模型使用，对比小算子做性能优化。

## 函数原型<a name="zh-cn_topic_0000002259402011_section45077512311"></a>

```
torch_npu.npu_grouped_matmul_swiglu_quant_v2(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale, Tensor group_list, *, Tensor? smooth_scale=None, Tensor[]? weight_assist_matrix=None, Tensor? bias=None, int? dequant_mode=0, int? dequant_dtype=0, int? quant_mode=0, int? quant_dtype=0, int? group_list_type=0, int[]? tuning_config=None) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002259402011_section112637101241"></a>

- **x**（`Tensor`）：输入，左矩阵，公式中的X，Device侧的aclTensor。shape支持2维，数据类型支持`int8`，数据格式支持ND，支持非连续的Tensor。
- **weight**（`TensorList`）：输入，权重矩阵，公式中的W，Device侧的aclTensor。shape支持5维，数据类型支持`int8`，数据格式支持FRACTAL_NZ，支持非连续的Tensor，需注意该接口会将weight的数据格式强制视为FRACTAL_NZ格式。
- **group\_list** （`Tensor`）：输入，指示每个分组参与计算的Token个数，公式中的grouplist，Device侧的aclTensor。shape支持1维，长度需与weight的首轴维度相等，数据类型支持`int64`，数据格式支持ND，支持非连续的Tensor，目前默认支持- **count类型。
- **weight\_scale** （`TensorList`）：输入，右矩阵的量化因子，公式中的w_scale，Device侧的aclTensor。shape支持2维，首轴长度需与weight的首轴维度相等，尾轴长度需要与weight还原为ND格式的尾轴相同，数据类型支持FLOAT，数据格式支持- **ND，支持非连续的Tensor。
- **x\_scale** （`Tensor`）：输入，左矩阵的量化因子，公式中的x_scale，Device侧的aclTensor。shape支持1维，长度需与x的首轴维度相等，数据类型支持`float`，数据格式支持ND，支持非连续的Tensor。
- **bias**（`可选，暂不支持，Tensor`）：输入，矩阵乘计算的偏移值，公式中的bias，shape支持2维，数据类型支持`int32`，预留输入，暂不支持。
- **weight\_assist\_matrix**（`可选，暂不支持，TensorList`）：输入，右矩阵的辅助矩阵，数据类型支持`float32`，预留输入，暂不支持
- **smoothScale**（`可选，暂不支持，Tensor`）：输入，量化的smooth_scales,数据类型为`float32`，预留输入，暂不支持
- **dequant\_dtype**（`可选，暂不支持，int`）：输入设置参数，0：pertoken 1：pergroup，数据类型为`int32`，预留输入，暂不支持
- **dequant\_mode**（`可选，暂不支持，int`）：输入设置参数，0：左pertoken，右perchannel 1：左pertoken，右pergroup，数据类型为`int32`，预留输入，暂不支持
- **quant\_dtype**（`可选，暂不支持，int`）：输入设置参数，量化后低比特数据类型。0：int8；1：fp8；2：E5M2；3：E4M3，数据类型为`int32`，预留输入，暂不支持
- **quant\_mode**（`可选，暂不支持，int`）：输入设置参数，siglu后的量化模式。0：pertoken 1：perchannel，数据类型为`int32`，预留输入，暂不支持
- **groupListType**（`可选，暂不支持，int`）：输入设置参数，grouplist的输入类型。0：cunsum 1：count，数据类型为`int32`，预留输入，暂不支持
- **tuningConfig**（`可选，Listint`）：输入设置参数，数组中的第一个元素表示各个专家处理的token数的预期值

## 返回值说明<a name="zh-cn_topic_0000002259402011_section222314121314"></a>

`Tensor`

output：输出的量化结果，公式中的Q，Device侧的aclTensor。数据类型支持`int8`，shape支持2维，Device侧的aclTensor。数据格式支持ND，支持非连续的Tensor。
output_scale：输出的量化因子，公式中的Q_scale，Device侧的aclTensor。数据类型支持`float`，shape支持1维，Device侧的aclTensor。数据格式支持ND，支持非连续的Tensor。

## 约束说明<a name="zh-cn_topic_0000002259402011_section12345532434255"></a>

-   该接口支持推理和训练场景下使用。
-   该接口支持图模式。
-   输入和输出Tensor支持的数据类型组合如下：
    |x|weight|group_list|weight_scale|x_scale|bias|weight_assit_matrix|smooth_scale|y|y_scale|
    |--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
    |`int8`|`int8`|`int64`|`float32`|`float`|`int32`|`float32`|`float32`|`int8`|`float`|

## 调用示例<a name="zh-cn_topic_0000002259402011_section12312421424213"></a>

-   单算子模式调用

    ```python
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax

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
     
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
     
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, weightscale, xscale, group_list):
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weightscale, xscale, group_list)
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
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
     
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu())
    ```

