# torch\_npu.npu\_dual\_level\_quant\_matmul

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- **API功能**：QuantMatmul的二级量化功能，减少精度损失。

- **计算公式**：

  ![](../../figures/zh-cn_formulaimage_0000002548302759.png)

  - x1<sub>ij</sub>、x2<sub>ij</sub>表示输入的左矩阵和右矩阵，i、j表示第i、j个分组；
  - x1Level1Scale为x1的第一级反量化参数，为MX量化，level1GroupSize为32；
  - x2Level1Scale为x2的第一级反量化参数，为MX量化，level1GroupSize为32；
  - x1Level0Scale为x1的第零级反量化参数，为PerGroup量化，level0GroupSize为512；
  - x2Level0Scale为x2的第零级反量化参数，为PerGroup量化，level0GroupSize为512；
  - bias为偏置矩阵。

## 函数原型

```python
torch_npu.npu_dual_level_quant_matmul(x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale, x2_level1_scale, bias=None, output_dtype=torch.bfloat16) -> Tensor
```

## 参数说明

- **x1**（`Tensor`）：必选参数，表示输入的左矩阵，公式中的x1<sub>ij</sub>，2维Tensor，不转置，shape为\[M, K\]。数据格式仅支持ND。数据类型仅支持`float4_e2m1fn_x2`。
- **x2**（`Tensor`）：必选参数，表示输入的右矩阵，公式中的x2<sub>ij</sub>，2维Tensor，转置，shape为\[N, K\]。数据格式仅支持FRACTAL\_NZ。数据类型仅支持`float4_e2m1fn_x2`。
- **x1\_level0\_scale**（`Tensor`）：必选参数，表示左矩阵的第零级反量化参数，公式中的x1Level0Scale，2维Tensor，不转置，shape为\[M, ceil\(K/level0GroupSize\)\]。数据格式仅支持ND。数据类型仅支持`float32`。
- **x2\_level0\_scale**（`Tensor`）：必选参数，表示右矩阵的第零级反量化参数，公式中的x2Level0Scale，2维Tensor，不转置，shape为\[ceil\(K/level0GroupSize\), N\]。数据格式仅支持ND。数据类型仅支持`float32`。
- **x1\_level1\_scale**（`Tensor`）：必选参数，表示左矩阵的第一级反量化参数，公式中的x1Level1Scale，3维Tensor，不转置，shape为\[M, ceil\(K/level1GroupSize\), 2\]。数据格式仅支持ND。数据类型仅支持`float8_e8m0fnu`。
- **x2\_level1\_scale**（`Tensor`）：必选参数，表示右矩阵的第一级反量化参数，公式中的x2Level1Scale，3维Tensor，转置，shape为\[N, ceil\(K/level1GroupSize\), 2\]。数据格式仅支持ND。数据类型仅支持`float8_e8m0fnu`。
- **bias**（`Tensor`）：可选参数，表示输入的偏置矩阵，公式中的bias，1维Tensor，shape为\[N\]。不支持非连续的Tensor。数据格式仅支持ND。数据类型仅支持`float32`。
- **output\_dtype**（`int`）：必选参数，表示输出的类型，可选类型为`torch.float16`/`torch.half`、`torch.bfloat16`，默认为`torch.bfloat16`。

## 返回值说明

**y**（`Tensor`）：表示计算结果，公式中的out，数据类型由`output_dtype`决定。数据格式支持ND。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口仅支持单算子模式调用。
- 所有输入和输出张量不支持空Tensor，不支持非连续的Tensor。
- level1量化类型仅支持MX量化，level1GroupSize=32。
- level0量化类型仅支持PerGroup量化，level0GroupSize仅支持512。

## 调用示例

单算子模式调用

```python
import torch
import torch_npu
import math
class A4W4Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale, x2_level1_scale, bias=None,
                output_dtype=torch.bfloat16):
        return torch_npu.npu_dual_level_quant_matmul(x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale,
                                                     x2_level1_scale, bias=bias, output_dtype=output_dtype)
def main():
    m = 256
    k = 1024
    n = 2048
    l0_group_size = 512
    l1_group_size = 32
    # float4_e2m1fn_x2数据类型可用int8表示
    cpu_x1 = torch.randint(-5, 5, (m, math.ceil(k / 2)), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (n, math.ceil(k / 2)), dtype=torch.int8)
    x1_level0_scale = torch.randint(-5, 5, (m, math.ceil(k / l0_group_size)), dtype=torch.float32)
    x2_level0_scale = torch.randint(-5, 5, (math.ceil(k / l0_group_size), n), dtype=torch.float32)
    # float8_e8m0fnu数据类型可用uint8表示
    x1_level1_scale = torch.randint(124, 130, (m, math.ceil(k / l1_group_size / 2), 2), dtype=torch.uint8)
    x2_level1_scale = torch.randint(124, 130, (n, math.ceil(k / l1_group_size / 2), 2), dtype=torch.uint8)
    x2 = torch_npu.npu_format_cast(cpu_x2.npu(), 29, customize_dtype=cpu_x2.dtype)
    bias = torch.randint(-5, 5, (n,), dtype=torch.float32)
    model = A4W4Net().npu()
    npu_out = model(cpu_x1.npu(), x2.npu(), x1_level0_scale.npu(), x2_level0_scale.npu(), x1_level1_scale.npu(),
                    x2_level1_scale.npu(), bias=bias.npu(), output_dtype=torch.bfloat16)
    print(npu_out.cpu())
if __name__ == '__main__':
    main()
```
