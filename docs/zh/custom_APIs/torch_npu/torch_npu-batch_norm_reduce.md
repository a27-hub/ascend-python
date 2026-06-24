# torch_npu.batch_norm_reduce

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- API功能：按BatchNorm通道维对输入Tensor进行规约，返回通道维上的元素和 `sum` 与平方和 `square_sum` 。
- 以4维NCHW输入 `(N, C, H, W)` 为例，计算C轴以外的N、H、W轴求和与平方和。
- 计算公式：

  $$
  sum_i = \sum_{n=0}^{N-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} input_{(n,i,h,w)}
  $$

  $$
  square\_sum_i = \sum_{n=0}^{N-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} input_{(n,i,h,w)}^2
  $$

## 函数原型

```python
torch_npu.batch_norm_reduce(input, eps) -> (Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，表示待规约的输入Tensor。
- **eps** (`float`)：必选参数，保留参数。该参数不参与 `sum` 和 `square_sum` 的计算。

## 返回值说明

- **sum** (`Tensor`)：输入Tensor按通道维规约后的元素和，为一维 `float32` Tensor，长度为通道维长度。例如，输入按NCHW格式排布且shape为 `(N, C, H, W)` 时，输出shape为 `(C,)` 。
- **square_sum** (`Tensor`)：输入Tensor按通道维规约后的平方和，为一维 `float32` Tensor，shape同 `sum` 。

## 约束说明

- `input` 的维度需大于1。

## 调用示例

```python
import torch
import torch_npu

input = torch.randn(2, 3, 12, 12, dtype=torch.float32, device="npu")

sum_out, square_sum_out = torch_npu.batch_norm_reduce(input, 1e-5)

print("sum_out.shape:", sum_out.shape)
print("sum_out.dtype:", sum_out.dtype)
print("square_sum_out.shape:", square_sum_out.shape)
print("square_sum_out.dtype:", square_sum_out.dtype)
```
