# torch_npu.npu_add_layer_norm

## 功能说明

融合计算 `x1 + x2` 及其 LayerNorm，返回归一化结果、均值、标准差倒数和可选的加法结果。

## 函数原型

```python
torch_npu.npu_add_layer_norm(x1, x2, gamma, beta, epsilon=1e-5, additional_output=False)
```

## 参数说明

- **x1、x2** (`Tensor`)：参与加法的两个输入，shape 相同。
- **gamma、beta** (`Tensor`)：LayerNorm 的缩放和偏移参数。
- **epsilon** (`float`)：加到方差上的数值稳定项，默认值为 `1e-5`。
- **additional_output** (`bool`)：是否返回加法结果，默认值为 `False`。该参数不影响归一化结果、均值和标准差倒数。

## 返回值说明

返回四元组 `(y, mean, rstd, x)`：

- **y**：归一化结果，shape 与 `x1` 相同。
- **mean、rstd**：归一化计算的均值和标准差倒数，数据类型为 `torch.float32`。
- **x**：`additional_output=True` 时为 `x1 + x2`；`False` 时为 shape `(0,)`、无元素存储的空 Tensor。`x` 与 `y` 的设备和数据类型相同，二者不共享存储。

当 `x1` 和 `x2` 的数据类型相同时，`y` 和 `x` 使用该数据类型；支持的混合精度输入使用 `torch.float32`。

关闭额外输出时，第四项仅表示未请求加法结果，不应作为加法结果使用。需要该结果时，请显式传入 `additional_output=True`。

## 验证范围

关闭额外输出的存储优化最低已验证 CANN 版本为 **8.5.0**，同时验证了 CANN 9.2.0。验证硬件为 aarch64 / Ascend 910B3，覆盖 ACLNN 与 aclops 路径，包括 `jit_compile=True` 的单算子编译模式。

该优化依赖后端在 `additional_output=False` 时不写入加法结果输出。上述版本为已验证范围，不代表所有后续版本或其他硬件均已验证；TorchAir 图模式尚未验证。

## 调用示例

```python
import torch
import torch_npu

x1 = torch.randn(16, 128, device="npu")
x2 = torch.randn_like(x1)
gamma = torch.ones(128, device="npu")
beta = torch.zeros_like(gamma)
y, mean, rstd, x = torch_npu.npu_add_layer_norm(x1, x2, gamma, beta)
assert x.numel() == 0

y, mean, rstd, x = torch_npu.npu_add_layer_norm(
    x1, x2, gamma, beta, additional_output=True)
assert x.shape == x1.shape
```
