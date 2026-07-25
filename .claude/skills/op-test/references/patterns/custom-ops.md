# Pattern 1: supported/custom Dual-Path Comparison

For `torch_npu.npu_xxx` custom operators — implement reference computation in Python/NumPy and compare against NPU operator output.

## 参考实现来源

根据算子分类选择参考实现方式（从 CHECKLIST.md「算子分类」判定）：

| 算子分类 | 参考实现来源 | 说明 |
|----------|-------------|------|
| 有 PyTorch 对应内置函数的算子（如 silu/gelu/relu/rms_norm） | PyTorch 内置函数 | 参考实现使用 `torch.nn.functional.xxx`，避免手动公式引入精度差导致假阳失败 |
| 量化算子（quantization/反量化/dynamic_quant） | NumPy 手动实现 | 量化/反量化无 PyTorch 内置函数，需手动实现 round/clamp 等 |
| 无 PyTorch 对应函数且非量化 | NumPy 手动实现 | 需根据算子文档手动实现参考逻辑 |

## 基本结构

```python
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuOperatorName(TestCase):

    def supported_op_exec(self, x, weight, *, optional_param=None):
        """Python/NumPy 参考实现，在 CPU 上计算期望结果"""
        result = x @ weight  # 示例：使用 torch CPU 或 numpy 实现算子逻辑
        return result.numpy()

    def custom_op_exec(self, x, weight, *, optional_param=None):
        """调用 NPU 算子，返回 CPU numpy 结果"""
        out = torch_npu.npu_operator_name(x, weight, optional_param=optional_param)
        return out.cpu().numpy()

    @SupportedDevices(['Ascend910B'])
    def test_npu_operator_name_basic(self, device="npu"):
        torch.manual_seed(0)
        x = torch.randn(8, 32, dtype=torch.float16).npu()
        weight = (torch.randn(32, 64, dtype=torch.float16) * 0.01).npu()
        x_clone = x.clone()
        weight_clone = weight.clone()

        supported_output = self.supported_op_exec(x.cpu(), weight.cpu())
        custom_output = self.custom_op_exec(x_clone, weight_clone)
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()
```

## 多输出算子

```python
def custom_op_exec(self, x1, x2, gamma, epsilon=1e-6):
    y, rstd, x = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon)
    return y.cpu().numpy(), rstd.cpu().numpy(), x.cpu().numpy()

def test_xxx(self):
    ...
    out0, out1, out2 = self.supported_op_exec(...)
    npu0, npu1, npu2 = self.custom_op_exec(...)
    self.assertRtolEqual(out0, npu0, 0.001)
    self.assertRtolEqual(out1, npu1, 0.001)
    self.assertRtolEqual(out2, npu2, 0.001)
```

## 参考实现 dtype 处理（bf16/fp16）

CPU 参考全程 float32，返回 float32 numpy；NPU 侧也转 float32 对齐。两条硬规则：

1. **bf16 tensor 不能 `.numpy()`** → `TypeError: Got unsupported ScalarType BFloat16`。禁止末尾 `.to(x.dtype).numpy()` 转回窄类型。
2. **`assertRtolEqual` 两侧 dtype 须一致**，否则 `dtype error`（与数值无关）。

窄类型精度差由阈值兜底（bf16 `0.004`、fp16 `0.001`）。可选 Tensor 入参（bias 等）参考侧也须 `.cpu()`，否则 `can't convert npu:0 ... to numpy`。

```python
def supported_op_exec(self, x, bias=None):
    x = x.cpu().to(torch.float32).numpy()      # 入口转 float32
    if bias is not None:
        bias = bias.cpu().to(torch.float32).numpy()
    return y.astype(np.float32)                 # 不要 .to(bf16).numpy()

def custom_op_exec(self, x):
    return torch_npu.npu_xxx(x).cpu().to(torch.float32).numpy()
```

## 量化 / int8 场景（手动参考实现）

量化算子无 PyTorch 内置函数，参考实现需手动编写 round/clamp 等：

```python
import numpy as np

def supported_op_exec(self, x, scale, *, dst_type=None):
    """NumPy 手动参考实现"""
    x_float = x.numpy().astype(np.float32)
    scaled = x_float / scale.numpy()
    if dst_type == "INT8":
        result = np.clip(np.round(scaled), -128, 127).astype(np.int8)
    elif dst_type == "INT4":
        result = np.clip(np.round(scaled), -8, 7).astype(np.int8)  # INT4 packed in INT8
    return result

# 传入 numpy 数组给 supported_op_exec；转为 tensor 给 custom_op_exec
x_np = np.random.randint(-128, 127, size=(8, 32), dtype=np.int8)
x_npu = torch.from_numpy(x_np).npu()
```

## 参考示例

- `test/test_custom_ops/test_npu_ffn.py`
- `test/test_custom_ops/test_npu_add_rms_norm.py`
- `test/test_custom_ops/test_npu_dynamic_quant.py`
