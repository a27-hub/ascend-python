# 测试模式参考

## 模式选择

| 算子类型 | 推荐模式 | 参考文件 |
|----------|----------|----------|
| 自定义 NPU 算子（npu_xxx），有明确数学定义 | supported/custom 双路对比 | `test_npu_ffn.py`, `test_npu_add_rms_norm.py` |
| PyTorch 原生算子的 NPU 实现 | cpu/npu 双路对比 | `test_abs.py`, `test_npu_addmm.py` |

---

## 模式一：supported/custom 双路对比（自定义算子）

适用于 `torch_npu.npu_xxx` 算子，用 Python/NumPy 实现参考计算，与 NPU 算子输出对比。

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
        # 用 numpy 或 torch CPU 实现算子逻辑
        result = x @ weight  # 示例
        return result

    def custom_op_exec(self, x, weight, *, optional_param=None):
        """调用 NPU 算子，返回 CPU numpy 结果"""
        out = torch_npu.npu_operator_name(x, weight, optional_param=optional_param)
        return out.cpu().numpy()  # 若 supported_op_exec 返回 numpy，则此处也转 numpy

    @SupportedDevices(['Ascend910B'])
    def test_npu_operator_name_basic(self, device="npu"):
        torch.manual_seed(0)
        x = torch.randn(8, 32, dtype=torch.float16).npu()
        weight = (torch.randn(32, 64, dtype=torch.float16) * 0.01).npu()
        x_clone = x.clone()
        weight_clone = weight.clone()

        supported_output = self.supported_op_exec(x, weight)
        custom_output = self.custom_op_exec(x_clone, weight_clone)
        # 验证输入未被修改（in-place 检查）
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_operator_name_fp32(self, device="npu"):
        torch.manual_seed(0)
        x = torch.randn(8, 32, dtype=torch.float32).npu()
        weight = torch.randn(32, 64, dtype=torch.float32).npu()

        supported_output = self.supported_op_exec(x, weight)
        custom_output = self.custom_op_exec(x.clone(), weight.clone())
        self.assertRtolEqual(supported_output, custom_output, 0.0001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_operator_name_optional_param(self, device="npu"):
        """测试可选参数分支"""
        torch.manual_seed(0)
        x = torch.randn(8, 32, dtype=torch.float16).npu()
        weight = (torch.randn(32, 64, dtype=torch.float16) * 0.01).npu()
        optional = torch.ones(64, dtype=torch.float16).npu()

        supported_output = self.supported_op_exec(x, weight, optional_param=optional)
        custom_output = self.custom_op_exec(x.clone(), weight.clone(), optional_param=optional.clone())
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()
```

### 多输出算子（如 npu_add_rms_norm）

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

### numpy 输入算子（int8/量化场景）

```python
import numpy as np

# numpy 数组直接传给 supported_op_exec，转 tensor 后传给 custom_op_exec
x_np = np.random.randint(-128, 127, size=(8, 32), dtype=np.int8)
x_npu = torch.from_numpy(x_np).npu()
```

---

## 模式二：cpu/npu 双路对比（PyTorch 原生算子）

适用于 `torch.xxx` 在 NPU 上的实现，直接对比 CPU 和 NPU 输出。

```python
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestOperatorName(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.op_name(input1)
        return output.numpy()

    def npu_op_exec(self, input1):
        output = torch.op_name(input1)
        return output.to("cpu").numpy()

    def test_op_fp16(self, device="npu"):
        format_list = [0, 3]
        shape_list = [[5], [5, 10], [1, 3, 2]]
        shape_format = [[np.float16, i, j] for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_input = cpu_input.to(torch.float32)  # fp16 算子 CPU 参考用 fp32
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_op_fp32(self, device="npu"):
        format_list = [0, 3]
        shape_list = [[5], [5, 10], [1, 3, 2]]
        shape_format = [[np.float32, i, j] for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
```

---

## 测试用例覆盖清单

为每个算子至少覆盖以下场景：

- [ ] 基础功能（主要数据类型：fp16）
- [ ] 其他支持的数据类型（fp32、bfloat16、int8 等）
- [ ] 可选参数为 None（默认值）
- [ ] 可选参数有值（各分支）
- [ ] 边界 shape（最小维度、最大维度）
- [ ] 量化/伪量化场景（如适用）

## 常用断言

```python
self.assertRtolEqual(expected, actual, 0.001)        # 相对误差，fp16
self.assertRtolEqual(expected, actual, 0.0001)       # 相对误差，fp32
self.assertRtolEqual(a, b, prec=1e-3, prec16=1e-3)  # 分别指定 fp32/fp16 精度
```
