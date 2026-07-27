# v1/v2 Variant Precision Comparison

修改模式下，调用同一 aclnn 的变体（v1→v2）必须包含 v1 与 v2 的精度对比测试——验证新变体与旧变体行为一致。

## 何时使用

- op-adaptation 修改模式，新增算子变体（如 `npu_operator_v2`）
- 新变体与旧变体调用同一 aclnn 接口
- 需验证两版本输出精度一致

## 模板代码

```python
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuOperatorV1V2(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_npu_operator_v1_vs_v2(self, device="npu"):
        """v1与v2变体精度对比"""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()

        v1_output = torch_npu.npu_operator_v1(x.clone()).cpu().numpy()
        v2_output = torch_npu.npu_operator_v2(x.clone()).cpu().numpy()
        self.assertRtolEqual(v1_output, v2_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_operator_v1_vs_v2_bfloat16(self, device="npu"):
        """v1与v2变体精度对比（bfloat16）"""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.bfloat16).npu()

        v1_output = torch_npu.npu_operator_v1(x.clone()).cpu().numpy()
        v2_output = torch_npu.npu_operator_v2(x.clone()).cpu().numpy()
        self.assertRtolEqual(v1_output, v2_output, 0.004)  # bf16 尾数 7bit, 阈值放宽


if __name__ == "__main__":
    run_tests()
```

## 注意事项

- 用 `.clone()` 保护输入，避免 v1 调用污染 v2 的输入
- 数值精度用 `assertRtolEqual`：fp16 阈值 0.001，bf16 阈值 0.004（bf16 尾数 7bit vs fp16 10bit，过紧易假阳失败）
- 多输出算子逐输出对比
- 多 dtype 场景至少覆盖 float16 + bfloat16
- 该方法用锚点定位追加到已有测试类（见 modify-mode-guide.md）

## 参考示例

- `test/test_custom_ops/test_npu_add_rms_norm.py`（如有 v1/v2 变体）
