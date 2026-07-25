# Pattern 2: cpu/npu Dual-Path Comparison

For NPU implementations of `torch.xxx` native operators — directly compare CPU and NPU outputs.

## Basic Structure

UT 规范同 custom 算子：每个测试方法必须有 `@SupportedDevices(['Ascend910B'])` 装饰器和 `torch.manual_seed(0)`。

```python
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestOperatorName(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.op_name(input1)
        return output.numpy()

    def npu_op_exec(self, input1):
        output = torch.op_name(input1)
        return output.to("cpu").numpy()

    @SupportedDevices(['Ascend910B'])
    def test_op_fp16(self, device="npu"):
        torch.manual_seed(0)
        format_list = [0, 3]
        shape_list = [[5], [5, 10], [1, 3, 2]]
        shape_format = [[np.float16, i, j] for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_input = cpu_input.to(torch.float32)  # use fp32 for CPU reference of fp16 ops
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
```

## Reference Examples

- `test/test_base_ops/test_abs.py`
- `test/test_base_ops/test_npu_addmm.py`
