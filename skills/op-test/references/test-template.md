# Complete Test Template

## Standard Imports

```python
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
```

## Basic Standards

- Test class inherits `TestCase`, imported from `torch_npu.testing.testcase`
- Use `@SupportedDevices(['Ascend910B'])` decorator
- Use `torch.manual_seed(0)` to fix random seed for reproducibility
- Prefer `assertRtolEqual` for precision assertions: `0.001` for fp16, `0.0001` for fp32
- Add `if __name__ == "__main__": run_tests()` at end of file

## Complete Template (Pattern 1: Custom Operators)

```python
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuOperator(TestCase):

    def supported_op_exec(self, x, *, optional_param=None):
        """CPU reference implementation"""
        # Implement operator logic using numpy/torch CPU
        result = x  # example
        return result.numpy() if isinstance(result, torch.Tensor) else result

    def custom_op_exec(self, x, *, optional_param=None):
        """NPU operator call"""
        out = torch_npu.npu_operator(x, optional_param=optional_param)
        return out.cpu().numpy()

    @SupportedDevices(['Ascend910B'])
    def test_basic_functionality(self, device="npu"):
        """Test basic functionality"""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()

        supported_output = self.supported_op_exec(x.cpu())
        custom_output = self.custom_op_exec(x.clone())
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_dtype_bfloat16(self, device="npu"):
        """Test bfloat16 data type"""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.bfloat16).npu()

        supported_output = self.supported_op_exec(x.cpu())
        custom_output = self.custom_op_exec(x.clone())
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_with_optional_param(self, device="npu"):
        """Test optional parameter"""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        optional_param = torch.randn((256,), dtype=torch.float16).npu()

        supported_output = self.supported_op_exec(x.cpu(), optional_param=optional_param.cpu())
        custom_output = self.custom_op_exec(x.clone(), optional_param=optional_param.clone())
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_min_dimension(self, device="npu"):
        """Test minimum dimension (2D)"""
        torch.manual_seed(0)
        x = torch.randn((2, 256), dtype=torch.float16).npu()

        supported_output = self.supported_op_exec(x.cpu())
        custom_output = self.custom_op_exec(x.clone())
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_invalid_dimension(self, device="npu"):
        """Test invalid dimension (1D should raise error)"""
        x = torch.randn((256,), dtype=torch.float16).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_operator(x)


if __name__ == "__main__":
    run_tests()
```

## Quality Checklist

- [ ] Import paths correct: `SupportedDevices` from `torch_npu.testing.common_utils`
- [ ] All test methods have `@SupportedDevices(['Ascend910B'])` decorator
- [ ] All test methods have `device="npu"` parameter
- [ ] All test methods set random seed `torch.manual_seed(0)`
- [ ] Prefer `assertRtolEqual` over `assertEqual` for numerical comparison
- [ ] Tests cover all supported data types
- [ ] Tests cover all optional parameters (both None and with-value cases)
- [ ] Tests cover boundary conditions (min/max dimensions)
- [ ] Tests include error handling (invalid inputs should raise RuntimeError)
- [ ] File ends with `if __name__ == "__main__": run_tests()`
- [ ] Tests can run independently
- [ ] Test method names are clear and have docstrings
