# Pattern 1: supported/custom Dual-Path Comparison

For `torch_npu.npu_xxx` custom operators — implement reference computation in Python/NumPy and compare against NPU operator output.

## Basic Structure

```python
class TestNpuOperatorName(TestCase):

    def supported_op_exec(self, x, weight, *, optional_param=None):
        """Python/NumPy reference implementation, computes expected result on CPU"""
        result = x @ weight  # example: implement operator logic using torch CPU or numpy
        return result.numpy()

    def custom_op_exec(self, x, weight, *, optional_param=None):
        """Call NPU operator, return CPU numpy result"""
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
```

## Multiple Output Operators

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

## Quantization / int8 Scenarios

```python
import numpy as np

# Pass numpy arrays to supported_op_exec; convert to tensor for custom_op_exec
x_np = np.random.randint(-128, 127, size=(8, 32), dtype=np.int8)
x_npu = torch.from_numpy(x_np).npu()
```

## Reference Examples

- `test/test_custom_ops/test_npu_ffn.py`
- `test/test_custom_ops/test_npu_add_rms_norm.py`
- `test/test_custom_ops/test_npu_dynamic_quant.py`
