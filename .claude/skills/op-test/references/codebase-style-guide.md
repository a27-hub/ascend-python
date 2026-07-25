# test_custom_ops Code Style Reference

> Distilled from 227 real test cases in `test/test_custom_ops/`. **MUST read before writing UT in step 4** to keep generated cases consistent with the repo's existing style. All items below are the repo's **mainstream conventions** (by frequency), not personal preference.

## 0. Golden Iron Rule (violation always fails)

**Before comparing NPU operator output against a CPU/numpy reference, the NPU side MUST be moved back to CPU.**

Consolidate device movement at the `*_op_exec` exit — **do not scatter it across every assertion line**:

```python
def npu_op_exec(self, x):
    x = x.npu()
    out = torch_npu.npu_xxx(x)
    return out.cpu().numpy()        # <- move once at exit, return numpy
```

Counter-example (a pitfall hit in this repo): `assert_close(cpu_ref, npu_out)` — `torch.testing.assert_close` defaults to `check_device=True`, so CPU×NPU reports a device mismatch. **Root cause: wrong assertion function + no device move at exit.**

Operations involving bit-view / INT4 unpacking / `.view(dtype)` also MUST `.cpu()` first (NPU does not support views on internal format).

## 1. Assertion Function: use `self.assertRtolEqual` (724x), not `assert_close` (39x)

`assertRtolEqual` is the repo standard, with dedicated handling for NPU tolerance / NaN co-location / dim reduction:

```python
self.assertRtolEqual(cpu_output, npu_output)            # default prec=1e-4, prec16=1e-3
self.assertRtolEqual(cpu_output, npu_output, 0.001)     # explicit threshold
```

- Numerical precision comparison → `assertRtolEqual` (accepts numpy or tensor)
- Shape / dtype / scalar equality → `assertEqual` (327x)
- Negative exceptions → `assertRaisesRegex` (33x) preferred, fall back to `assertRaises` (19x)

**Common precision thresholds** (by frequency): `0.001` (fp16, mainstream) > `0.0001` (fp32) > `0.005` > `0.01` (quant/matmul). bf16 relaxed to `0.004`.

## 2. Standard Imports

```python
import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
# When needing paired cpu+npu tensors: from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
```

Fixed file footer:
```python
if __name__ == "__main__":
    run_tests()
```

## 3. Class and Method Naming

- Class name `class TestXxx(TestCase)`, inheriting `torch_npu.testing.testcase.TestCase`
- Two method-naming schemes coexist; **pick one and keep it consistent within a file**:
  - `supported_op_exec` (66) / `custom_op_exec` (69) — torch-style dual path
  - `cpu_op_exec` (31) / `npu_op_exec` (37) — numpy-golden style
- Data generation: `generate_input` / `generate_data_xxx`; complex inputs can be wrapped in a `DataInfo` class holding shape/dtype/range

## 4. Case Decorators and Seed

- Every `test_` method gets `@SupportedDevices([...])`, by target platform:
  - `['Ascend910B']` (351, default) / `['Ascend950']` (173) / `['Ascend910B', 'Ascend950']` / `['Ascend310P']`
- Fix randomness: `torch.manual_seed(0)` (mainstream) or `np.random.seed(0)`
- Method signature conventionally carries `device="npu"` (common in dual-path style)

## 5. Two Mainstream Skeletons

### Style A — supported/custom dual path (torch reference)

```python
class TestNpuXxx(TestCase):
    def supported_op_exec(self, x):
        """CPU reference: torch/numpy implementation, returns CPU result"""
        return ref.cpu()        # or .numpy()

    def custom_op_exec(self, x):
        out = torch_npu.npu_xxx(x)
        return out.cpu()        # move device at exit

    @SupportedDevices(['Ascend910B'])
    def test_basic(self, device="npu"):
        torch.manual_seed(0)
        x = torch.randn([32, 256], dtype=torch.float16).npu()
        self.assertRtolEqual(self.supported_op_exec(x.cpu()), self.custom_op_exec(x.clone()))
```

### Style B — numpy-golden (quantization / no torch built-in, recommended)

```python
class TestNpuXxx(TestCase):
    def cpu_op_exec(self, x_np):
        """numpy reference implementation, all on CPU"""
        ...
        return y_np.astype("int8")

    def npu_op_exec(self, x):
        x = x.npu()
        out = torch_npu.npu_xxx(x)
        return out.cpu().numpy()    # move device at exit + to numpy

    @SupportedDevices(['Ascend910B'])
    def test_npu_xxx(self):
        x = self.generate_data(-1, 1, (16, 128), np.float16)
        self.assertRtolEqual(self.cpu_op_exec(x.numpy()), self.npu_op_exec(x))
```

## 6. Negative Cases

```python
@SupportedDevices(['Ascend910B'])
def test_xxx_invalid_param(self, device="npu"):
    x = torch.randn((32, 256), dtype=torch.bfloat16).npu()
    with self.assertRaisesRegex(RuntimeError, "actual_seq_lengths must be specified"):
        torch_npu.npu_xxx(x, bad_param=None)
```

- Specific error string obtainable → `assertRaisesRegex(RuntimeError, "<fragment>")`; otherwise fall back to `assertRaises(RuntimeError)`
- Negative cases also need `@SupportedDevices` (if they rely on NPU validation); pure param-validation cases may use `@unittest.skipIf(not torch.npu.is_available(), ...)`

## 7. Self-Check List (verify each after writing)

- [ ] All NPU outputs `.cpu()` before comparison (consolidated at exit, not scattered)
- [ ] Numerical comparison uses `assertRtolEqual`, not `assert_close`
- [ ] **Every value returned by the reference enters assertRtolEqual (if you computed cpu_y/cpu_s, both MUST be value-compared; feeding only shape is forbidden)**
- [ ] **Integer quant (int8/int32) main-function cases compare values, not downgraded to pure shape/dtype checks; fp8/int4 structure-only assertions (no golden) are noted in docstring**
- [ ] bit-view/INT4 unpacking `.cpu()` first
- [ ] Every `test_` has `@SupportedDevices`
- [ ] `manual_seed` is set
- [ ] Method naming consistent within file (supported/custom or cpu/npu, pick one)
- [ ] File footer has `if __name__ == "__main__": run_tests()`
