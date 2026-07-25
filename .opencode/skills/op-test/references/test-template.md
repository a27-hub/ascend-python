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
- Prefer `assertRtolEqual` for precision assertions: `0.001` for fp16, `0.004` for bf16 (7-bit mantissa), `0.0001` for fp32
- Add `if __name__ == "__main__": run_tests()` at end of file

## Reference Implementation Guidelines

When implementing CPU reference for precision comparison, prefer PyTorch built-in functions over manual math:

| Formula | Use | Avoid |
|---------|-----|-------|
| Swish/SiLU | `torch.nn.functional.silu(x)` | `x * torch.sigmoid(x)` |
| GELU | `torch.nn.functional.gelu(x)` | manual approximation |
| ReLU | `torch.nn.functional.relu(x)` | `torch.maximum(x, 0)` |
| Sigmoid | `torch.sigmoid(x)` | `1 / (1 + torch.exp(-x))` |
| Tanh | `torch.tanh(x)` | manual formula |

Manual implementations can introduce subtle precision differences that cause false-positive test failures.

## Assertion Value Source

Shape/dtype expected values in assertions are derived from the **Shape Rule** and **Dtype Rule** columns of the CHECKLIST.md **Output specs** table, not guessed. Example: output spec `y: shape=same as x1.shape, dtype=same as x1.dtype` → `self.assertEqual(out.shape, x1.shape)`. Use `assertRtolEqual` for numerical precision against reference implementation output. Use `assertEqual` for shape/dtype assertions.

## 🔒 Assertion Completeness Iron Rule (violation = fake test, hard gate)

**Every value returned by the reference (`supported_op_exec`/`cpu_op_exec`) must be consumed by `assertRtolEqual`. Computing a golden but asserting only shape/dtype = fake test.**

Check each case; failing any rule is a defect.

1. **Consume every return value.** Reference returns `(y, scale)` → the case must have `assertRtolEqual(cpu_y, npu_y)` AND `assertRtolEqual(cpu_s, npu_s)`. Comparing one but not the other = under-assert.
2. **No compute-then-discard.** After `cpu_y, cpu_s = self.cpu_op_exec(...)`, both variables must enter `assertRtolEqual`. Feeding only `cpu_y.shape` to assertEqual and dropping the value is a defect.
3. **Precise golden exists → compare values, never downgrade to shape check.**
   - int8/int32 symmetric quant: golden is exact (deterministic round/clip) → require `assertRtolEqual(cpu_y, npu_y, 1)` (±1 LSB tolerance).
   - int4 (quint4x2): int32-packed. Unpack and compare y using `convert_int4_to_int8` from `test_npu_dynamic_quant_asymmetric.py`. Also compare scale.
   - fp8 (e4m3/e5m2) / hifloat8: y has no numpy golden → assert dtype+shape only, and add docstring note `no golden, structural assertion only`. But scale (float32, from DTYPE_MAX) IS comparable — extend cpu_op_exec's `dtype_max` and compare scale. y being hard does not exempt scale.
     - DTYPE_MAX caveat: vendor fp8 max may differ from IEEE (e4m3: 448 or 240; e5m2: 57344). If unconfirmed, IEEE max as golden gives false positives → docstring note `scale golden pending vendor DTYPE_MAX` and defer. Never skip silently.
4. **Same-family cases must align.** Within `test_dst_type_*`, all cases assert the same dimensions. If int8/f8e4m3/f8e5 check `y.dtype`, hifloat8 must too (meta returns `torch.uint8`). A sibling missing an assertion others have = defect; align to the fullest sibling.
5. **`_shape`/`_dtype` naming does not auto-exempt.** Exemption applies only to outputs with no numpy golden (fp8/int4 y). Any output with a deterministic golden (int8/int32/scale) must compare values regardless of case name. Quant `scale` always has a golden → any case asserting only `scale.shape/dtype` is a defect.
6. **Never skip building a golden.** Rules 1-2 catch "computed but only fed shape"; this catches "never called `*_op_exec`". A main/boundary/large-data case (name not ending in `_shape`/`_dtype`) with no `*_op_exec` call and only `assertEqual(y.shape)` = fake test. Dimension-independent references (e.g. `np.max(axis=-1)`) produce goldens for 3D/large shapes too — no "special shape" excuse. Large-data cases: compare scale at minimum.
7. **Cross-case dedup.** Two cases with equivalent assertion sets (e.g. both verify only INT4 packed shape, neither compares values) → merge into one case with value verification.

> Counter-example (real defect here): `cpu_y, cpu_s = self.cpu_op_exec(x); self.assertEqual(npu_y.shape, cpu_y.shape); self.assertRtolEqual(cpu_s, npu_s)` — `cpu_y` computed but only its shape used; y value unverified. Fix: add `self.assertRtolEqual(cpu_y, npu_y, 1)`.

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
        self.assertRtolEqual(supported_output, custom_output, 0.004)  # bf16 mantissa 7bit, relaxed threshold

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


if __name__ == "__main__":
    run_tests()
```

## Quality Checklist

- [ ] Import paths correct: `SupportedDevices` from `torch_npu.testing.common_utils`
- [ ] All test methods have `@SupportedDevices(['Ascend910B'])` decorator
- [ ] All test methods have `device="npu"` parameter
- [ ] All `test_` (positive) methods using random data set `torch.manual_seed(0)`; `test_neg_` exempt (verify error only, no golden compared)
- [ ] Prefer `assertRtolEqual` over `assertEqual` for numerical comparison
- [ ] Tests cover all supported data types
- [ ] Tests cover all optional parameters (both None and with-value cases)
- [ ] Tests cover boundary conditions (min/max dimensions)
- [ ] Tests include error handling (invalid inputs should raise RuntimeError)
- [ ] File ends with `if __name__ == "__main__": run_tests()`
- [ ] Tests can run independently
- [ ] Test method names are clear and have docstrings
- [ ] Report verification results to CHECKLIST.md

## Negative Test Pattern

Negative test methods are appended to the positive test class, unified prefix `test_neg_`. Generate inputs mechanically from the **Violation Construction** code in `### Negative Coverage Matrix`.

### Assertion

```python
@unittest.skipIf(not torch.npu.is_available(), "NPU required for negative validation")
def test_neg_{category}(self, device="npu"):
    x = torch.randn({base_shape}, dtype=torch.{main_dtype}).npu()
    with self.assertRaisesRegex(RuntimeError, r"{error_category}"):
        # violation construction code: {violation_code}
        torch_npu.npu_{op}(x, {violated_call})
```

- Prefer `assertRaisesRegex(RuntimeError, r"PARAM_INVALID")` — error-category granularity
- If the specific message can't be obtained, downgrade to `assertRaises(RuntimeError)` and mark the scenario column `negative-message pending`
- Derive large shape from the operator's `dim_range`: rank = `dim_range[1]`, size = `CLASS_SHAPES` baseline

### Violation Construction Code → Code Mapping

| Violation Code | Step 4 Translation |
|-----------|----------------|
| `dtype=float32` | corresponding param `torch.randn(..., dtype=torch.float32).npu()` |
| `shape=(16,64)` | corresponding param `torch.randn((16,64), ...)` |
| `dst_type=99` | corresponding Attr `op(..., dst_type=99)` |
| `conflict` | pass two mutually-exclusive params together |

## 🔒 dst_type Enum Value — Canonical Source (hard gate)

Scaffold emits scenarios as `dst_type=<dtype_name>` (e.g. `dst_type=float8_e4m3fn`). Step 4 MUST translate the **name** to the int the operator's C++ expects. The C++ passes this int straight to `c10_npu::GetAclDataType(dst_type)` — a wrong int silently maps to an unrelated ScalarType and the kernel rejects it with `<X> has not been supported` (the op never sees your intended dtype; the bug looks like "operator doesn't support this dtype" but is actually a wrong magic number).

**Do NOT hand-derive ints from `c10/core/ScalarType.h` aten ordinals.** NPU quant dtypes do NOT follow aten ordinals — fp8/hifloat8 live in a torch_npu-private enum block (290+). aten `Float8_e5m2=23` is WRONG for `dst_type`; the operator wants `291`.

**Single source of truth** lives IN THE REPO: `op_plugin/python/meta/_meta_registrations.py` → `TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP` (int→dtype, ~line 34) and `TORCH_NPU_DTYPE_TO_STRING_MAP`. The vendor `torch_npu.<dtype>` constants mirror the fp8/hifloat8 half of it. Canonical values:

| dtype name | dst_type int | How to write it in the test (drift-safe form) |
|---|---|---|
| `int8` | 1 | `dst_type=1  # torch.int8` — stable aten ordinal |
| `quint4x2` (INT4) | **16** | `dst_type=16  # torch.quint4x2` — NOT 13 (13=QUInt8) |
| `float8_e5m2` | **291** | `dst_type=torch_npu.float8_e5m2` — import, do NOT write 291/23/30 |
| `float8_e4m3fn` | **292** | `dst_type=torch_npu.float8_e4m3fn` — import, do NOT write 292/24/31 |
| `hifloat8` | **290** | `dst_type=torch_npu.hifloat8` — import, do NOT write 290 |

Rules:
1. **fp8 / hifloat8 → MUST use the `torch_npu.<name>` constant, never a bare int.** These live in the torch_npu-private enum block (290+) and are vendor-owned; a literal 291 drifts if the vendor renumbers. Importing the constant is the only drift-safe form. (The int in the table is for recognition only — do not paste it.)
2. **int8 / quint4x2 have NO `torch_npu` constant** (they are classic aten ScalarTypes). Hardcode `1` / `16` with a `# torch.<name>` comment — these ordinals are frozen and never renumber. `int(torch.int8)` raises TypeError, so live-cast is not an option for these two; 1 and 16 are the only literals allowed.
3. **Do NOT hand-derive from `c10/core/ScalarType.h` aten ordinals.** aten `Float8_e5m2=23` is WRONG for `dst_type`; the operator wants `291`. aten and torch_npu are two different enums that happen to collide in dtype but not in int.
4. **To confirm a value, read the repo map — do NOT invert it in your head.** The map is **many-to-one** (285/290/296/297 all → `uint8`; both 23 and 291 → `float8_e5m2`), so a naive reverse-lookup `{v:k}` silently drops keys and may pick the aten int over the torch_npu one. Resolve forward instead: `python3 -c "import torch_npu; print(torch_npu.float8_e5m2, torch_npu.float8_e4m3fn, torch_npu.hifloat8)"` (→ 291 292 290), or read `TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP` in `_meta_registrations.py` directly.
5. **Symptom → cause map** (so a failing run is diagnosed, not re-guessed): `QUInt8 has not been supported` ⇒ you passed 13 for INT4, use 16. `UInt1/UInt2 has not been supported` ⇒ you passed an aten fp8 ordinal (23/24 or 30/31), use `torch_npu.float8_e5m2/e4m3fn`. These are wrong-magic-number bugs, NOT operator/constraint gaps — do not "fix" the constraint file.
