# FakeTensor Test Guide

## Purpose

FakeTensor tests verify that operators correctly implement **meta tensor / shape inference**, ensuring compatibility with `torch.compile`, `torch.export`, and PyTorch graph mode.

In `FakeTensorMode`, operations do not execute actual computation — they only derive output tensor metadata (shape, dtype, device). If an operator lacks proper meta registration, `torch.compile` will fail at graph tracing time.

## FakeTensor Test Standards

FakeTensor tests are **meta-only** — they run without NPU hardware. This is their core value: CI graph tracing validation should not depend on specific devices. Clearly distinguished from UT test standards:

```python
# 1. Do NOT use @SupportedDevices decorator (meta-only, no NPU hardware dependency)
# 2. Do NOT use device="npu" parameter
# 3. Import from torch._subclasses.fake_tensor import FakeTensorMode
# 4. All operations inside with FakeTensorMode():
# 5. Shape/dtype assertions use assertEqual (not assertRtolEqual)
# 6. Derive assertion values from CHECKLIST.md Output specs table. Do not guess.
```

> Note: only a few classes in existing `test_fake_tensor.py` have `@SupportedDevices` (most don't). This is correct — FakeTensor tests should NOT add this decorator.

## Required Import

```python
from torch._subclasses.fake_tensor import FakeTensorMode
```

## Assertion Value Source

Read assertion values from the **Shape Rule** and **Dtype Rule** columns of the CHECKLIST.md **Output specs** table. Do not guess.

Example: output spec `y: shape=same as x1.shape, dtype=same as x1.dtype` → assert `self.assertEqual(y.shape, torch.Size([2, 256]))` with comment `# same as x1.shape`.

## Target File

All FakeTensor tests go into:

```
test/core_tests/test_fake_tensor.py
```

追加规则（**锚点定位**，不通读全文重写，与 modify-mode 同一原则）：

1. 搜索 `test_fake_tensor.py` 中 `class Test{OperatorName}` 和 `if __name__` 行号
2. **create 模式**：在 `if __name__` 行之前插入完整新类
3. **modify 模式**：
   - 已有同名类 → 在类末尾（下一个 `class ` 或 `if __name__` 之前）追加方法
   - 无同名类 → 在 `if __name__` 行之前插入新类
4. 用锚点行号 + Edit 精确替换插入，不读全文重写整个文件（避免截断/找错类/破坏缩进）

## Test Patterns

### Pattern A: Pure FakeTensor (most common)

Construct inputs inside `FakeTensorMode()`, call the operator, assert metadata.

```python
class Test{OperatorName}(TestCase):
    def test_npu_{operator_name}_meta(self):
        with FakeTensorMode():
            x = torch.randn([B, D], dtype=torch.float16).npu()  # .npu() only sets device metadata — no actual hardware transfer
            result = torch_npu.npu_{operator_name}(x, ...)
            self.assertEqual(result.shape, expected_shape)
            self.assertEqual(result.dtype, expected_dtype)
            self.assertEqual(result.device.type, "npu")
```

### Pattern B: FakeTensor + Autograd

If the operator supports autograd (i.e., has a backward implementation), also test backward shape inference.

```python
    def test_npu_{operator_name}_backward_meta(self):
        with FakeTensorMode():
            x = torch.randn([B, D], dtype=torch.float16, requires_grad=True).npu()
            result = torch_npu.npu_{operator_name}(x, ...)
            result.sum().backward()
            self.assertEqual(x.grad.shape, x.shape)
            self.assertEqual(x.grad.dtype, x.dtype)
```

### Pattern C: Multi-dtype Coverage

Test multiple input dtypes if the operator supports them (e.g., float16, bfloat16, int8).

```python
    def test_npu_{operator_name}_multi_dtype(self):
        with FakeTensorMode():
            for dtype in [torch.float16, torch.bfloat16]:
                x = torch.randn([2, 16], dtype=dtype).npu()
                result = torch_npu.npu_{operator_name}(x, ...)
                self.assertEqual(result.shape, expected_shape)
                self.assertEqual(result.dtype, expected_dtype_for_this_type)
```

### Pattern D: Multi-return Values

For operators that return multiple tensors, assert each one individually.

```python
    def test_npu_{operator_name}_meta(self):
        with FakeTensorMode():
            x = torch.randn([B, D], dtype=torch.float16).npu()
            y, scale = torch_npu.npu_{operator_name}(x, ...)
            self.assertEqual(y.shape, expected_y_shape)
            self.assertEqual(y.dtype, expected_y_dtype)
            self.assertEqual(scale.shape, expected_scale_shape)
            self.assertEqual(scale.dtype, torch.float32)
```

### Pattern E: Real vs Fake Comparison

For some operators (especially those with complex output shapes), compare fake output against real output.

```python
    def test_npu_{operator_name}_meta(self):
        # Real execution
        x = torch.randn([2, 16], dtype=torch.float16).npu()
        real_out = torch_npu.npu_{operator_name}(x, ...)

        # Fake execution
        with FakeTensorMode():
            fake_x = torch.randn([2, 16], dtype=torch.float16).npu()
            fake_out = torch_npu.npu_{operator_name}(fake_x, ...)
            self.assertEqual(fake_out.shape, real_out.shape)
            self.assertEqual(fake_out.dtype, real_out.dtype)
```

## Naming Convention

- Class name: `Test{PascalCaseOperatorName}` (e.g., `TestNpuRmsNorm`, `TestGroupedMatmul`)
- Method name: `test_npu_{operator_name}_meta` or `test_npu_{operator_name}_backward_meta`

## Checklist

After writing the FakeTensor test, verify:

- [ ] 不使用 `@SupportedDevices` 装饰器
- [ ] 导入 `from torch._subclasses.fake_tensor import FakeTensorMode`
- [ ] Test class appended via anchor location before `if __name__ == "__main__"` line
- [ ] All operations are inside `with FakeTensorMode():`
- [ ] Derive assertion values from CHECKLIST.md Output specs table, not guessing
- [ ] Inputs use `.npu()` device
- [ ] Shape assertions use `self.assertEqual(result.shape, ...)`
- [ ] Dtype assertions use `self.assertEqual(result.dtype, ...)`
- [ ] Device assertions use `self.assertEqual(result.device.type, "npu")`
- [ ] If operator supports autograd, backward test is included
- [ ] Multiple dtypes are covered if applicable
- [ ] Multiple return values are each asserted
- [ ] Report verification results to CHECKLIST.md
