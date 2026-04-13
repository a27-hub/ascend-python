# Meta Registration Guide

Meta Registration enables operators to work with `torch.compile`, `torch.export`, and FakeTensor mode by providing shape/dtype inference without actual execution.

## Location

Append to `op_plugin/python/meta/_meta_registrations.py`

## Basic Template

```python
@impl(m, "npu_operator_name")
def npu_operator_name_meta(input, *, optional_param=None, int_param=1):
    # 1. Parameter validation (optional but recommended)
    torch._check(
        input.dim() >= 2,
        lambda: f"input must be at least 2D, but got {input.dim()}D" + ops_error(ErrCode.VALUE),
    )

    # 2. Infer output shape
    output_shape = list(input.shape)
    # Modify based on operator logic

    # 3. Infer output dtype
    output_dtype = input.dtype
    # Modify based on operator logic

    # 4. Return Meta Tensor
    return torch.empty(output_shape, dtype=output_dtype, device='meta')
```

## Common Patterns

```python
# Output same shape as input
return torch.empty_like(input, device='meta')

# Different output dtype
return torch.empty(input.shape, dtype=torch.int8, device='meta')

# Multiple outputs
return (
    torch.empty(shape1, dtype=input.dtype, device='meta'),
    torch.empty(shape2, dtype=torch.int64, device='meta'),
)
```

## Quantization Operators

```python
# Dtype mapping
DTYPE_MAP = {
    2: torch.int8,
    13: torch.quint4x2,  # INT4 packed
    30: torch.float8_e4m3fn,
    31: torch.float8_e5m2,
}

# INT4 packed: last_dim / 8
if dst_type == 13:
    output_shape[-1] = output_shape[-1] // 8
    output = torch.empty(output_shape, dtype=torch.int32, device='meta')
```

For complete examples, refer to existing Meta registrations in the file.
