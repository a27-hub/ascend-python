# YAML Configuration Guide

## Basic Structure

Add to `op_plugin/config/op_plugin_functions.yaml`:

```yaml
custom:
  - func: npu_operator_name(Tensor input, *, Tensor? optional=None, int attr=1) -> (Tensor, Tensor)
    op_api: [v2.1, newest]
    exposed: [v2.1, newest]
```

## Configuration Fields

| Field | Description |
|-------|-------------|
| `all_version` | All currently supported PyTorch versions, can be referenced directly |
| `official` | PyTorch native operators |
| `custom` | Custom operators (prefixed with npu_) |
| `symint` | Operators supporting SymInt type parameters, must be configured together with custom |
| `func` | Operator schema: `name(parameter_list) -> return_type` |
| `op_api` | Version range supporting aclnn calls |
| `exposed` | Version range for commercial (externally exposed) operators. **Only set for forward operators**; omit for backward operators, internal helpers, or operators not intended for external use. |

## Versioned Operators (e.g., V2)

Some operators have versioned variants (e.g., `npu_fused_infer_attention_score` and `npu_fused_infer_attention_score_v2`). **Treat each versioned variant as an independent operator**:
- Separate YAML entry (different `func` name)
- Separate C++ implementation file (`{Name}V2KernelNpuOpApi.cpp`)
- Separate Meta registration
- Separate test file

## .List Overloads

Some operators have a `.List` overload where a parameter that is `Tensor?` becomes `int[]?`. This is a **PyTorch overload** of the same operator:
- Both variants share the same C++ implementation file (two overloaded functions)
- Both need separate YAML entries: `func: npu_op(...)` and `func: npu_op.List(...)`
- Both need separate Meta registrations
- A single test file can cover both variants

## Version Range Selection

```yaml
op_api: all_version        # Use for operators available in all supported versions
op_api: [v2.1, newest]     # Use for operators added since v2.1 (most common for new ops)
op_api: v2.7               # Use for operators only in a specific version
```

**Rule**: Use `[v2.1, newest]` for new custom operators unless the operator documentation specifies a different version range.

## Type Rules

```yaml
# Required params before *, optional after
func: npu_op(Tensor input, int dim, *, Tensor? mask=None, int? mode=None, str layout="BSH") -> Tensor

# SymInt needs both "custom:" and "symint:" sections
custom:
  - func: npu_op(Tensor input, SymInt[] sizes) -> Tensor
    op_api: [v2.1, newest]
symint:
  - func: npu_op(Tensor input, SymInt[] sizes) -> Tensor
    op_api: [v2.1, newest]
```
