# Meta Registration Guide

Meta Registration enables operators to work with `torch.compile`, `torch.export`, and FakeTensor mode by providing shape/dtype inference without actual execution.

> **Before writing meta**: obey any CHECKLIST.md "Pitfall Alerts" whose steps include C-meta. Meta must use the **same** shape rule C++ used — the same bug lands independently in both.

## Location

Append to `op_plugin/python/meta/_meta_registrations.py`

## ACLGraph Auxiliary Interfaces Exception

`.out` / `_get_max_workspace` (attention ACLGraph manual-update interfaces; 仓内既有的 `_infer_output` 同) are **not** registered in meta — they are invoked directly by the torch_npu ACLGraph handler and never go through Dynamo dispatch; `.default`'s existing meta covers graph mode. Do not add meta for them, and do not treat their absence as a gap during modification or verification. Full rationale: `references/guides/aclgraph-attention-guide.md` §Meta 与 Derivatives.

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

## M-adapt: A5 Platform Meta Registration (stub)

> **M-adapt meta content has been split.** Mandatory rules + checklist remain in `m-adapt-guide.md` §8. Per-scenario Meta patterns (output shape alignment, multi-output tuple, A5-only Meta) are in scenario files.

**Quick reference for M-adapt** (Meta rules in `m-adapt-guide.md` §8; scenario-specific Meta in scenario files):

| Concern | See |
|---------|------|
| Why meta is mandatory in M-adapt (torch.compile) | `m-adapt-guide.md` §8.1 |
| Meta function body principles (shape/dtype only, no compute) | `m-adapt-guide.md` §8.2 |
| Scenario 1: aclnn output shape ≠ user-facing API | `m-adapt-guide.md` §8.3 + `scenarios/scenario-1-standard.md` |
| Scenario 2: multi-output internal API (`_npu_xxx`) | `m-adapt-guide.md` §8.3 |
| Scenario 3: backward is NOT registered | `m-adapt-guide.md` §8.1 rule 4 |
| Scenario 4 (A5-only): Meta per form | `scenarios/scenario-4-a5-only.md` §Meta (each form) |
| M-adapt Meta registration checklist | `m-adapt-guide.md` §8.4 |

## Modifying Existing Meta Registration

> This section is the **modify-mode counterpart** of the create-mode workflow. When C++ `_out_size` / dtype logic changes or a new variant is added, use this section to update meta efficiently.

### Modification Decision Table

Use this table to determine exactly what to change based on what changed in C++:

| C++ Change | Meta Action | Reference | Risk |
|-----------|------------|-----------|------|
| `_out_size` shape formula changed | Update shape derivation in meta function body | - | **High** - silent wrong output |
| `apply_tensor_without_format` dtype changed | Update `dtype=` arg in meta output tensor(s) | - | **High** - dtype mismatch |
| New `TORCH_CHECK` added | Add corresponding `torch._check` in meta | - | **Medium** - missing guard |
| New parameter added to `_out_size` | Add parameter to meta function signature | - | **Medium** - signature mismatch |
| Parameter type changed (e.g., `int` to `int64_t`) | Update meta parameter type annotation (if any) | - | **Low** - Python duck-typing |
| Sentinel value behavior changed | Update sentinel guard in meta | Sentinel Patterns (below) | **High** - shape divergence |
| New variant (`_v2`, `_v3`) added | Create NEW `@impl` for new variant name; do NOT modify old variant's meta | - | **Medium** - variant confusion |
| No shape/dtype logic changed | **Skip** - meta unchanged | - | - |

### Sentinel Value Patterns

Sentinel values are the #1 cause of meta/C++ divergence in modify mode. When C++ uses a sentinel, meta must replicate the same guard:

| Sentinel | C++ Pattern | Meta Pattern |
|----------|------------|-------------|
| `-1` (default dim) | `int64_t dim = (output_dim == -1) ? shape[-1]/2 : output_dim;` | `if output_dim == -1: shape[-1] = shape[-1] // 2` |
| `0` (world_size) | `if (world_size <= 0) world_size = 1;` | `if world_size <= 0: world_size = 1` |
| `nullptr` (optional tensor) | `if (bias && bias->defined()) { ... }` | `if bias is not None: ...` |
| `0` (empty placeholder) | Return empty tensor for unused output — **only when the kernel tolerates a rank drop** | `torch.empty([0], ...)` **only if** C++ also allocates `{0}`; if the kernel's tiling requires the output to keep its real rank (e.g. `CheckOutShape`), allocate the real shape here too and let the flag gate writes, not shape (see Pitfall Gate) |
| `290` (hifloat8 dtype) | Special dtype enum handling | `if dst_type == 290: dtype = torch.uint8` |

> **Rule**: For every sentinel in C++ `_out_size`, there must be an identical guard in meta.

### Modification Workflow (5 Steps)

**Step M1: Locate**
```bash
grep -n "def npu_{op_name}_meta" op_plugin/python/meta/_meta_registrations.py
```
Note the exact line number and function name. If not found, operator has no meta - skip to "Adding Meta to an Existing Operator" below.

**Step M2: Diff the C++ change**
Read the C++ diff (`git diff` or the change description). Identify which of the 4 dimensions changed:
- **Shape**: `_out_size` formula, `SmallVector` construction, `apply_tensor_without_format` shape arg
- **Dtype**: `apply_tensor_without_format` dtype arg, `c10::ScalarType`, `TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP` lookup
- **Signature**: parameter count, types, names
- **Constraints**: `TORCH_CHECK` additions, modifications, or removals

**Step M3: Update meta**
Apply the corresponding change from the Decision Table above:
- Shape: update shape derivation code (use method selection: `empty_like` / `new_empty` / `torch.empty`)
- Dtype: update dtype arg (rstd/scale always float32, INT4 use int32, etc.)
- Signature: update function parameters (required->positional, optional->=None, attr->keyword-only)
- Constraints: add/remove `torch._check` (use `torch._check` for shape/dtype, `raise RuntimeError` for config/enum)

**Do NOT** duplicate the `@impl` block. Edit the existing function in-place.

**Step M4: Modify Mini Self-Check** (5 items, focused on what changed)

```
[ ] 1. Meta signature still matches YAML func: line?
      (required->positional, optional->=None, attr->keyword-only)
[ ] 2. Changed shape dimension matches C++ _out_size computation?
      (compare dimension-by-dimension)
[ ] 3. Changed dtype matches C++ apply_tensor_without_format / TensorWrapper?
[ ] 4. New C++ TORCH_CHECK -> corresponding torch._check added?
[ ] 5. Sentinel values in changed code handled identically in both?
```

**Step M5: Verify and test**
1. Run C-verify cross-verification (shape/dtype/constraint/output order) against updated C++
2. Run FakeTensor test: `pytest test/core_tests/test_fake_tensor.py -k "{op_name}" -v`
3. Run graph mode test: wrap the eager test with `torch.compile(backend="npugraph_ex")` and verify output matches eager

### Adding Meta to an Existing Operator (No Prior Meta)

When an operator exists in the codebase but has no meta registration, and graph mode support is needed:

1. Write a new `@impl(m, "npu_{op_name}")` function in `_meta_registrations.py`
2. Use the **existing** C++ implementation (not new code) as the shape/dtype reference
3. Follow the create-mode template patterns (Basic Template / Common Patterns sections above)
4. Insert the new `@impl` at the end of `_meta_registrations.py`
5. Expect that C-test and C-docs may need updates for graph mode coverage

### Common Modification Pitfalls

| Pitfall | Symptom | Prevention |
|---------|---------|-----------|
| Changed `_out_size` but forgot meta | torch.compile output shape wrong, eager correct | C-verify cross-check |
| Added parameter to C++ but not meta | `TypeError: npu_xxx_meta() missing argument` | M4 check #1 |
| Sentinel `-1` handled differently | Shape dimension wraps around, silent corruption | Sentinel Patterns table |
| INT4 dtype: used `torch.int8` instead of `torch.int32` | Packed dimension mismatch | Dtype rules |
| Multi-output: returned list instead of tuple | `expected tuple, got list` | M4 check #1 |
| Modified old variant's meta for new variant | Old variant graph mode broken | Decision Table "New variant" row |


## Troubleshooting

(Meta registration and FakeTensor issues. Accumulate from cases.)
