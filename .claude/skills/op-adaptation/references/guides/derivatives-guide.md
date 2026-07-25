# Derivatives Configuration Guide

`derivatives.yaml` binds forward operators to their backward gradient formulas. Each entry maps a forward operator's input parameters to backward operator calls.

## File Structure

```yaml
all_version: [v2.1, v2.2, ...]
backward:
  - name: <forward_operator_schema>   # Must match op_plugin_functions.yaml func line
    <param1>: <backward_expression>   # How to compute gradient for param1
    <param2>: <backward_expression>   # How to compute gradient for param2
    ...
    version: [v2.1, newest]           # Must match forward operator's op_api range
```

### `output_differentiability` (Multi-Output Operators)

For forward operators returning multiple Tensors, use `output_differentiability` to declare which outputs participate in autograd:

```yaml
- name: npu_multi_output_op(Tensor x1, Tensor x2) -> (Tensor, Tensor, Tensor)
  output_differentiability: [true, false, true]   # Outputs 0 and 2 are differentiable
  x1, x2: npu_multi_output_backward(grads[0], x1, x2, grads[2])
  version: all_version
```

- `[true, false, true]` → outputs 0 and 2 are differentiable, output 1 is not
- Non-differentiable outputs are excluded from the `grads[]` array in the backward expression
- If omitted, all outputs default to differentiable

## Relationship with op_plugin_functions.yaml

| File | Role |
|------|------|
| `op_plugin_functions.yaml` | Registers forward AND backward operator **interfaces** (both need `func:` entries) |
| `derivatives.yaml` | Binds forward ↔ backward — tells autograd WHICH backward op to call for each forward input gradient |

Both are required for autograd to work:
- Backward op registered in yaml but no derivatives binding → autograd won't call it
- Derivatives references an unregistered backward op → runtime error

## Placeholder Reference

| Placeholder | Meaning | Example |
|-------------|---------|---------|
| `grad` | Incoming gradient from downstream layers | Always available |
| `result0`, `result1`, ... | N-th return value of the forward operator | `result1` = second output Tensor |
| `grads[0]`, `grads[1]`, ... | N-th gradient component from multi-output returns | Used with `-> (Tensor, Tensor)` |
| `<param_name>` | Forward input parameter, referenced by name | `self`, `alpha`, `dim` |
| `<param>.sym_sizes()` | Symbolic sizes of a Tensor input | Used when backward needs shape info |
| `<param>.sym_size(N).expect_int()` | N-th dimension value as int | Used when backward needs a specific dim |
| `<param>.scalar_type()` | dtype of a Tensor input | Used when backward output dtype matches input |
| `non_differentiable` | Marks a parameter that doesn't need gradient | `index: non_differentiable` |
| `auto_linear` | PyTorch handles the gradient automatically | `result: auto_linear` |

## Entry Templates

### Single output, single differentiable input

```yaml
- name: fast_gelu(Tensor self) -> Tensor
  self: npu_fast_gelu_backward(grad, self)
  version: [v2.1, newest]
```

### Multi-output, using forward intermediate result

```yaml
- name: _npu_dropout(Tensor self, float p) -> (Tensor, Tensor)
  self: npu_dropout_backward(grad, result1, p)
  version: [v2.1, newest]
```
`result1` is the mask Tensor from forward's second output — used because it carries the dropout pattern.

### Multi-input differentiable

```yaml
- name: _npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, int mode=0) -> (Tensor, Tensor)
  self, gtboxes: npu_ciou_backward(grad, self, gtboxes, result1, trans, mode)
  version: [v2.1, newest]
```

### Extracting scalar metadata from Tensor inputs

```yaml
- name: npu_moe_token_permute(Tensor tokens, Tensor indices, ...) -> (Tensor, Tensor)
  tokens: "npu_moe_token_permute_grad_v2(grad, result1, tokens.sym_size(0).expect_int(), tokens.scalar_type(), (indices.dim() == 1) ? 1 : indices.sym_size(1).expect_int(), padded_mode)"
  version: [v2.1, newest]
```
This pattern extracts `tokens.shape[0]`, `tokens.dtype`, and `indices.shape[1]` as scalar arguments — avoiding the backward kernel from holding full Tensor references, saving memory.

### Multi-output gradient components

```yaml
- name: some_op(...) -> (Tensor, Tensor, Tensor)
  input1, input2: backward_op(grads[0], grads[1], input1, input2)
  version: [v2.1, newest]
```
`grads[0]` and `grads[1]` reference the different gradient components from the forward's multi-output return.

### Non-differentiable parameter

```yaml
- name: gather(Tensor self, int dim, Tensor index, bool sparse_grad=False) -> Tensor
  self: npu_gather_backward(grad, self.sym_sizes(), dim, index, sparse_grad)
  index: non_differentiable
  version: all_version
```

## Generation Checklist

When adding a derivatives entry:

1. **Verify backward op is registered**: The backward operator (`npu_xxx_grad`) must have a `func:` entry in `op_plugin_functions.yaml` before adding the derivatives binding
2. **Match forward signature**: The `name:` field must exactly match the `func:` line from `op_plugin_functions.yaml` (same parameter names, types, order)
3. **Use correct placeholders**: `grad` for incoming gradient, `result0/1` for forward outputs saved for backward
4. **Match version range**: Same `version:` as the forward operator's `op_api:` field
5. **Sort alphabetically**: Insert the new entry at the correct alphabetical position within the `backward:` list
6. **Test with FakeTensor**: Verify the binding works in C-test (op-test) — FakeTensor mode exercises the derivatives dispatch path

---

## Rebind

When adding a backward operator variant (e.g., `_grad_v2`), the forward operator's derivatives.yaml formula must be updated to call the new variant.

### When to Rebind

| Scenario | Action |
|----------|--------|
| Adding backward variant whose forward already has a derivatives binding | **Rebind** — update formula to call new variant |
| Adding forward variant | **New binding** — see Entry Templates above |
| Adding standalone variant (not in autograd chain) | **Skip** |
| Operator has no derivatives binding | **Skip** |

### Pattern Selection

Compare old backward and new variant signatures. Choose the matching pattern:

| Old vs New | Pattern |
|------------|---------|
| Same parameter list, different function name | **Pattern A: Simple Replacement** |
| Old uses Tensor for metadata, new uses scalar instead | **Pattern B: Tensor→Scalar Replacement** |

### Pattern A: Simple Replacement

Same parameters, just a different function name.

1. **Read current binding** — grep forward operator name in `derivatives.yaml`
2. **Replace function name** — e.g., `npu_xxx_grad(...)` → `npu_xxx_grad_v2(...)`
3. **Present diff and confirm**
4. **Verify after build** — `forward → backward → assert grad is not None and shape matches`

### Pattern B: Tensor→Scalar Replacement

Old backward takes Tensor params that are only used for metadata (shape, dtype, dim). New variant takes scalar values. The extraction expressions go into the derivatives.yaml formula itself.

**1. Read Current Binding**

Grep the forward operator name in `derivatives.yaml` to find the existing formula.

**2. Identify Tensor→Scalar Map**

Compare old and new signatures. Map each removed Tensor to its scalar replacement:

```
Old: npu_moe_token_permute_grad(Tensor tokens, Tensor grad, Tensor indices, Tensor result1, bool padded_mode) -> Tensor
New: npu_moe_token_permute_grad_v2(Tensor grad, Tensor result1, int size_0, ScalarType dtype, int topK, bool padded_mode) -> Tensor

Replaced:
  tokens (Tensor) → size_0 (int) + dtype (ScalarType)
  indices (Tensor) → topK (int)

Kept:
  grad → grad_permuted_tokens
  result1 → sorted_indices
  padded_mode → padded_mode
```

**3. Generate Extraction Expressions**

Map each removed Tensor to the expression that extracts its scalar value from the forward operator's available tensors:

| Old param | New param | Extraction expression |
|-----------|-----------|----------------------|
| `tokens` (Tensor) | `size_0` (int) | `tokens.sym_size(0).expect_int()` |
| `tokens` (Tensor) | `dtype` (ScalarType) | `tokens.scalar_type()` |
| `indices` (Tensor) | `topK` (int) | `(indices.dim() == 1) ? 1 : indices.sym_size(1).expect_int()` |

Rules:
- Use `sym_size(N)` (not `size(N)`) for PyTorch dynamic shape compatibility
- Use `.expect_int()` to unwrap `c10::SymInt` → `int64_t` when target is `int`
- Use `.scalar_type()` to get `ScalarType` from a Tensor
- Conditional expressions: `(condition) ? value_if_true : value_if_false`

**4. Build New Formula**

Parameters are positional — map them to the new variant's `func:` signature. Quote the entire formula with double quotes when it contains `?`, `:`, or spaces.

```yaml
# Old formula:
  tokens: npu_moe_token_permute_grad(tokens, grad, indices, result1, padded_mode)

# New formula:
  tokens: "npu_moe_token_permute_grad_v2(grad, result1, tokens.sym_size(0).expect_int(), tokens.scalar_type(), (indices.dim() == 1) ? 1 : indices.sym_size(1).expect_int(), padded_mode)"
```

**5. Present Diff and Confirm**

Side-by-side diff. User explicitly confirms before applying.

**6. Verify After Build**

Run autograd verification:

```python
input_tensor = torch.randn(input_shape).npu().to(dtype).requires_grad_(True)
indices = torch.randint(0, num_classes, index_shape).npu()
output = torch_npu.npu_{forward_op}(input_tensor, indices)
output.backward(torch.ones_like(output))
assert input_tensor.grad is not None
assert input_tensor.grad.shape == input_tensor.shape
```

### Rebind Common Mistakes

- **Forgetting `.expect_int()`**: `sym_size()` returns `c10::SymInt`, not `int64_t`
- **Wrong parameter order**: Positional args must match v2 `func:` signature in `op_plugin_functions.yaml`
- **Not quoting the formula**: Expressions with `?`, `:`, or spaces must be double-quoted in YAML

## Troubleshooting

**Backward propagation not working / autograd returns None**
- Backward operator must be registered in `op_plugin_functions.yaml` AND bound in `derivatives.yaml`
- Verify `name:` in derivatives matches forward `func:` line exactly
- When a backward variant is added but autograd still calls the old version: derivatives rebind was skipped — see Rebind section above
