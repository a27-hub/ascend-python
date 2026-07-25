# YAML Configuration Guide

Detailed instructions for the YAML generation step. Reference this guide when executing C-yaml. **Create-mode prerequisite**: C-confirm must be complete (all ⚠️ conflicts resolved) first — YAML is the first code step.

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
| `official` | PyTorch native operators |
| `custom` | Custom operators (prefixed with npu_) |
| `symint` | Operators supporting SymInt type parameters, must be configured together with custom |
| `func` | Operator schema: `name(parameter_list) -> return_type` |
| `op_api` | Version range supporting aclnn calls |
| `exposed` | Version range for commercial (externally exposed) operators. **Only set for forward operators**; omit for backward operators, internal helpers, or operators not intended for external use. |
| `internal_format_opapi` | Whitelist field for Ascend NZ format dispatch. By default, operators with Ascend-native format inputs dispatch to acl_op; adding this field enables dispatch to op_api instead. |

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

## Attention ACLGraph Interfaces（条件触发）

**Trigger**: attention-class operator (FA/IFA/MLA/paged-attention) with dynamic seq-len host parameters that needs ACLGraph manual update support — confirmed in C-confirm Q2（含门禁）. Rules: 本节（权威）；仓内映射见 `references/guides/aclgraph-attention-guide.md`。

Besides `.default`, generate **2** extra entries (per logical operator; grad repeats the same pair as an independent logical operator):

> **执行顺序（强制）**：①先跑 `scripts/op_adapt.py gen-aclgraph --op-name <name> --out-names <names>` 出 `.out`/`_get_max_workspace` 草稿 + C++ 骨架（含 ⚠️ codegen-type 标注）；②仅人工填脚本留空字段（版本字段 op_api/internal_format_opapi/device_check/exposed、out 张量命名）；③下方手工模板仅作**校验参考**，禁止跳过脚本直接手写。

```yaml
# .out — .default 参数段逐字照抄（含既有 `*,`），尾部追加 workspace + alias 输出
- func: npu_xxx.out(Tensor query, *, ..., bool softmax_lse_flag=False, Tensor? workspace=None, Tensor(a!) attention_out, Tensor(b!) softmax_lse) -> (Tensor(a!), Tensor(b!))

# _get_max_workspace — .default 参数段逐字照抄，返回 Tensor
- func: _npu_xxx_get_max_workspace(Tensor query, *, ..., bool softmax_lse_flag=False) -> Tensor
```

Rules：
- `.out` 在 workspace 之前的参数与 `.default` 逐字一致（名称/顺序/类型/默认值）；alias 标注 `(a!)/(b!)/...` 与返回值一致，输出顺序与 Host API 一致。
- 全位置参数 `.default`（无 `*,`）追加形式推荐值：`Tensor? workspace=None, *, Tensor(a!)...`（alias 输出必须 keyword-only），C-confirm 确认。
- **SymInt 参数默认照抄保留**（IFA 形态），并同步 `symint:` 分组（生成函数名带 `_symint` 后缀）。`SymInt[]?`→`int[]?` 降级仓内仅 FA v3 一例且未经验证——不作先例，确需降级必须用户明确确认并记入 CHECKLIST（⚠️ codegen-type）。
- 两个新接口均不设 `exposed:`（内部接口）；`op_api` / `internal_format_opapi` / `device_check` 按目标算子真实支持范围填写，**不得照抄其他 FA 算子**。
- 不新增 `_infer_output`（属 auto dispatch 范围）；仓内已有条目不动。
- 样板 = IFA（`op_plugin_functions.yaml` 搜 `npu_fused_infer_attention_score`）。必须先执行 `scripts/op_adapt.py gen-aclgraph --op-name <name> --out-names <names>` 生成两条目草稿（含 ⚠️ codegen-type 标注），再人工确认版本字段与 out 张量命名。

## Version Range Selection

```yaml
op_api: all_version        # Use for operators available in all supported versions
op_api: [v2.1, newest]     # Use for operators added since v2.1 (most common for new ops)
op_api: v2.7               # Use for operators only in a specific version
```

**Rules**:
- `all_version` is a keyword representing all currently supported PyTorch versions, not a separate YAML field. It can be used as a value for `op_api` and `exposed`.
- Use `[v2.1, newest]` for new custom operators unless the operator documentation specifies a different version range.

## M-adapt: A5 Platform YAML Configuration (stub)

> **M-adapt YAML content has been split by scenario.** This section is a stub. For `op_api: all_version` semantics, see `m-adapt-guide.md` §6. For per-scenario YAML templates (Scenarios 1-4), see:
> - `scenarios/scenario-1-standard.md` — standard M-adapt (acl_op + op_api, both platforms)
> - `scenarios/scenario-2-backward-harderror.md` — backward hard-error on A5
> - `scenarios/scenario-3-pure-aclnn.md` — pure ACLNN replacement (no legacy path)
> - **`scenarios/scenario-4-a5-only.md`** — A5-only operator (4 implementation forms) ⭐ NEW (Gap #11)

**Quick reference for M-adapt** (full YAML templates in scenario files):

| Concern | See |
|---------|------|
| `op_api: all_version` semantics (NOT SoC routing) | `m-adapt-guide.md` §6 |
| Scenario 1 YAML | `scenarios/scenario-1-standard.md` |
| Scenario 2 YAML | `scenarios/scenario-2-backward-harderror.md` |
| Scenario 3 YAML | `scenarios/scenario-3-pure-aclnn.md` |
| Scenario 4 YAML (4 forms) | `scenarios/scenario-4-a5-only.md` §3-§6 |
| YAML change checklist (3 points) | `m-adapt-guide.md` §6 (footer) |

## Type Rules

```yaml
# Required params before *, optional after
func: npu_op(Tensor input, int dim, *, Tensor? mask=None, int? mode=None, str format="default") -> Tensor

# SymInt needs both "custom:" and "symint:" sections
custom:
  - func: npu_op(Tensor input, SymInt[] sizes) -> Tensor
    op_api: [v2.1, newest]
symint:
  - func: npu_op(Tensor input, SymInt[] sizes) -> Tensor
    op_api: [v2.1, newest]
```

## SymInt Operators

When a YAML entry has `SymInt` type parameters, two rules apply:

1. **Dual registration**: The `func:` must be registered in BOTH `custom:` (or `official:`) AND `symint:` sections
2. **C++ naming**: The C++ function name must have a `_symint` suffix. The first SymInt parameter maps to `c10::SymIntArrayRef` or `c10::SymInt`
3. **Version guard**: SymInt is only supported from PyTorch v2.1+, so wrap the C++ implementation with `#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)` / `#endif`

```yaml
# YAML — dual registration
custom:
  - func: npu_op(Tensor input, SymInt[] sizes) -> Tensor
    op_api: [v2.1, newest]
symint:
  - func: npu_op(Tensor input, SymInt[] sizes) -> Tensor
    op_api: [v2.1, newest]
```

```cpp
// C++ — _symint suffix + version guard
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor npu_op_symint(const at::Tensor& input, c10::SymIntArrayRef sizes) {
    // ...
}
#endif
```

## Output Size/Dtype Functions

The YAML `size:` and `dtype:` fields (inside `gen_opapi` or standalone) can reference custom functions defined in:
- `op_plugin/utils/KernelNpuOutputSize.h` — for output shape inference
- `op_plugin/utils/KernelNpuOutputDtype.h` — for output dtype inference

```yaml
# size: references a custom function declared in KernelNpuOutputSize.h
size: npu_operator_out_size(input, param)

# dtype: references a custom function declared in KernelNpuOutputDtype.h
dtype: infer_dtype(input, param)
```

When adding such references, the corresponding function must be declared in the `.h` and defined in the matching `.cpp`. See `references/guides/cpp-guide.md` "Output Size Functions" section for the code template.

## Special Schema Requirements

Some operator names or patterns require additional YAML fields to satisfy PyTorch's code generation validator:

### `dispatch: CompositeExplicitAutograd`

**Required when**: Function name starts with `new_`, ends with `_like`, or has a `tensor_options` parameter without a Tensor input.

```yaml
- func: empty_with_format(int[] size, *, ScalarType? dtype=None, int acl_format=2) -> Tensor
  dispatch:
    CompositeExplicitAutograd: empty_with_format
```

### `tags: nondeterministic_seeded`

**Required when**: Function name contains `rand`, `dropout`, or the function has a `generator` input parameter.

```yaml
- func: dropout_with_byte_mask(Tensor self, float p, bool train) -> Tensor
  tags: nondeterministic_seeded
```

Failure to include these fields when required will cause a build error from torchgen.

## Parameter Position Rules

`AttrType` from `_def.cpp` determines whether a parameter goes before or after `*` in the YAML `func` signature, **regardless of whether it has a default value**:

| AttrType | Has default | YAML position | Example |
|----------|------------|---------------|---------|
| REQUIRED | Yes | Before `*` | `float scale_value=1.0` before `*` |
| REQUIRED | No | Before `*` | `Tensor input` before `*` |
| OPTIONAL | Yes | After `*` | `int? mode=0` after `*` |
| OPTIONAL | No | After `*` | `Tensor? mask=None` after `*` |

**Rule**: REQUIRED parameters always go before `*`, even if `_def.cpp` provides a default value via `Attr(default_value)`. OPTIONAL parameters always go after `*`. This matches Python's `def func(a, b=1, *, c=None)` semantics.

**Incorrect** (common mistake):
```yaml
# scale_value is REQUIRED in _def.cpp, should NOT be after *
func: npu_op(Tensor input, *, float scale_value=1.0) -> Tensor
```

**Correct**:
```yaml
# scale_value is REQUIRED → before *, regardless of default
func: npu_op(Tensor input, float scale_value, *, Tensor? mask=None) -> Tensor
```

## Best Practices

- Verify YAML `func:` signature matches `_def.cpp` and `.md` before proceeding to C-cpp — catch mismatches early.

## Troubleshooting

(YAML-specific issues. Accumulate from cases.)
