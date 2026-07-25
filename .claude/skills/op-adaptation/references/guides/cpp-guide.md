# C++ Implementation Guide

> **Before writing C++**: obey any CHECKLIST.md "Pitfall Alerts" whose steps include C-cpp — each names a case to read and applied C++/meta must share its rule.

## Critical Requirements

Type mapping rules:
- `int[]?` → `c10::OptionalIntArrayRef` (NOT `c10::optional<at::IntArrayRef>`)
- `str` → `c10::string_view` (NOT `std::string`)
- `ScalarType?` → `c10::optional<at::ScalarType>`
- `Tensor?` → `const c10::optional<at::Tensor>&`

## Standard Headers

```cpp
#include "op_plugin/OpApiInterface.h"        // aclnn operator interface
#include "op_plugin/utils/op_api_common.h"   // Macros and utilities

// Only include when DO_COMPATIBILITY fallback is needed:
// #include "op_plugin/AclOpsInterface.h"
```

## Find Similar Operators

Before implementing, search for similar operators:

**By category:**
- Quantization: `*Quant*.cpp`
- Matrix ops: `*Matmul*.cpp`, `*Mm*.cpp`
- Attention: `*Attention*.cpp`, `*Flash*.cpp`

**By parameter patterns:**
- IntArray: Search for `OptionalIntArrayRef`
- string_view: Search for `c10::string_view`
- Custom dtypes: Search for `TensorWrapper`

**Recommended reading:**
- `TanhKernelNpuOpApi.cpp` - simplest example
- `TopKKernelNpuOpApi.cpp` - multiple outputs
- `DynamicBlockQuantNpuOpApi.cpp` - quantization with TensorWrapper

## Standard Code Structure

```cpp
namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// Helper functions
c10::SmallVector<int64_t, SIZE> calc_output_size(const at::Tensor& x) {
    auto shape = op_infer::array_to_small_vector(x.sizes());
    return shape;
}
}

ReturnType npu_operator_name(
    const at::Tensor& input,
    c10::string_view str_param,
    c10::OptionalIntArrayRef int_array_param,
    const c10::optional<at::Tensor>& optional_tensor) {

    // 1. Validate inputs
    TORCH_CHECK(input.dim() >= 2, "input must be 2D+", OPS_ERROR(ErrCode::PARAM));

    // 2. Handle optional parameters
    at::Tensor optional_val;
    if (optional_tensor.has_value() && optional_tensor.value().defined()) {
        optional_val = optional_tensor.value();
    }

    // 3. Calculate output shapes
    auto output_shape = calc_output_size(input);

    // 4. Create output tensors
    at::Tensor output = npu_preparation::apply_tensor_without_format(
        output_shape, input.options());

    // 5. Execute aclnn operator
    EXEC_NPU_CMD(aclnnOperatorName, input, str_param.data(),
                 int_array_param, optional_val, output);

    return output;
}

}  // namespace op_api
```

## Key Patterns

### EXEC_NPU_CMD args must be named lvalues

`EXEC_NPU_CMD` is a macro that binds every arg by non-const lvalue reference (`T&`), so it **cannot take an rvalue** (enum constant / literal / `c10::nullopt` / temporary from a function call).

```cpp
// ❌ aclDataType::ACL_INT8 is an rvalue → error: cannot bind non-const lvalue reference of type 'aclDataType&' to an rvalue
EXEC_NPU_CMD(aclnnDynamicQuantV2, input, ..., aclDataType::ACL_INT8, y, scale, offset);

// ✅ hoist into a named lvalue first
aclDataType dst_type = aclDataType::ACL_INT8;
EXEC_NPU_CMD(aclnnDynamicQuantV2, input, ..., dst_type, y, scale, offset);
```

C-verify can't catch this (type is correct, only value category is wrong) — it surfaces only at C-build. When mirroring a sibling, copy its lvalue handling (e.g. `attr.dst_type`).

### Custom Output Dtypes (Quantization)

```cpp
aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
at::ScalarType y_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
at::Tensor y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(y_dtype));
TensorWrapper y_wrapper = {y, y_acltype};
EXEC_NPU_CMD(aclnnOperator, input, y_wrapper, output);
```

### Workspace Management

```cpp
uint64_t workspace_size = 0;
aclOpExecutor* executor = nullptr;
aclnnOperatorNameGetWorkspaceSize(..., &workspace_size, &executor);

at::Tensor workspace;
if (workspace_size > 0) {
    workspace = npu_preparation::apply_tensor_without_format(
        {static_cast<int64_t>(workspace_size)}, input.options().dtype(at::kByte));
}
EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnOperatorName, workspace, workspace_size, executor, ...);
```

### Out Variant (`_out`)

```cpp
at::Tensor& npu_operator_name_out(
    const at::Tensor& input,
    int64_t param,
    at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnOpName, acl_op::npu_operator_name_out(input, param, result));
    npu_preparation::check_tensor({input}, result, input);
    EXEC_NPU_CMD(aclnnOpName, input, param, result);
    return result;
}
```

### Attention ACLGraph Variants（条件触发）

> **Trigger**: attention-class operator with dynamic seq-len host parameters needing ACLGraph manual update support (confirmed in C-confirm Q2). 模板与细则权威：本节（.out 双路径 / get-max / host 序列参数处理细则 / Constraints）；仓内映射见 `references/guides/aclgraph-attention-guide.md`。
>
> This section is the ACLGraph combination of `### Workspace Management` and `### Out Variant (_out)` above — read those first.

**3 个函数**（每逻辑算子；grad 为独立逻辑算子重复同一组；canonical in-repo sample = IFA，`FusedInferAttentionScoreKernelNpuOpApi.cpp`，anchor by function name, e.g. `construct_fia_output_tensor`）：

```cpp
// 1. Shared helper — output shape/dtype derivation, reused by functional and get-max.
//    新适配统一：op_api 命名空间下的普通函数（非匿名命名空间），
//    以 op_api::construct_xxx_output_tensor(...) 调用（IFA 风格）。
std::tuple<at::Tensor, at::Tensor> construct_xxx_output_tensor(/* 推导所需参数 */ ...);

// 2. .out — workspace 双路径；返回引用 tuple
std::tuple<at::Tensor &, at::Tensor &> npu_xxx_out_symint(
    /* .default 全参数 */,
    const c10::optional<at::Tensor> &workspace,
    at::Tensor &attention_out, at::Tensor &softmax_lse)
{
    if (workspace.has_value()) {
        void *workspace_addr = const_cast<void *>(workspace.value().storage().data());
        uint64_t workspace_size = static_cast<uint64_t>(
            workspace.value().numel() * workspace.value().element_size());  // 字节数，不能把 numel 当字节
        EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnnXxx, workspace_addr, workspace_size,
            query, ..., attention_out, softmax_lse);
    } else {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnXxx, query, ..., attention_out, softmax_lse);
    }
    return std::tie(attention_out, softmax_lse);
}

// 3. _get_max_workspace — 内部构造临时输出 + EXEC_GET_MAX_WORKSPACE_CMD
at::Tensor _npu_xxx_get_max_workspace_symint(/* .default 全参数 */)
{
    auto out = op_api::construct_xxx_output_tensor(...);
    uint64_t workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnXxx,
        query, ..., std::get<0>(out), std::get<1>(out));
    return npu_preparation::apply_tensor_without_format(
        {static_cast<int64_t>(workspace_size)}, query.options().dtype(at::kByte));  // 新接口统一 kByte（numel==字节数）
        // NOTE: 仓内 IFA 历史实现用 query dtype，自洽但多分配；新适配不沿用
}
```

Constraints：
- 实现顺序：helper → `.out` 普通路径 → get-max → `.out` workspace 路径（先证明无 workspace 的 out 与 functional 一致）。
- 同一环境下 `.out` 两条路径与 functional 必须选同一 aclnn 版本；get-max 与执行 API 对应（跨版本覆盖需 CANN 明确契约并逐分支验证）。
- workspace 路径必须用 `EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD`（比 EXEC 多 `workspace_addr`/`workspace_size` 两参数，紧跟 aclnn 名）；无 workspace 路径必须保留。
- 版本/SoC 分派与 functional 一致（仓内先例：`op_plugin::utils::is_gte_cann_version_810rc1`）。
- **预处理审查**：layout 限制、动态参数非空等限制必须转为显式 `TORCH_CHECK`（不只写注释）。
- **No new meta registration** for `.out` / `_get_max_workspace` — see `aclgraph-attention-guide.md` §Meta 与 Derivatives。
- 不实现 `_infer_output`（不在手工 update 范围）。

TODO 清单（与 `gen-aclgraph` 骨架一致，固定 5 项）：① aclnn API 名核对（脚本从 functional C++ 提取，LLM 确认）；② 版本/SoC 分派复制；③ helper 函数体；④ host 序列参数转换（见下"host 序列参数处理细则"）；⑤ 预处理限制转 `TORCH_CHECK`（见上 Constraints 预处理审查）。

#### host 序列参数处理细则

attention 算子的动态 seq-len host 参数有两类形态，处理方式不同：

| Python 形态 | YAML 类型 | C++ 类型 | 处理方式 |
|---|---|---|---|
| `list[int]` | `SymInt[]?` | `OptionalArrayRef<SymInt>` | 直接交给公共转换和 `CopyTypesV2` |
| CPU Tensor | `Tensor?` | `optional<Tensor>` | 校验后转 `IntArrayRef`（见下） |

**不得为了接入 update 擅自改变已有 public schema。** CPU Tensor → `IntArrayRef` 须满足：

- Tensor 在 CPU；Tensor 为一维；dtype 为整数；
- 连续 `int64` 可零拷贝；其他整数 dtype 转换并复制到作用域覆盖 Host API 调用的 `std::vector<int64_t>`；
- **不要让 `IntArrayRef` 指向已销毁的临时 Tensor 或局部容器**——异步 task queue 下的最终参数生命周期由 `EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD` 内部的 `CopyTypesV2` 保证。

长度参数还须满足目标 Host API 对当前 layout 的语义约束，不能只查"类型是整数"：

- BNSD/BSND 等非 packed layout：Q 有效长度不得超过 query 物理 S，KV 不得超 key/value 物理 S；`actual_seq_lengths` 与 `actual_seq_lengths_kv` 不可互换；
- TND 等 packed layout：用累计长度数组，须单调不减、末值与对应 packed token 数匹配；
- page attention / prefix / sink 等模式可能重定义长度参数约束——**逐项读目标 aclnn API 文档，不得仅凭参数名推断含义**。

### Inplace Variant (`_`)

```cpp
at::Tensor& npu_operator_name_(at::Tensor& self, int64_t param)
{
    DO_COMPATIBILITY(aclnnOpName, acl_op::npu_operator_name_(self, param));
    op_api::npu_operator_name_out(self, param, self);
    return self;
}
```

### Multi-Output

```cpp
std::tuple<at::Tensor, at::Tensor> npu_multi_output_op(const at::Tensor& input)
{
    auto output_size = calc_output_size(input);
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size, input.options());
    at::Tensor out2 = npu_preparation::apply_tensor_without_format(output_size, input.options());

    EXEC_NPU_CMD(aclnnOpName, input, out1, out2);
    return std::make_tuple(out1, out2);
}
```

### Output Size Functions (KernelNpuOutputSize.h)

When the YAML `size:` field references a custom output size function, it must be declared in `op_plugin/utils/KernelNpuOutputSize.h` and defined in `op_plugin/utils/KernelNpuOutputSize.cpp`:

**Header** (`KernelNpuOutputSize.h`):
```cpp
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> npu_operator_out_size(
    const at::Tensor& input, int64_t param);
```

**Implementation** (`KernelNpuOutputSize.cpp`):
```cpp
c10::SmallVector<int64_t, SIZE> npu_operator_out_size(
    const at::Tensor& input, int64_t param)
{
    TORCH_CHECK(input.dim() >= 2,
        "input must be at least 2-dimensional." + OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, SIZE> output_shape;
    // Calculate output shape from input dimensions
    output_shape = {input.size(0), input.size(1)};
    return output_shape;
}
```

**Validation rule**: Every Tensor parameter passed to `_out_size` MUST have a `TORCH_CHECK` dimension validation, even if the tensor is not directly used in the shape computation logic. This ensures bad inputs produce clear error messages rather than obscure downstream ACL errors.

Similarly, when the YAML `dtype:` field references a custom infer dtype function, it is declared in `op_plugin/utils/KernelNpuOutputDtype.h` and defined in `op_plugin/utils/KernelNpuOutputDtype.cpp`.

For complete examples, refer to similar operators in the codebase.
#### Graph Mode Implications

`_out_size` is the **reference implementation** for meta shape derivation in graph mode (`torch.compile` / `npugraph_ex`) — if `_out_size` and meta disagree, the graph is built with wrong shapes, producing **silent incorrect results**. Keep sentinel guards (`-1`/`0`/`nullptr`), shape/dtype formulas, and output count/order identical in C++ and meta. For the modification workflow, sentinel patterns, side-by-side comparison, and cross-verification procedure, see `meta-guide.md` §Modifying Existing Meta Registration and `verify-guide.md` §Meta Shape/Dtype Logic Verification.

## Complex Operators (multi-branch output derivation)

> This section applies only when the grouped/branch mode was triggered in C-parse (see `parse-guide.md` § "When to enable the grouped/branch mode"). It is a *semantic* trigger, not a parameter count — skip this section for linear-shape operators.

### Shape Derivation from Dimension Branch Table

Read CHECKLIST "Dimension Branch Table". Every row must have a corresponding shape calculation path. Fill the "C++" column with line numbers after implementation. Any row left as `MISSING` = incomplete.

This section is **methodology only** — it does not encode any specific operator's formulas. When the operator has domain-specific shape rules (e.g. layout dimension systems, page-attention value layouts, multi-path dtype derivation), do **not** hard-code them here. Instead:

- Derive every branch from the operator's own `.md` / `_def.cpp` and the aclnn signature, recording concrete formulas in the CHECKLIST Dimension Branch Table rows for *that* operator.
- After C-confirm, check `references/examples/` for a case matching the operator family (e.g. attention-class operators). Read matching cases for concrete shape patterns and pitfall warnings; treat their formulas as facts to re-verify against source, not as copy-paste templates.

For groups marked "C++ Pre-proc: Y" in the Feature Parameter Groups table, add dedicated shape branches. For "C++ Pre-proc: N" groups, pass parameters through to aclnn unchanged.

### Generic shape-derivation skeleton

Two equivalent structural forms — pick by branch count. Both keep shape derivation **decoupled from the `EXEC_NPU_CMD` call** (compute the output tensor first, then call aclnn). Formulas below are placeholders; fill each branch from the *current* operator's `.md` / aclnn signature.

**Form A — inline switch** (few branches, ~3–4 values):

```cpp
// Resolve the output tensor before calling aclnn.
c10::SmallVector<int64_t, SIZE> out_shape;
std::string dim_value = std::string({dim_param});
if (dim_value == "{value1}") {
    out_shape = { /* formula from Dimension Branch Table row for {value1} */ };
} else if (dim_value == "{value2}") {
    out_shape = { /* formula from row for {value2} */ };
} else {
    TORCH_CHECK(false, "{dim} only supports {value1}/{value2}, but got ", dim_value,
                OPS_ERROR(ErrCode::VALUE));
}
at::Tensor output = npu_preparation::apply_tensor_without_format(out_shape, {input}.options());
EXEC_NPU_CMD(aclnn{Operator}, {input}, ..., output);
```

**Form B — extracted helper** (many branches, or a formula reused across `_out`/workspace variants):

```cpp
// In an anonymous namespace above the entry function:
namespace {
c10::SmallVector<int64_t, SIZE> calc_{op}_out_shape(
    const at::Tensor& {input}, std::string {dim}, int64_t {scalar}) {
    if ({dim} == "{value1}") {
        return { /* formula */ };
    } else if ({dim} == "{value2}") {
        return { /* formula */ };
    }
    TORCH_CHECK(false, "...", OPS_ERROR(ErrCode::VALUE));
}
}  // namespace

// Entry function:
auto out_shape = calc_{op}_out_shape({input}, {dim}, {scalar});
at::Tensor output = npu_preparation::apply_tensor_without_format(out_shape, {input}.options());
EXEC_NPU_CMD(aclnn{Operator}, {input}, ..., output);
```

> Prefer Form B when the same shape logic is needed in more than one place (e.g. the main entry, the `_out` variant, and a `_get_max_workspace` variant all need to construct the output) — extract once, call from each. This matches the established pattern in the codebase of factoring shape/dtype derivation into named helpers.

## Constraint Coverage Self-Check (C-constraints)

**After C++ implementation, before C-meta.** Create mode only.

Goal: every implementable constraint from C-parse is backed by a `TORCH_CHECK` (or validation logic) in C++.

### Run the checker — extractor + heuristic DRAFT

`scripts/op_adapt.py check-constraints --parse-json <path> --cpp-file <path>` → JSON:
- **Deterministic extraction** (trust): all `TORCH_CHECK`/`TORCH_CHECK_VALUE`/`TORCH_CHECK_INDEX` statements (condition + message + line); the constraint list from parse JSON, **pre-filtered to `is_likely_constraint=true`** — noise rows (param descriptions, error codes, workspace/executor/stream) are skipped and reported as `noise_skipped`.
- **Heuristic coverage DRAFT** (do NOT trust): each constraint flagged `covered`/`uncovered` via substring + bilingual (ZH→EN) keyword matching. `covered`/`uncovered` here are only a starting point.

### LLM verification — the matcher is heuristic

The keyword matcher is loose: a single generic keyword (`dtype`/`dim`/`type`/`must`) substring-matches unrelated checks, and `.md` (Chinese) vs C++ (English) causes keyword gaps. Verify **both directions**:

1. **Confirm each `covered`** is a real semantic match, not a substring coincidence (e.g. `必须`→`must` hitting an unrelated check message). Read the matched `TORCH_CHECK` and confirm it actually enforces the constraint.
2. **Probe each `uncovered`** — it may be a genuine gap *or* an EN/ZH-keyword mismatch. Read the C++ to confirm no enforcing check exists before adding one.
3. For genuine gaps → add the `TORCH_CHECK` immediately, before proceeding to C-meta.

Mark the "C++ (C-cpp)" column and the "Constraint Coverage Self-Check" table in CHECKLIST.md.

## Troubleshooting

(Build and runtime issues specific to C++ implementation. Accumulate from cases.)

## Best Practices

- Find similar operators in `op_plugin/ops/opapi/` before implementing — reuse patterns instead of inventing new ones.

- **Fail-fast before `EXEC_NPU_CMD`**: add `TORCH_CHECK` dtype/shape guards *before* the aclnn call. A bad input that reaches `EXEC_NPU_CMD` typically raises (e.g. aclnn 107000) but leaves a **latched aclnn error state** that corrupts the *next* unrelated call in the same process. Surface invalid input as a clean sync `PARAM`/`TYPE` error instead.
- **INT4 output packing is NOT unique — verify against a sibling.** Two conventions coexist in the repo: (a) 2-per-byte → dtype `at::kByte` (`uint8`), physical last dim halved (see `MoeInitRoutingV2KernelNpuOpApi.cpp`, `INT4_NUMS_IN_INT8=2`); (b) 8-per-int32 → dtype `at::kInt` (`int32`), last dim `/8`. Wrap with `TensorWrapper{y, ACL_INT4}` either way. Never assume which — grep the aclnn header + a sibling quant op for the packing rule and mirror it in both C++ and meta.
