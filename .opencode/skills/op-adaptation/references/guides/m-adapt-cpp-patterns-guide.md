# M-adapt: C++ Implementation Patterns

> **Scope**: M-adapt submode **C++ patterns** — boilerplate, three-segment skeleton, routing mechanism, runtime kernel gating, segment details, backward hard-error template, file naming, full examples, and the M-adapt vs existing-patterns table.
> **When to load this file**: after `m-adapt-guide.md` §6 (scenario routing) tells you which scenario you're in, and you need the C++ implementation details to write `*KernelNpuOpApi.cpp`.
> **Companion files**:
> - `m-adapt-guide.md` — M-adapt orchestration + scenario routing (§1-§6)
> - `scenarios/scenario-N-*.md` — per-scenario YAML/C++ templates
> - `cpp-guide.md` — generic Create/Modify C++ patterns (only for non-M-adapt concerns)

---

## §1 Boilerplate at the top of every M-adapt OpApi file (do NOT change)

The include block + `namespace op_api` + `using npu_preparation = ...` at the top of every `*KernelNpuOpApi.cpp` is **boilerplate that must be copied verbatim from an existing opapi file** (e.g. `MishKernelNpuOpApi.cpp`, `IouKernelNpuOpApi.cpp`). Do not "improve" the includes, do not drop the legacy `#include "op_plugin/AclOpsInterface.h"` (it's needed for the ACL fallback in Segment 1), do not change the namespace or the `using` aliases. End-to-end validation: deviating from this block breaks the link step with missing symbols.

Minimum mandatory prelude (Scenario 1/2: standard M-adapt with legacy ACL path):

```cpp
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
```

Add `#include "torch_npu/csrc/aten/CustomFunctions.h"` **only if you use `_npu_dtype_cast`** in §6 (dtype cast). Add other includes only as their symbols are actually used.

> **Scenario 4 (A5-only) exception**: do **NOT** include `"op_plugin/AclOpsInterface.h"` — there is no legacy ACL path to forward to. See `scenarios/scenario-4-a5-only.md` §3.

---

## §2 Three-segment skeleton

The OpApi file for A5 adaptation follows a **three-segment skeleton**:

```cpp
#include "torch_npu/csrc/aten/CustomFunctions.h"   // for _npu_dtype_cast
#include "op_plugin/OpApiInterface.h"              // aclnn operator interface
#include "op_plugin/AclOpsInterface.h"             // legacy acl_op path
#include "op_plugin/utils/op_api_common.h"         // Macros and utilities

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

ReturnType npu_xxx(...) {
    // ========== Segment 1: SOC routing (fixed) ==========
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_xxx(...);  // forward directly to legacy path
    }

    // ========== Segment 2: adaptation (fill as needed) ==========
    // 2.1 optional: dtype cast (when aclnn doesn't support original dtype)
    // 2.2 infer output size
    // 2.3 pre-allocate output tensor
    // 2.4 invoke aclnn operator

    // ========== Segment 3: special handling (fill as needed) ==========
    // 3.1 optional: dtype cast back
    // 3.2 optional: output transpose
    // 3.3 return result
}

}  // namespace op_api
```

> **Per-scenario skeleton differences**:
> - Scenario 1 (standard): Segment 1 is `GetSocVersion() < Ascend950 ? acl_op::npu_xxx : ...`
> - Scenario 2 (backward harderror): Segment 1 is `GetSocVersion() >= Ascend950 ? TORCH_CHECK(false) : acl_op::npu_xxx_backward`
> - Scenario 3 (pure ACLNN): Segment 1 is **no routing** (legacy removed), only A5 path
> - Scenario 4 (A5-only): Segment 1 is `GetSocVersion() >= Ascend950 ? <A5 path> : TORCH_CHECK(false, "only supported on Ascend950 and above")`

---

## §3 Choosing the A5 routing mechanism

| Situation | Routing | Reason |
|-----------|---------|--------|
| Operator's aclnn is **confirmed present** on A5 in the installed CANN (verified via `m-adapt-guide.md` §4 M-coverage) | `GetSocVersion()` | lightweight runtime query, no dlsym overhead |
| aclnn symbol exists but the **A5 kernel is version-dependent** (present in a newer CANN, maybe absent here) | `GetSocVersion()` **+ `check_aclnn_kernel_available(...)`** | must probe the *exact variant* at runtime and degrade gracefully — see §4 below |
| aclnn availability **unverified**, single-path op | `DO_COMPATIBILITY` | dlsym-looks up the operator symbol in libopapi.so |

> ⚠️ **Do not blindly default to plain `GetSocVersion()`.** The old "M-adapt default: GetSocVersion(), ciou/iou confirm the aclnn exists" assumption only holds for simple ops whose A5 kernel you verified in M-coverage. For real transformer/MC2 ops (fusion_attention, moe_distribute_dispatch) the needed aclnn variant (`V4`/`V5`) may be missing from the installed CANN even though the symbol links — a bare `GetSocVersion()` + `EXEC_NPU_CMD` there crashes on A5. Use runtime gating.

`GetSocVersion` template (simple op, A5 kernel verified present):
```cpp
if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
    return acl_op::npu_xxx(...);
}
```

---

## §4 Runtime kernel gating (version-dependent A5 kernels)

When the A5 kernel is version-gated (§4 of `m-adapt-guide.md` found the symbol but a specific variant may be absent, or the op dispatches across `V2/V3/V4/V5`), do **not** call `EXEC_NPU_CMD(aclnnXxx)` unconditionally. Probe the exact variant with `check_aclnn_kernel_available`, fall back down the version chain, and hard-error with an actionable message when a required feature/version is unavailable. This is the pattern shipped in `MoeDistributeDispatchV2KernelOpApi.cpp` and `FlashAttentionV2KernelNpuOpApi.cpp`:

```cpp
// A5 branch (GetSocVersion() >= Ascend950 already established)
if (check_aclnn_kernel_available("aclnnXxxV4")) {
    EXEC_NPU_CMD(aclnnXxxV4, /* ...full V4 args incl. new params... */);
} else if (check_aclnn_kernel_available("aclnnXxxV3")) {
    // a feature that only V4 supports must hard-error, not silently drop:
    TORCH_CHECK(!new_param.has_value(),
        "The param <x> is not supported in this CANN version, aclnnXxxV4 is not available, "
        "please update CANN version.", OPS_ERROR(ErrCode::PARAM));
    EXEC_NPU_CMD(aclnnXxxV3, /* ...V3 args... */);
} else {
    // no A5 kernel at all in this CANN → refuse loudly instead of crashing
    TORCH_CHECK(false,
        "aclnnXxx has no Ascend950 kernel in the installed CANN; "
        "requires a newer CANN (e.g. 9.1.0).", OPS_ERROR(ErrCode::NOT_SUPPORT));
}
```

Rules:
- Probe the **exact variant name** the branch calls (`aclnnXxxV4`), not the base name — a lower version being available does not mean the one you call is.
- Every feature that a newer variant adds must be guarded by a `TORCH_CHECK(!feature.has_value(), "... please update CANN version")` on the older-variant path, so passing that feature on an old CANN fails clearly instead of being silently ignored.
- The final `else` (no A5 kernel) hard-errors — matching `m-adapt-guide.md` §4 M-coverage's "report and stop" outcome at runtime.

> **Scenario 4 (A5-only)** also uses this pattern as its primary mechanism — see `scenarios/scenario-4-a5-only.md` §3 form 2.

---

## §5 Segment 2.1: dtype cast (when aclnn doesn't support the original dtype)

**Typical case**: aclnnCIoU does not support fp16 input; needs fp16→fp32→fp16.

```cpp
bool self_is_half = self.scalar_type() == at::kHalf;
at::Tensor self_cp = self_is_half
    ? at_npu::native::custom_ops::_npu_dtype_cast(self, at::kFloat)
    : self;
```

**Mandatory rules**:
- Use `at_npu::native::custom_ops::_npu_dtype_cast` (takes `ScalarType`)
- Do **NOT** use the legacy `npu_dtype_cast` (takes `int64_t`)
- Requires `#include "torch_npu/csrc/aten/CustomFunctions.h"`

---

## §6 Segment 2.2-2.3: infer output size + pre-allocate

```cpp
c10::SmallVector<int64_t, op_infer::SIZE> output_size = {self.size(1), gtboxes.size(1)};
at::Tensor overlap = npu_preparation::apply_tensor_with_format(
    output_size, self_cp.options(), ACL_FORMAT_ND);
```

**ACL_FORMAT choice**:
- Default `ACL_FORMAT_ND`
- If the aclnn operator requires a specific format, consult the aclnn doc

---

## §7 Segment 2.4: invoke the aclnn operator

```cpp
EXEC_NPU_CMD(aclnnCIoU, self_cp, gtboxes_cp, trans, is_cross, mode_str, overlap, atan_sub);
```

The `EXEC_NPU_CMD` macro handles automatically:
- operator availability check
- workspace allocation
- stream synchronization

---

## §8 Segment 3.1: dtype cast back

```cpp
if (self_is_half) {
    overlap = at_npu::native::custom_ops::_npu_dtype_cast(overlap, at::kHalf);
}
```

---

## §9 Segment 3.2: output transpose (when aclnn output shape differs from user-facing API)

**Typical case**: aclnnIou outputs `(n, m)`; the user-facing API expects `(m, n)`.

```cpp
return overlap.transpose(0, 1).contiguous();
```

---

## §10 Backward unsupported hard-error template

**Scope**: A5 aclnn has no backward (e.g. `npu_ciou_backward`).

```cpp
std::tuple<at::Tensor, at::Tensor> npu_xxx_backward(
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    ...)
{
    // Ascend950 and above do not support backward
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        TORCH_CHECK(false,
            "npu_xxx_backward is not supported on Ascend950 and above",
            OPS_ERROR(ErrCode::NOT_SUPPORT));
    }
    return acl_op::npu_xxx_backward(grad, bboxes, ...);
}
```

**Mandatory rules**:
- Use `TORCH_CHECK(false, ...)` to fail fast (do **NOT** silently fall back)
- Must include `OPS_ERROR(ErrCode::NOT_SUPPORT)` error code
- The message must state `"Ascend950 and above"` and the specific operator name
- See `m-adapt-guide.md` §6 Scenario 2 for the matching YAML configuration

---

## §11 File naming convention

OpApi files follow the `<OpName>KernelNpuOpApi.cpp` pattern (with the **dual-form** detected: `*KernelNpuOpApi.cpp` (majority) **or** `*KernelOpApi.cpp` (28 ops) — see `m-adapt-guide.md` §3 condition 2):

| Operator | Filename |
|----------|----------|
| `npu_ciou` | `CiouKernelNpuOpApi.cpp` |
| `npu_iou` | `IouKernelNpuOpApi.cpp` |
| `npu_xxx` | `XxxKernelNpuOpApi.cpp` (or `XxxKernelOpApi.cpp` — both are valid) |

Location: `op_plugin/ops/opapi/`

---

## §12 Full examples

Two canonical end-to-end C++ examples (transcribed from master commits) live at `references/examples/m-adapt/`:
- `npu_ciou-a5.md` — typical: fp16→fp32 cast, backward hard-error, multi-output internal API
- `npu_iou-a5.md` — typical: output transpose, no dtype cast

For Scenario 4 (A5-only) examples, see:
- `references/examples/m-adapt/a5-only/` (after the case archive is added per `m-adapt-guide.md` §11.1)

Read them as working code references, not as templates to copy verbatim.

---

## §13 M-adapt vs existing patterns (quick contrast)

| Aspect | Plain OpApi (C-mode) | M-adapt mode |
|--------|----------------------|--------------|
| Includes | `OpApiInterface.h` + `op_api_common.h` | `+ AclOpsInterface.h` (legacy path) + `CustomFunctions.h` (dtype cast) — except Scenario 4 drops `AclOpsInterface.h` |
| Platform routing | usually `DO_COMPATIBILITY` (acl_op fallback) | `GetSocVersion()` (ACL vs ACLNN routing) |
| Legacy path | usually none | keeps `acl_op::npu_xxx` legacy path, forwarded directly — except Scenario 3/4 |
| Output post-processing | simple | often needs dtype cast back / transpose |

---

## §14 See Also

- `m-adapt-guide.md` §1-§6 (overview, state classification, conditions, M-coverage, scenario routing)
- `scenarios/scenario-N-*.md` — per-scenario YAML + C++ + Meta templates
- `cpp-guide.md` §(M-adapt stub) — quick reference table for M-adapt C++ concerns
- `references/examples/m-adapt/` — end-to-end worked examples