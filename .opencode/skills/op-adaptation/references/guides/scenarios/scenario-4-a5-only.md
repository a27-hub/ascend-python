# Scenario 4: A5-only Operator (no legacy ACL path)

> **Scope**: M-adapt submode **Scenario 4** — adapt a brand-new operator that exists **only on Ascend950**, never supported on 910/legacy platforms. There is **no legacy ACL path** to fall back to.
> **When to use this scenario**: `m-adapt-guide.md` §2 Step 0 classifies the operator as **State E (A5-only candidate)**: `op_api:` line exists (or will be added), no `acl_op:` line, no `aclops/*KernelNpu.cpp`.
> **Real-world prevalence**: 242 A5-only candidate operators in `op-plugin/op_plugin/ops/opapi/` (2026-07-06 baseline). Concentrated in MoE training, DeepSeek-V3/R1 inference, KV cache optimization — the A5 differentiation surface.
> **Companion files**:
> - `m-adapt-guide.md` §2 (State E detection), §6 (scenario routing)
> - `m-adapt-cpp-patterns-guide.md` — common C++ boilerplate, three-segment skeleton, runtime gating details
> - `meta-guide.md` — Meta registration rules (general)

---

## §1 Definition and Detection

### §1.1 What is an A5-only operator

**A5-only operator**: in torch_npu, **only Atlas A5 (Ascend950) provides an aclnn implementation** for this operator. The 910 (A/B), 310P, A2/A3 platforms have **no ACL path and no usable aclnn kernel**.

**User-facing behavior**: calling on 910 → immediate "not supported" or link/runtime failure; calling on A5 → aclnn path executes normally.

### §1.2 How State E differs from State C/D

| Aspect | State C (standard M-adapt) | State D (route to Create) | **State E (A5-only)** |
|--------|---------------------------|--------------------------|------------------------|
| `acl_op:` line in YAML | ✅ yes | maybe | ❌ **no** |
| `aclops/<Op>.cpp` | ✅ exists | ❌ missing | ❌ **missing** |
| `op_api:` line | to be added | to be added | ✅ already or to be added |
| 910 fallback | yes (`acl_op::npu_xxx`) | n/a | **no** — must hard-error |
| M-adapt scenario | Scenario 1 / 2 | route to Create | **Scenario 4 (this file)** |

### §1.3 State E detection command

```bash
yaml=$(grep -A6 "func: <op_name>(" op_plugin/config/op_plugin_functions.yaml)
echo "$yaml" | grep -q "op_api:" \
  && ! echo "$yaml" | grep -q "acl_op:" \
  && ! grep -rln "<op_name>" op_plugin/ops/aclops/ --include="*.cpp" \
  && echo "STATE E: A5-only candidate → Scenario 4"
```

If the operator is brand-new (no `func:` line in YAML yet), this detection returns no match — handle as "new A5-only operator": skip M-discover entirely and apply the YAML + C++ templates in this file directly.

---

## §2 Four Implementation Forms

The repo shows **4 mature implementation forms** for A5-only operators. Pick by business scenario:

| Form | When to use | Real-world examples |
|------|------------|---------------------|
| **1. Hard-error template** | 910 explicitly unsupported; user must see clear error | `npu_prompt_flash_attention`, 18 hard-error ops |
| **2. Multi-version runtime gating** | aclnn has V2/V3/V4; CANN version-dependent fallback | `npu_moe_distribute_dispatch_v2`, MoE series |
| **3. `internal_format_opapi`** | MC2/collective-comm fused ops; compile-time version range | `npu_grouped_matmul`, 19 MC2 fused ops |
| **4. `gen_opapi`** | Pure parameter forwarding; no manual glue | `npu_moe_distribute_dispatch_setup`, 12 pure-forward ops |

**Decision tree**:
```
                         A5-only operator
                              │
                ┌─────────────┼─────────────┐
                │             │             │
           Pure        Needs dtype/shape   MC2 fusion /
           forwarding   conversion/glue    multi-version
                │             │             │
                ▼             ▼             ▼
            Form 4         Form 2/3     Form 3
            (gen_opapi)    pick one     (internal_format_opapi)
                            │
                  ┌─────────┴────────┐
                  │                  │
            Multi-version      Single aclnn
            aclnn fallback     simple forwarding
            chain
                  │                  │
                  ▼                  ▼
              Form 2              Form 1
       (check_aclnn_kernel)   (hard-error template)
```

**Recommendation order**: **Form 4 > Form 3 > Form 2 > Form 1** (cleanest → most explicit error reporting).

---

## §3 Form 1: Hard-error template

### §3.1 YAML

```yaml
- func: npu_xxx_a5_only(Tensor x, ...) -> Tensor
  op_api: all_version       # Note: no acl_op: line
  exposed: all_version
```

### §3.2 C++ (`opapi/XxxKernelNpuOpApi.cpp`)

> Note: **do NOT include `AclOpsInterface.h`** — there is no legacy ACL path.

```cpp
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
// Note: no need to #include "op_plugin/AclOpsInterface.h" (no legacy path)

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_xxx_a5_only(const at::Tensor& x, ...) {
    // ===== Segment 1: Hard error =====
    TORCH_CHECK(
        c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
        "npu_xxx_a5_only is only supported on Ascend950 and above, "
        "current SoC version does not support this operator.",
        OPS_ERROR(ErrCode::NOT_SUPPORT)
    );

    // ===== Segment 2: Standard OpApi adaptation (fill as needed) =====
    // 2.1 dtype conversion
    // 2.2 Derive output size
    // 2.3 Pre-allocate output tensor
    // 2.4 Call aclnn operator
    EXEC_NPU_CMD(aclnnXxx, x, ..., output);

    // ===== Segment 3: Special handling (as needed) =====
    // 3.1 dtype convert back / 3.2 output transpose
    return output;
}

}  // namespace op_api
```

### §3.3 Meta

```python
@impl(m, "npu_xxx_a5_only")
def npu_xxx_a5_only_meta(x, ...):
    output_size = [/* same as the final user-facing shape in C++ */]
    return torch.empty(output_size, dtype=x.dtype, device='meta')
```

### §3.4 Trade-offs

- ✅ Clear, actionable error on legacy SoC
- ✅ Compile-time-known semantics
- ❌ No graceful degradation path

---

## §4 Form 2: Multi-version runtime gating

### §4.1 YAML

```yaml
- func: npu_moe_distribute_dispatch_v2(Tensor x, Tensor expert_ids, ...) -> (Tensor, ...)
  op_api: all_version
  exposed: all_version
```

### §4.2 C++ (real-world example, from `MoeDistributeDispatchV2KernelOpApi.cpp`)

```cpp
at::Tensor npu_moe_distribute_dispatch_v2(...) {
    // Segment 1: Legacy-SoC-friendly hard error (optional — if omitted, relies on underlying implicit failure)
    TORCH_CHECK(
        c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
        "npu_moe_distribute_dispatch_v2 is only supported on Ascend950 and above.",
        OPS_ERROR(ErrCode::NOT_SUPPORT)
    );

    // Segment 2: Multi-version aclnn gating
    if (check_aclnn_kernel_available("aclnnMoeDistributeDispatchV4")) {
        EXEC_NPU_CMD(aclnnMoeDistributeDispatchV4, /* V4 full args */);
    } else if (check_aclnn_kernel_available("aclnnMoeDistributeDispatchV3")) {
        // V4-added params must be explicitly rejected on V3
        TORCH_CHECK(!performance_info.has_value(),
            "The performance_info is not supported in this CANN version, "
            "aclnnMoeDistributeDispatchV4 is not available, please update CANN version.",
            OPS_ERROR(ErrCode::PARAM));
        EXEC_NPU_CMD(aclnnMoeDistributeDispatchV3, /* V3 args */);
    } else {
        TORCH_CHECK(false,
            "aclnnMoeDistributeDispatch has no Ascend950 kernel in the installed CANN; "
            "requires a newer CANN (e.g. 9.1.0).",
            OPS_ERROR(ErrCode::NOT_SUPPORT));
    }
}
```

### §4.3 Meta

Same as Form 1 (§3.3) — Meta describes user-facing shape, not aclnn internal shape.

### §4.4 Trade-offs

- ✅ Robust across CANN versions
- ✅ New params on V4 explicitly fail-fast on V3 (no silent degradation)
- ❌ Legacy SoC error message depends on CANN-level failure if Segment 1 hard-error is omitted

---

## §5 Form 3: `internal_format_opapi`

### §5.1 YAML

```yaml
- func: npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, ...) -> Tensor[]
  op_api: [v2.1, newest]
  exposed: [v2.1, newest]
  internal_format_opapi: [v2.1, newest]
```

### §5.2 C++ / Meta

**No hand-written cpp required** — compile-time codegen produces the boilerplate. Meta is also auto-generated.

### §5.3 Trade-offs

- ✅ Minimum hand-written code (codegen handles ND-array conversion, version gating, etc.)
- ✅ Compile-time version range + runtime dlsym, double safety
- ❌ Inflexible — only suitable for standardized forwarding ops

---

## §6 Form 4: `gen_opapi` (pure forwarding)

### §6.1 YAML

```yaml
- func: npu_moe_distribute_dispatch_setup(Tensor x, Tensor expert_ids, ...) -> (Tensor, Tensor, Tensor)
  op_api: all_version
  gen_opapi:
    out:
      size: '...'
```

### §6.2 C++ / Meta

**Fully codegen** — no hand-written cpp, no manual Meta registration.

### §6.3 Trade-offs

- ✅ Zero hand-written code
- ❌ Strictest constraints — only pure forwarding; any dtype cast / output transpose disqualifies this form

---

## §7 M-scope User Confirmation

Before committing to one of the 4 forms, M-scope MUST ask the user:

> "This operator is A5-only (no 910 implementation). What's the expected behavior on legacy SoC?"
> - (a) **Clear hard-error on legacy** → Form 1
> - (b) **Runtime multi-version gating** → Form 2 (for V2/V3/V4 CANN compatibility)
> - (c) **Standardized MC2/collective-comm fused** → Form 3
> - (d) **Pure parameter forwarding** → Form 4

Default: pick Form 4 → 3 → 2 → 1 based on feasibility (pure forwarding first, fallback to hard-error if nothing else fits).

---

## §8 Real-World Case Archive

| Operator | Form | Notes |
|----------|------|-------|
| `npu_swiglu_quant` | 2 | swiGLU + quant fusion |
| `npu_moe_distribute_dispatch` / `_v2` / `_setup` | 2 / 4 | V2/V3/V4 multi-version; pure forwarding for `_setup` |
| `npu_moe_distribute_combine` / `_v2` | 2 | V2/V3 multi-version |
| `npu_grouped_matmul` | 3 | MC2 grouped matmul |
| `npu_apply_rotary_pos_emb` | 2 | RoPE |
| `npu_kv_rmsnorm_rope_cache` / `_v2` | 2 | KV cache + RoPE + RMSNorm fusion |
| `npu_fused_infer_attention_score` / `_v2` | 1 | Hard-error (legacy SoC unsupported) |
| `npu_scaled_dot_product_attention` | 1 | Hard-error |
| `npu_weight_quant_batch_matmul_v2` | 1 | Hard-error |
| `npu_dual_level_quant_matmul` | 1 | Hard-error |
| `npu_chunk_cat` | 1 | Hard-error |
| `npu_embedding_bag` | 1 | Hard-error |
| `npu_foreach_norm` | 1 | Hard-error |
| `npu_add_rms_norm_quant` | 1 | Hard-error |
| `npu_rotary_mul_backward` | 1 | Hard-error |
| `npu_convert_weight_to_int4_pack` | 1 | Hard-error |
| `npu_dequant_swiglu_quant` | 1 | Hard-error |
| `npu_grouped_matmul_swiglu_quant_v2` | 1 | Hard-error |

---

## §9 Key Differences from Standard M-adapt (Scenarios 1/2)

| Aspect | Scenario 1/2 (standard) | **Scenario 4 (A5-only)** |
|--------|------------------------|--------------------------|
| `acl_op:` YAML line | ✅ required | ❌ **absent** |
| `#include "op_plugin/AclOpsInterface.h"` | ✅ required | ❌ **absent** (no legacy path to forward to) |
| Segment 1 behavior | `GetSocVersion() < Ascend950 ? acl_op::npu_xxx : ...` (pass-through) | `GetSocVersion() >= Ascend950 ? <A5 path> : TORCH_CHECK(false)` (hard error) |
| Legacy SoC error message | "no aclnn implementation" (CANN implicit) | "only supported on Ascend950 and above" (custom) |
| Meta shape | matches user-facing API | matches user-facing API (same rule, but no transpose variation needed since no legacy to align with) |

---

## §10 See Also

- `m-adapt-guide.md` §2 — Step 0 State E detection
- `m-adapt-guide.md` §6 — scenario routing table
- `m-adapt-cpp-patterns-guide.md` — boilerplate, three-segment skeleton, runtime gating details
- `meta-guide.md` — Meta registration general rules
- `cpp-guide.md` §(M-adapt stub) — quick reference