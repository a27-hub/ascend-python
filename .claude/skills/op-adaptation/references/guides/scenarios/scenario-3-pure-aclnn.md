# Scenario 3: Pure ACLNN Replacement (no legacy path)

> **Scope**: M-adapt submode **Scenario 3** — the legacy ACL path has been deprecated/removed; everything goes through ACLNN. YAML has only `op_api:`, no `acl_op:`. C++ drops the Segment 1 ACL forwarding.
> **When to use**: future evolution path. As of 2026-07, no live examples in op-plugin yet; reserved for after the ACL → ACLNN migration is complete.
> **Note**: structurally similar to Scenario 4 (A5-only) but with a different intent — Scenario 3 means "we're deprecating the old path", Scenario 4 means "the op never had an old path".

---

## §1 YAML

```yaml
- func: npu_xxx(...) -> Tensor
  op_api: all_version       # only op_api; runtime GetSocVersion decides ACL vs ACLNN
  exposed: all_version
```

**Differences vs Scenario 1**:
- **No** `acl_op: all_version` line (legacy path removed)
- Only `op_api: all_version`

---

## §2 C++ skeleton

The C++ is essentially the same as Scenario 1, but Segment 1's ACL forwarding branch is removed:

```cpp
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
// 注意: Scenario 3 不需要 #include "op_plugin/AclOpsInterface.h"
// (no legacy path to forward to)

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

ReturnType npu_xxx(...) {
    // ========== Segment 1: no ACL forwarding ==========
    // (legacy path is deprecated; we always go through ACLNN)

    // ========== Segment 2: adaptation ==========
    // ...same as Scenario 1...

    // ========== Segment 3: special handling ==========
    // ...same as Scenario 1...
}
}
```

If you want to **hard-error on legacy SoC** rather than fail implicitly, add a guard at the top:

```cpp
TORCH_CHECK(
    c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
    "npu_xxx is only supported on Ascend950 and above.",
    OPS_ERROR(ErrCode::NOT_SUPPORT)
);
```

This pattern is functionally identical to **Scenario 4 Form 1** (hard-error template). The difference is purely intent: Scenario 3 says "legacy is deprecated, will be removed"; Scenario 4 Form 1 says "legacy never existed".

---

## §3 Meta registration

Same as Scenario 1 — register the user-facing API normally in `_meta_registrations.py`. See `m-adapt-guide.md` §8.

---

## §4 Migration path from Scenario 1 → Scenario 3

This is the natural evolution:

1. **Phase 1 (Scenario 1)**: Both `acl_op:` and `op_api:` registered. 910 → ACL, A5 → ACLNN.
2. **Phase 2 (Scenario 2 for backward)**: Backward hard-errors on A5 as A5 aclnn doesn't have it.
3. **Phase 3 (Scenario 3)**: Legacy ACL deprecated. YAML drops `acl_op:`. C++ removes Segment 1 forwarding.
4. **Phase 4 (optional, Scenario 4-like)**: If the op is logically A5-only (e.g. new fusion that 910 never had), migrate to Scenario 4 patterns.

---

## §5 See Also

- `m-adapt-guide.md` §6 (Scenario 3 routing)
- `scenarios/scenario-1-standard.md` — the "before" state
- `scenarios/scenario-4-a5-only.md` — structurally similar but different intent (A5-only from birth)