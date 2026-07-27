# Scenario 1: Standard M-adapt (acl_op + op_api, both platforms)

> **Scope**: M-adapt submode **Scenario 1** — the most common case. Operator exists on **both 910 (ACL path) and A5 (ACLNN path)**. Legacy platforms keep ACL, A5 takes ACLNN, runtime routes by SoC.
> **When to use**: `m-adapt-guide.md` §2 Step 0 classified the operator as **State C**, and the 4-condition AND gate (§3) all-checked.
> **Real-world examples**: `npu_ciou`, `npu_iou`, `npu_mish_backward`.

---

## §1 YAML

```yaml
- func: npu_xxx(Tensor self, Tensor gtboxes, int mode=0) -> Tensor
  acl_op: all_version       # legacy platforms (A2/A3/310P/910) take the ACL path
  op_api: all_version       # register the op_api path for all PyTorch versions (runtime routes by SoC)
  exposed: all_version
```

**Key points**:
- `acl_op: all_version` MUST be kept — A2/A3/310P/910 will use the legacy ACL path
- `op_api: all_version` is added — A5 uses the new ACLNN path
- Runtime routing between the two is decided by `GetSocVersion()` in C++, **not by YAML**

---

## §2 C++ skeleton (full details: `m-adapt-cpp-patterns-guide.md`)

```cpp
#include "torch_npu/csrc/aten/CustomFunctions.h"   // for _npu_dtype_cast
#include "op_plugin/OpApiInterface.h"              // aclnn operator interface
#include "op_plugin/AclOpsInterface.h"             // legacy acl_op path
#include "op_plugin/utils/op_api_common.h"         // Macros and utilities

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

ReturnType npu_xxx(...) {
    // ========== Segment 1: SOC routing ==========
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_xxx(...);  // forward directly to legacy path
    }

    // ========== Segment 2: adaptation ==========
    // 2.1 optional: dtype cast (when aclnn doesn't support original dtype)
    // 2.2 infer output size
    // 2.3 pre-allocate output tensor
    // 2.4 invoke aclnn operator

    // ========== Segment 3: special handling ==========
    // 3.1 optional: dtype cast back
    // 3.2 optional: output transpose
    // 3.3 return result
}
}
```

For segment-by-segment details (boilerplate, dtype cast, output size, `EXEC_NPU_CMD`, transpose), see `m-adapt-cpp-patterns-guide.md` §2-§9.

---

## §3 Meta registration

Per `m-adapt-guide.md` §8, register:
- **User-facing API** `npu_xxx` → `@impl(m, "npu_xxx")`
- **Internal API** `_npu_xxx` (if exposed) → `@impl(m, "_npu_xxx")`

### Common scenario 1 Meta patterns

**aclnn output shape ≠ user-facing API** (e.g. aclnnIou outputs `(n,m)`, user expects `(m,n)`):
```python
@impl(m, "npu_iou")
def npu_iou_meta(bboxes, gtboxes, mode=0):
    # user-facing API returns (m, n), not aclnn's (n, m)
    output_size = [gtboxes.size(0), bboxes.size(0)]
    return torch.empty(output_size, dtype=bboxes.dtype, device='meta')
```

**Multi-output internal API** (e.g. `_npu_ciou` returns `(overlap, atan_sub)`):
```python
@impl(m, "_npu_ciou")
def _npu_ciou_meta(self, gtboxes, trans=False, is_cross=True, mode=0, atan_sub_flag=False):
    if is_cross:
        output_size = [gtboxes.size(1), self.size(1)]
    else:
        output_size = [1, self.size(1)]
    return (
        torch.empty(output_size, dtype=self.dtype, device='meta'),
        torch.empty(output_size, dtype=self.dtype, device='meta'),
    )
```

---

## §4 Backward handling (special sub-case)

If the operator also has a backward (`npu_xxx_backward`):

```yaml
- func: npu_xxx_backward(Tensor grad, ..., bool is_cross=True, int mode=0) -> (Tensor, Tensor)
  acl_op: all_version       # legacy platforms take the ACL backward
  op_api: all_version       # A5 also registers op_api (so the C++ TORCH_CHECK error template takes effect)
```

**Key point**: you MUST keep the `op_api: all_version` line; do **NOT** write only `acl_op: all_version`. Otherwise, on A5, calling `npu_xxx_backward` won't hit the OpApi path and the hard-error template won't fire. **This pattern is shared with Scenario 2 (backward hard-error)**.

For the C++ hard-error template, see `m-adapt-cpp-patterns-guide.md` §10.

---

## §5 See Also

- `m-adapt-guide.md` §2 (Step 0), §3 (4-condition), §6 (scenario routing)
- `m-adapt-cpp-patterns-guide.md` — boilerplate, three-segment skeleton, runtime gating
- `m-adapt-guide.md` §8 — Meta registration rules + checklist
- `references/examples/m-adapt/npu_ciou-a5.md`, `npu_iou-a5.md`, `npu_mish_backward-a5.md` — end-to-end worked examples
- `scenarios/scenario-2-backward-harderror.md` — backward hard-error pattern (used by Scenario 1's backward sub-case)