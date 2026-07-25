# Scenario 2: Backward Hard-Error on A5

> **Scope**: M-adapt submode **Scenario 2** — operator exists on both 910 (ACL forward + backward) and A5 (ACLNN forward only). **A5 has no backward implementation**; calling the backward on A5 must produce a clear hard-error.
> **When to use**: `m-adapt-guide.md` §2 Step 0 classified as State C, AND the operator has a `npu_xxx_backward` entry, AND the A5 aclnn has no backward.
> **Real-world example**: `npu_ciou_backward`.

---

## §1 YAML

```yaml
- func: npu_xxx_backward(Tensor grad, Tensor bboxes, Tensor gtboxes, Tensor? atan_sub, bool trans=False, bool is_cross=True, int mode=0) -> (Tensor, Tensor)
  acl_op: all_version       # legacy platforms take the ACL backward
  op_api: all_version       # A5 also registers op_api (so the C++ TORCH_CHECK error template takes effect)
```

**CRITICAL**: you **MUST keep the `op_api: all_version` line**; do **NOT** write only `acl_op: all_version`. Otherwise, on A5, calling `npu_xxx_backward` won't hit the OpApi path and the hard-error template won't fire.

If both forward and backward exist, register both entries:
```yaml
# Forward — standard Scenario 1
- func: npu_xxx(Tensor self, ...) -> Tensor
  acl_op: all_version
  op_api: all_version
  exposed: all_version

# Backward — Scenario 2 hard-error on A5
- func: npu_xxx_backward(Tensor grad, ...) -> (Tensor, Tensor)
  acl_op: all_version
  op_api: all_version        # ← MUST keep this
```

---

## §2 C++ template (full details: `m-adapt-cpp-patterns-guide.md` §10)

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
- The forward entry (if exists) is normal Scenario 1 code; only the backward gets the hard-error pattern

### §2.1 Alternative implementation: `DO_COMPATIBILITY` (looser fallback)

**When to prefer**: when the A5 aclnn may or may not have the backward across CANN versions (i.e., you'd rather let dlsym probe automatically instead of hard-coding a `>= Ascend950` guard). This is the pattern shipped in `npu_rotary_mul_backward`, `npu_fusion_attention_backward`, `npu_scaled_masked_softmax_backward`.

```cpp
std::tuple<at::Tensor, at::Tensor> npu_xxx_backward(
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    ...)
{
    // dlsym-probe aclnnXxxBackward at runtime; if absent, fall back to legacy acl_op
    DO_COMPATIBILITY(aclnnXxxBackward, acl_op::npu_xxx_backward(grad, bboxes, ...));
    // ... aclnn path ...
}
```

**How `DO_COMPATIBILITY(aclnnXxx, fallback)` works**:
1. dlsym lookup for `aclnnXxx` symbol
2. **Symbol present** → call aclnn path
3. **Symbol absent** (e.g. A5 aclnn lacks backward) → execute `fallback` (= `acl_op::npu_xxx_backward(...)`)

**Trade-offs vs the strict `TORCH_CHECK` template**:

| Aspect | `TORCH_CHECK` (strict, §2 template) | `DO_COMPATIBILITY` (loose, this §2.1) |
|--------|--------------------------------------|----------------------------------------|
| A5 behavior | Explicit hard-error with operator name in message | dlsym probe → if symbol missing, run legacy fallback (which may or may not work on A5) |
| Error message | Clear "backward is not supported on Ascend950" | Generic dlsym/link error (if fallback also fails) |
| Compile-time | Knows the boundary at compile time | Decides at runtime |
| Backward on A5 | Guaranteed to fail | May accidentally succeed (if `acl_op::npu_xxx_backward` runs on A5) |
| Maintenance | Must update guard when backward becomes available | Just remove `DO_COMPATIBILITY` when backward is available |

**Recommendation**:
- Use the **strict `TORCH_CHECK` template** (§2 above) if you **know** A5 will never have backward for this op, and you want a guaranteed fail-fast with a clear message.
- Use the **loose `DO_COMPATIBILITY` template** (§2.1) if A5 may eventually add backward across CANN versions, or if the legacy `acl_op::` path is also acceptable on A5 (e.g., `npu_rotary_mul_backward` where the legacy path happens to work).

For new code, prefer the strict template for explicitness. Existing `DO_COMPATIBILITY`-based operators (like `npu_rotary_mul_backward`) are valid and don't need migration.

---

## §3 Meta registration

**Do NOT register backward in Meta.** Per `m-adapt-guide.md` §8.1 rule 4:
> `npu_xxx_backward` → does **NOT** need registration (consistent with Create mode: backward is not Meta-registered)

This avoids `torch.compile` calling the backward (which would just hard-error) and avoids generating a backward graph for A5.

The forward `npu_xxx` Meta is registered normally (Scenario 1 §3).

---

## §4 Why the YAML keeps `acl_op:` for the backward

The forward `npu_xxx` exists on 910 via `acl_op::npu_xxx` (kept) AND on A5 via `op_api::npu_xxx` (added). The backward `npu_xxx_backward` only exists on 910 via `acl_op::npu_xxx_backward` (kept for legacy) — on A5, the OpApi path is added purely to **trigger the hard-error template**, not to provide an implementation.

---

## §5 See Also

- `m-adapt-guide.md` §6 (Scenario 2 routing), §8 (Meta rules — backward NOT registered)
- `m-adapt-cpp-patterns-guide.md` §10 — full backward hard-error template details
- `scenarios/scenario-1-standard.md` — forward op (same op, scenario 1 forward)
- `references/examples/m-adapt/npu_ciou-a5.md` — `npu_ciou_backward` is the canonical example