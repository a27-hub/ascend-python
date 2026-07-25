# M-adapt: A5 (Ascend950) Platform Adaptation Guide

> **Scope**: Modify submode **M-adapt** — adapt an existing operator for the Atlas A5 (Ascend950) platform by adding an ACLNN path while keeping the legacy ACL path.
> **When to use this guide**: any time the task involves "adapt an existing operator for A5", or M-discover auto-detects M-adapt conditions.
> **Companion guides** (load only as needed for cross-cutting concerns):
> - `modify-guide.md` — Modify-mode orchestration (M-discover / M-scope / M-gate / M-execute) and the generic 4-condition gate
> - `build-guide.md` — Build/compile workflow + cgroup/MAX_JOBS warning
> - `m-adapt-cpp-patterns-guide.md` — C++ boilerplate, three-segment skeleton, runtime gating, file naming, full examples
> - `references/examples/m-adapt/` — Real operator case archive (ciou, iou, mish_backward, moe_dispatch_v2, swiglu)
> - `references/guides/scenarios/` — Per-scenario YAML + C++ + Meta templates (Scenario 1 standard / 2 backward / 3 pure / **4 A5-only**)
>
> **This file is the orchestration entry point.** C++ implementation details are in `m-adapt-cpp-patterns-guide.md`; per-scenario YAML/Meta specifics are in `scenarios/scenario-N-*.md`.

---

## §1 Overview & Scope

**What M-adapt does**: adds an ACLNN (newer two-phase C API) call path to an existing operator, so the operator works on Atlas A5 training series (Ascend950) hardware — which **only supports aclnn, not the older single-step ACL path**.

**Constraints**:
- The operator already exists in op-plugin (no .md to parse; no new signature).
- The PyTorch user-facing signature is **unchanged** — `npu_xxx(...)` still takes the same args and returns the same shape.
- At runtime, the SoC decides which path runs: legacy platforms (A2/A3/310P/910) keep the ACL path, A5 takes the ACLNN path. This is wired in C++ via `GetSocVersion()`, **not in YAML**.

**M-adapt has 4 scenarios** — see §6 Scenario Routing.

---

## §2 M-discover Step 0 — classify the operator's op_api state

**Run FIRST, before the 4-condition check below.** The 4-condition check is an AND-gate that can accidentally reach the right verdict for the wrong reason. Step 0 short-circuits the obvious "already done" cases and "A5-only" cases.

| State | Signal in YAML block / files | Verdict |
|-------|------------------------------|---------|
| **A. Already adapted (hand-written)** | yaml has `op_api:` **or** `internal_format_opapi:` (see note) **and** a `<OpName>Kernel(Npu)?OpApi.cpp` exists in `opapi/` | **STOP** — already A5-adapted. Do NOT trigger M-adapt. (But note: `op_api: all_version` does NOT guarantee the installed CANN has an A5 kernel — see §4 M-coverage.) |
| **B. Already adapted (structured / gen_opapi)** | YAML block contains a **`gen_opapi:`** sub-block (e.g. `exec: aclnnXxx`), usually with `op_api:` and no hand-written cpp | **STOP** — already op_api-routed via an auto-generated adapter. Modify mode explicitly does NOT handle `gen_opapi` (see `SKILL.md`). Report "structured adapter, out of M-adapt scope" and stop. |
| **C. Genuine M-adapt candidate** | `acl_op:` only, **no** `op_api:`, no `internal_format_opapi:`, no `gen_opapi:`, no OpApi.cpp, and an `aclops/<OpName>KernelNpu.cpp` exists | Proceed to the 4-condition check (§3), then M-scope → Scenario 1/2/3 |
| **D. Nothing to modify** | operator absent from YAML, or `acl_op:` present but no `aclops/*KernelNpu.cpp` (AND no `op_api:` either) | Not a Modify/M-adapt case — route to **Create mode** |
| **E. A5-only candidate** | yaml has `op_api:` (or `internal_format_opapi:` / `gen_opapi:`) **but no `acl_op:`**, and no `aclops/<OpName>KernelNpu.cpp` | **Scenario 4 (A5-only)** — see `scenarios/scenario-4-a5-only.md`. There is no legacy ACL path; 910 must hard-error. **This is the most common "no legacy" case in practice (242 ops in op-plugin/op_plugin/ops/opapi/).** |

### Detection commands (run in order — first match wins, with proper priority)

```bash
yaml=$(grep -A6 "func: <op_name>(" op_plugin/config/op_plugin_functions.yaml)

# State A check (catches both routing labels)
if echo "$yaml" | grep -qE "op_api:|internal_format_opapi:" \
   && grep -rlq "<op_name>" op_plugin/ops/opapi/ --include="*.cpp"; then
  echo "STATE A: hand-written OpApi present — out of scope"
  exit 0
fi

# State B check (gen_opapi structured adapter) — must come BEFORE State E (Defect fix: an op with both op_api: line and gen_opapi: sub-block is already structured-adapted, NOT an A5-only candidate. Without this priority, e.g. npu_moe_distribute_dispatch_setup would be misclassified as State E.)
if echo "$yaml" | grep -q "gen_opapi:"; then
  echo "STATE B: gen_opapi structured adapter — out of scope"
  exit 0
fi

# State C check — only reached if no State A/B
if echo "$yaml" | grep -q "acl_op:" \
   && ! echo "$yaml" | grep -qE "op_api:|internal_format_opapi:" \
   && grep -rln "<op_name>" op_plugin/ops/aclops/ --include="*.cpp" >/dev/null; then
  echo "STATE C: genuine M-adapt candidate — proceed to 4-condition gate (§3)"
  exit 0
fi

# State D check — no yaml, or acl_op without aclops cpp
if ! grep -q "func: <op_name>(" op_plugin/config/op_plugin_functions.yaml; then
  echo "STATE D: operator absent from YAML — route to Create mode"
  exit 0
fi

# State E check (A5-only)
if echo "$yaml" | grep -qE "op_api:|internal_format_opapi:" \
   && ! echo "$yaml" | grep -q "acl_op:" \
   && ! grep -rln "<op_name>" op_plugin/ops/aclops/ --include="*.cpp"; then
  echo "STATE E: A5-only candidate — Scenario 4"
  exit 0
fi

echo "STATE ?: ambiguous — manual review required"
```

### ⚠️ Two routing labels for hand-written A5 adaptation — both qualify as State A

- `op_api: all_version` (or `op_api: [v2.1, newest]` etc.) — the common label, **static** dispatch.
- `internal_format_opapi: ...` — a third label used by some operators (e.g. `npu_mla_prolog`). Same A5 coverage meaning as `op_api:`, but commonly paired with **compile-time version-range codegen** (`[v2.1, newest]`) rather than runtime `check_aclnn_kernel_available`. Don't treat this label as "missing" — it is the same A5-adapted state.
- If you only check for `op_api:`, you will miss `internal_format_opapi:`-routed ops and incorrectly classify them as State C/D, then trigger M-adapt and collide with the existing hand-written cpp at link time.

**Only State C / State E** reach the actual M-adapt flow. State C → 4-condition check (§3) → Scenario 1/2/3. State E → directly to **Scenario 4** (no 4-condition check needed; the conditions are different).

---

## §3 Entry Conditions (4-condition auto-detection, State C only)

M-discover auto-detects the following 4 conditions; **all must be met** to enter Scenarios 1/2/3 (State C). **State E skips this check** — see §2.

| # | Condition | Detection command |
|---|-----------|-------------------|
| 1 | YAML has only an `acl_op:` line for the operator (no `op_api:` line) | `grep -A4 "func: <op_name>(" op_plugin/config/op_plugin_functions.yaml` |
| 2 | No OpApi impl exists under `op_plugin/ops/opapi/` (⚠️ two naming conventions exist: `<OpName>KernelNpuOpApi.cpp` AND `<OpName>KernelOpApi.cpp` — a literal `ls` of one form misses the other) | `grep -rln "<op_name>" op_plugin/ops/opapi/ --include="*.cpp"` (empty output = condition met) |
| 3 | Original ACL impl `<OpName>KernelNpu.cpp` exists (repo uses a **flat** `op_plugin/ops/aclops/` dir, not per-operator subdirs) | `grep -rln "<op_name>" op_plugin/ops/aclops/ --include="*.cpp"` |
| 4 | PyTorch interface signature unchanged | `git diff <user_input_files>` shows no signature change |

**M-discover report template** (output when all 4 conditions are met):

```
## M-adapt suggestion
- Operator npu_xxx meets all 4 M-adapt conditions:
  ✓ YAML has only acl_op: all_version (no op_api: line)
  ✓ no OpApi impl under op_plugin/ops/opapi/ (checked both XxxKernelNpuOpApi.cpp and XxxKernelOpApi.cpp)
  ✓ op_plugin/ops/aclops/ contains XxxKernelNpu.cpp
  ✓ PyTorch interface signature unchanged
- **M-adapt mode recommended** (see m-adapt-guide.md)
- **If you actually intend to change the interface, say explicitly "modify X's interface signature" to bypass this suggestion**
```

**Edge cases (do NOT trigger M-adapt Scenarios 1/2/3)**:
- If an OpApi impl already exists (`XxxKernelNpuOpApi.cpp` **or** `XxxKernelOpApi.cpp` — check both) → already A5-adapted (Step 0 state A)
- If the YAML block contains a `gen_opapi:` sub-block → already op_api-routed via a structured adapter; out of scope (Step 0 state B)
- If YAML has both `acl_op:` and `op_api:` lines → partially adapted already
- If `op_plugin/ops/aclops/` has no original `*KernelNpu.cpp` AND no `op_api:` line → no legacy ACL implementation; use Create mode instead (Step 0 state D)
- If `op_plugin/ops/aclops/` has no original `*KernelNpu.cpp` **but** `op_api:` line exists → **State E → Scenario 4** (A5-only, see `scenarios/scenario-4-a5-only.md`)

### M-scope user confirmation

Once M-discover reports an M-adapt suggestion, the M-scope step **MUST ask the user**:

> Operator X meets the 4 M-adapt conditions. Run it as the A5 platform adaptation submode?

- User confirms → proceed with M-adapt
- User says "No, I want to change a feature of X" → route to plain Modify mode

For State E (A5-only), M-scope asks:

> This operator is A5-only (no 910 implementation). What's the expected behavior on legacy SoC? See `scenarios/scenario-4-a5-only.md` §7 for the 4-form decision tree.

---

## §4 M-coverage — verify A5 kernel availability in the installed CANN (REQUIRED, before C-cpp)

**This step is the most common real-world failure.** `op_api: all_version` and a linkable `aclnnXxx` symbol only mean the op **builds**; they do NOT mean the **installed CANN actually ships an Ascend950 kernel**. The op-plugin adaptation layer and the CANN kernel library are versioned independently — an op can compile cleanly yet crash (or silently degrade) at runtime on A5 because the installed CANN lacks the A5 kernel (typical when a kernel lands in CANN 9.1.0 but the box has 9.0.0).

Two distinct notions of "available":

| | Link-time (symbol) | Runtime (A5 kernel) |
|---|---|---|
| Question | Does `aclnnXxx` exist to link against? | Does the **Ascend950** kernel binary exist in the installed CANN? |
| Check | header present in `.../include/aclnnop/` | present in `opp/.../ascend950/aic-ascend950-ops-info-*.json` |
| Consequence if missing | build fails (`undefined reference`) | **builds fine, crashes/degrades on A5 at runtime** |

**Do all three before writing C++:**

1. **Identify the exact aclnn variant the A5 path will call** — not the base name. Real ops dispatch to a specific version (`aclnnXxxV4`, `aclnnFlashAttentionVarLenScoreV5`). The A5 coverage question is about *that* variant.

2. **Grep the installed CANN's Ascend950 op-info for the real kernel name** (note: the kernel name may differ from the aclnn name — e.g. `aclnnSilu` → `Swish`; the version suffix matters — `moedistributedispatchv2` present while `v3`/`v4` absent):
   ```bash
   D=$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/config/ascend950
   grep -rhoiE '"[a-z0-9]*<kernelname>[a-z0-9]*"' "$D"/*.json | tr 'A-Z' 'a-z' | sort -u
   ```

   ⚠️ **False positives** (moe-style): a coarse base-name grep finds `v2` registered — does NOT mean `V4` your code calls is registered. Always grep the **exact variant**.

   ⚠️ **False negatives** (grouped_matmul-style): the aclnn host-API version does NOT always map to a versioned opp kernel. Some ops route `aclnnXxxV4/V5` to a single **base** opp kernel with no version suffix. If the versioned opp entry is absent, also check the base name before concluding "unsupported".

3. **Confirm empirically when feasible** — run the op (or a standard aten op that routes to the same kernel) on the A5 device.

**Branch on the result**:
- **A5 kernel present** → proceed to §6 scenario routing → write C++.
- **Symbol present but A5 kernel absent** (this CANN doesn't support A5 yet) → do **not** silently emit `op_api: all_version` + bare `EXEC_NPU_CMD`. Either:
  - **(preferred)** implement the runtime-gated pattern (see `m-adapt-cpp-patterns-guide.md` §4 "Runtime kernel gating"): `check_aclnn_kernel_available("aclnnXxxV4")` with a graceful fallback chain and a `TORCH_CHECK(..., "... please update CANN version", OPS_ERROR(...))` hard-error for the unsupported case; **or**
  - **(Scenario 4 form 2 default)** write the runtime gating into the A5-only C++ skeleton (no legacy path needed); **or**
  - **report and stop**: "current CANN (x.y.z) has no Ascend950 kernel for `aclnnXxx`; needs upgrade (e.g. 9.1.0). Adaptation would build but crash on A5." Let the user decide.

---

## §5 Skip / Required Step Checklist

| Step | Action | Reason |
|------|--------|--------|
| ~~C-parse~~ | **Skip** | Interface signature unchanged; no .md to parse |
| ~~C-confirm~~ | **Skip** | No new params/options to confirm |
| ~~C-derivatives~~ | **Skip** | Autograd bindings unchanged |
| ~~C-constraints~~ | **Skip** | No new constraints to implement |
| C-yaml | **Required** | Add `op_api: all_version` line (see §6 → scenarios/scenario-N for full YAML) |
| **M-coverage** | **Required** | Verify the installed CANN ships an Ascend950 kernel for the exact aclnn variant, BEFORE C-cpp (see §4) |
| C-cpp | **Required** | Create `*KernelNpuOpApi.cpp` (see `m-adapt-cpp-patterns-guide.md` + per-scenario files) |
| C-meta | **Required** | Register Meta impl (see §8 below) |
| C-verify | **Required** | Cross-check YAML ↔ C++ ↔ Meta |
| C-build | **Required** | Code change requires recompile (see `build-guide.md` — pay attention to cgroup/MAX_JOBS) |
| C-test | **Required** | Add A5-specific tests (see `op-test` skill §M-adapt) |
| C-docs | **Required** | Doc must add the A5 product-support row (see `op-apidoc` skill) |
| C-baseline | **Required** | Update compatibility baseline |
| C-summary | **Required** | Generate integration summary (archiving as an m-adapt case strongly recommended) |

---

## §6 Scenario Routing

After Step 0 (state classification) and §4 (M-coverage), route to one of **4 scenarios**:

| Scenario | When | YAML template | C++ specifics | Real-world examples |
|----------|------|---------------|---------------|---------------------|
| **1. Standard M-adapt** | Operator exists on 910 (ACL) and A5 (ACLNN); legacy platforms keep ACL, A5 takes ACLNN | `acl_op:` + `op_api:` | Segment 1 `GetSocVersion() < Ascend950 ? acl_op : ...` | `npu_ciou`, `npu_iou` |
| **2. Backward hard-error** | A5 has no backward for this op (e.g. `npu_xxx_backward`); A5 must hard-error | `acl_op:` + `op_api:` (must keep `op_api:` so hard-error fires) | Segment 1 `GetSocVersion() >= Ascend950 ? TORCH_CHECK(false)` | `npu_ciou_backward` |
| **3. Pure ACLNN replacement** | Future evolution: legacy ACL path deprecated; everything goes through ACLNN | only `op_api:` | Segment 1 = no routing (legacy removed) | (reserved, no live examples yet) |
| **4. A5-only** | Brand-new operator, **only A5 has aclnn**; no 910 implementation | only `op_api:` (or `internal_format_opapi:` / `gen_opapi:`) | Segment 1 `GetSocVersion() >= Ascend950 ? <A5 path> : TORCH_CHECK(false)`; no `AclOpsInterface.h` include | 242 ops including `npu_swiglu_quant`, `npu_moe_distribute_dispatch_v2`, `npu_grouped_matmul`, `npu_apply_rotary_pos_emb` |

**Scenario 4 has 4 implementation forms** — pure forwarding, MC2-fused, multi-version gating, hard-error. See `scenarios/scenario-4-a5-only.md` §2 for the form-decision tree.

### YAML change checklist (Scenarios 1/2)

Before submitting the yaml change, confirm these 3 points:
- [ ] **Keep** the `acl_op: all_version` line (unless Scenario 3/4 explicitly removes it)
- [ ] **Add** the `op_api: all_version` line
- [ ] **Scenario 2 backward entry** must also have `op_api: all_version` (so the hard-error template fires)

For full per-scenario YAML templates, see `scenarios/scenario-1-standard.md` / `scenario-2-backward-harderror.md` / `scenario-3-pure-aclnn.md` / **`scenario-4-a5-only.md`**.

### `op_api: all_version` semantics

`op_api: all_version` in M-adapt mode means:
- ✅ "register the op_api path for all PyTorch versions"
- ❌ does **NOT** mean "all SoC platforms take the ACLNN path"

ACL vs ACLNN routing is decided by `GetSocVersion()` in C++; **the YAML does not distinguish SoC**.

---

## §7 C++ Implementation

> **Detailed C++ patterns have been extracted to `m-adapt-cpp-patterns-guide.md`.** This section is the routing summary.

| Concern | See `m-adapt-cpp-patterns-guide.md` § |
|---------|---------------------------------------|
| Boilerplate prelude (do not change) | §1 |
| Three-segment skeleton (universal) | §2 |
| Routing mechanism choice (GetSocVersion / runtime gating / DO_COMPATIBILITY) | §3 |
| Runtime kernel gating (multi-version aclnn) | §4 |
| Segment 2.1 dtype cast | §5 |
| Segment 2.2-2.3 infer output size + pre-allocate | §6 |
| Segment 2.4 invoke aclnn operator | §7 |
| Segment 3.1 dtype cast back | §8 |
| Segment 3.2 output transpose | §9 |
| Backward hard-error template | §10 |
| File naming convention | §11 |
| Full examples | §12 |
| M-adapt vs existing patterns | §13 |

**Per-scenario C++ templates**:
- Scenario 1: standard skeleton, see `m-adapt-cpp-patterns-guide.md` §2 + §3
- Scenario 2: backward hard-error, see `m-adapt-cpp-patterns-guide.md` §10
- Scenario 3: standard skeleton, drop Segment 1 ACL forwarding
- Scenario 4: see `scenarios/scenario-4-a5-only.md` §3-§6 (4 forms, each with full C++ template)

---

## §8 Meta Registration

### §8.1 Meta registration is **mandatory** in M-adapt mode

**Mandatory rules**:

1. In M-adapt mode, every new op_api entry MUST register a Meta in `_meta_registrations.py`
2. **User-facing API** `npu_xxx` → register `@impl(m, "npu_xxx")`
3. **Internal API** `_npu_xxx` (if exposed by C++) → register `@impl(m, "_npu_xxx")`
4. **Backward** `npu_xxx_backward` → does **NOT** need registration (consistent with Create mode: backward is not Meta-registered)

**Why mandatory**:
- A missing Meta makes `torch.compile` throw obscure errors on A5
- A5 commonly uses `torch.compile` to optimize training/inference
- Registration is cheap (a few lines of Python per entry)

### §8.2 Meta function body principles

- Only infer output shape and dtype; **do not perform actual computation**
- Build with `torch.empty(output_size, dtype=input.dtype, device='meta')`
- shape/dtype MUST **exactly match** the return value of the ACLNN path in the C++ OpApi

### §8.3 Per-scenario Meta patterns

For scenario-specific Meta templates (output shape alignment, multi-output tuple, etc.), see:

- Scenario 1 (standard): `scenarios/scenario-1-standard.md` §Meta
- Scenario 2 (backward hard-error): no Meta registration needed (§8.1 rule 4)
- Scenario 3 (pure ACLNN): same as Scenario 1
- Scenario 4 (A5-only): `scenarios/scenario-4-a5-only.md` §Meta (per form)

Common patterns:
- **aclnn output shape differs from user-facing API** (Scenario 1 example: aclnnIou outputs `(n,m)`, user expects `(m,n)`): Meta MUST be written to the **user-facing API's final shape**:

```python
@impl(m, "npu_iou")
def npu_iou_meta(bboxes, gtboxes, mode=0):
    # the user-facing API ultimately returns (m, n), not aclnn's (n, m)
    output_size = [gtboxes.size(0), bboxes.size(0)]
    return torch.empty(output_size, dtype=bboxes.dtype, device='meta')
```

- **Multi-output internal API** (Scenario 1 example: `_npu_ciou` returns `(overlap, atan_sub)`):

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

### §8.4 M-adapt Meta registration checklist

- [ ] User-facing API `npu_xxx` registered with `@impl(m, "npu_xxx")`
- [ ] Internal API `_npu_xxx` (if present) registered with `@impl(m, "_npu_xxx")`
- [ ] Meta body output_size matches the user-facing API's final return shape (**not** the aclnn output shape)
- [ ] Backward `npu_xxx_backward` is **NOT** Meta-registered

---

## §9 M-adapt Parameter Alignment Checklist (check before writing C++)

Before writing C++, **consult the aclnn operator doc** against the acl_op implementation and check each of these 5 items:

#### 1. Input dtype support
- [ ] Which dtypes does the aclnn operator support? (fp16/fp32/bf16?)
- [ ] If the original dtype is unsupported, do you need a dtype cast in C++?
- [ ] Always use `at_npu::native::custom_ops::_npu_dtype_cast(t, at::kFloat)` (takes ScalarType), NOT the legacy `npu_dtype_cast`

#### 2. Output shape alignment
- [ ] What is the aclnn operator's output shape?
- [ ] Does it match the acl_op implementation (or the user-facing API expectation)?
- [ ] If not: transpose (e.g. iou's `(n, m) → (m, n)`) or another dim transform?

#### 3. Backward support
- [ ] Does the aclnn operator have a backward?
- [ ] Does the original acl_op have a backward?
- [ ] If A5 lacks backward: hard-error template (`TORCH_CHECK(false, "...", OPS_ERROR(ErrCode::NOT_SUPPORT))`) — see `m-adapt-cpp-patterns-guide.md` §10
- [ ] If A5 supports backward: same GetSocVersion routing as forward

#### 4. Enum/string parameter conversion
- [ ] What enum type does acl_op use? int? bool? char*?
- [ ] What type does the aclnn operator expect? (often a const char* string)
- [ ] Conversion example: `const char *mode_str = mode == 1 ? "iof" : "iou";`

#### 5. aclnn new parameters
- [ ] Does the aclnn operator have "new params" absent from acl_op? (e.g. iou's `eps=0.01, aligned=false`)
- [ ] How are new params valued? (check aclnn doc; usually has defaults)
- [ ] Document the rationale for new param values in C++ comments

---

## §10 Build & Test Considerations

### §10.1 Build: cgroup / MAX_JOBS (mandatory pre-check)

`setup.py build` (line 369) reads `MAX_JOBS` from env, falling back to `multiprocessing.cpu_count()`. On a 128-core box inside a 32G cgroup (typical container), the default **silently OOM-kills `cc1plus` ~50+ times in a tight loop**. The error is `c++: fatal error: Killed signal terminated program cc1plus` — looks like a compiler bug, actually memory pressure.

Pre-check + mitigation, in order:

1. **Detect cgroup limit**:
   ```bash
   cat /sys/fs/cgroup/memory.max 2>/dev/null \
     || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null \
     || echo "no cgroup"
   ```
2. **Detect CPUs**: `nproc`.
3. **Set `MAX_JOBS` before invoking build.sh** (env passes through to `setup.py build`):
   - Conservative default: `export MAX_JOBS=4` (each `cc1plus` peaks at ~3-5 GB for typical op-plugin TUs).
   - Raise cautiously if memory headroom allows (e.g. 64G cgroup → `MAX_JOBS=8`).
4. **Pass `--jobs N` only if supported** by the underlying `build_ext`. `ci/build.sh` does NOT currently forward `-j`; use the env var.
5. **If you see `Killed signal terminated program cc1plus`** during build: STOP. Do not let it retry. Lower `MAX_JOBS` and restart (the previously-cloned `build/pytorch` is reusable; only `setup.py build bdist_wheel` re-runs).

Heuristic: `MAX_JOBS = min(nproc, cgroup_mem_GB / 4)`.

### §10.2 Build: `--pytorch` version (mandatory version self-check)

`--pytorch` is the branch/tag of ascend/pytorch. The branch naming scheme is `v<torch_ver>-<op_plugin_release>` (e.g. `v2.10.0-26.0.0`), and a bare `v<torch_ver>` (e.g. `v2.10.0`) frequently **also exists as a different branch**. Passing the bare form checks out the wrong tree and fails during PyTorch codegen.

```bash
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
RELEASE=$(git -C <op-plugin-root> rev-parse --abbrev-ref HEAD)
PYTORCH_TAG="v${TORCH_VER}-${RELEASE}"
# Verify the tag actually exists on ascend/pytorch BEFORE building:
git ls-remote https://gitcode.com/ascend/pytorch.git "${PYTORCH_TAG}" | grep -q . \
  && echo "OK: --pytorch=${PYTORCH_TAG}"
```

- **Never use `--pytorch=master`**: master is a rolling branch with an unstable version.
- **Never assume `--pytorch=v${TORCH_VER}` (bare)**: on suffixed release lines this points to the wrong branch.

### §10.3 Build: run command

```bash
# First run: full clone (15-40 min depending on network + parallel)
export MAX_JOBS=4
nohup stdbuf -oL -eL bash ci/build.sh --python=3.12 --pytorch=v2.10.0-26.0.0 \
  > build_log/build.log 2>&1 &
```

- `build/pytorch/` (cloned by build.sh) is reusable across rebuilds — only `setup.py build bdist_wheel` re-runs.
- After build: `pip install --force-reinstall --no-deps build/pytorch/dist/torch_npu-*.whl`

### §10.4 Test design notes

- **Platform isolation**: `@SupportedDevices(['Ascend950'])` for A5-specific tests, `@SupportedDevices(['Ascend910A', 'Ascend910B'])` for legacy paths.
- **Precision assertion strategy depends on op type**:
  - **Deterministic ops** (e.g. mish_backward): compare against hand-derived closed-form formula; fp32 max_abs_err ~1e-6, fp16/bf16 ~1e-2 acceptable.
  - **Non-deterministic ops** (yaml `tags: nondeterministic_seeded`, e.g. dropout): hand-formula does NOT apply — use `cpu_op_exec` reference path; tolerance accommodates seed-driven output drift.
  - **Multi-version aclnn**: test matrix must cover all CANN/A5 kernel combinations (V2-only CANN, V3+CANN, etc.).
  - **Scenario 4 (A5-only) tests**: skip the `@SupportedDevices(['Ascend910A', 'Ascend910B'])` decorator — the op simply doesn't exist on those platforms. Use `@SupportedDevices(['Ascend950'])` only, or omit the decorator and add an expected-failure test for legacy SoC.
- **A5 verification**: after install, run directly on the A5 device (`torch_npu.npu_xxx(...)`) — adapt `python3` script that constructs inputs, calls the op, and compares.

---

## §11 M-adapt Wrap-up

### §11.1 Case archiving (strongly recommended)

After completing M-adapt, **strongly recommended** to archive the integration as `references/examples/m-adapt/<op_name>-a5.md`:

- Helps future developers reference the acl_op → ACLNN routing pattern
- Helps surface common pitfalls (dtype cast, output shape alignment, aclnn new params, etc.)
- 5 case archetypes for Scenarios 1-3 (existing): see `references/examples/m-adapt/README.md` (case index + 4-dimension selection guide).
- 4 case archetypes for Scenario 4 (A5-only) — see `scenarios/scenario-4-a5-only.md` §8 for the case archive map; recommended additions:
  - `npu_prompt_flash_attention-a5-only.md` — Form 1 hard-error template (reverse-direction example: Ascend950 no longer supports)
  - `npu_moe_distribute_dispatch_v2-a5-only.md` — Form 2 multi-version runtime gating
  - `npu_grouped_matmul-a5-only.md` — Form 3 `internal_format_opapi`
  - `npu_moe_distribute_dispatch_setup-a5-only.md` — Form 4 `gen_opapi` structured

### §11.2 Archiving workflow

1. Copy `references/templates/case-template.md` to `references/examples/m-adapt/<op_name>-a5.md` (Scenario 1/2/3) or `references/examples/m-adapt/a5-only/<op_name>-a5-only.md` (Scenario 4)
2. Fill the "A5 Platform Specific Info" section
3. Add a row to `references/examples/m-adapt/README.md` (or `references/examples/m-adapt/a5-only/README.md` for Scenario 4)

### §11.3 Key decision quick reference

| Decision | Rule | See |
|----------|------|------|
| State classification | Step 0: A (hand-written) / B (gen_opapi) / C (standard candidate) / D (Create) / **E (A5-only)** | §2 |
| 4-condition AND | State C only | §3 |
| A5 kernel coverage | verify installed CANN has an Ascend950 kernel for the exact aclnn variant before writing C++ | §4 |
| Scenario routing | 1 standard / 2 backward / 3 pure / 4 A5-only | §6 |
| YAML change | per-scenario template + `op_api: all_version` semantics | §6 + `scenarios/` |
| C++ patterns | boilerplate / 3-segment / routing / runtime gating / segment details | `m-adapt-cpp-patterns-guide.md` |
| C++ runtime gating | `check_aclnn_kernel_available` + version fallback + "update CANN" hard-error (for version-dependent A5 kernels) | `m-adapt-cpp-patterns-guide.md` §4 |
| Meta registration | required; handle user-facing vs internal API separately | §8 |
| Backward unsupported | hard-error template | `m-adapt-cpp-patterns-guide.md` §10 |
| Test platform isolation | `@SupportedDevices` enforced (Scenario 4 = A5 only) | §10.4 + `op-test` skill |
| Doc update | must add A5 product-support row | `op-apidoc` skill |
| Case archiving | strongly recommended under `examples/m-adapt/` | §11.1 |

---

## §12 See Also

- `modify-guide.md` §M-discover / M-scope / M-gate / M-execute — Modify-mode orchestration
- `build-guide.md` — Build/compile workflow + cgroup/MAX_JOBS + --pytorch version
- `m-adapt-cpp-patterns-guide.md` — All C++ implementation patterns (boilerplate, skeleton, routing, runtime gating, segment details, file naming, examples)
- `scenarios/scenario-1-standard.md` — Standard M-adapt (forward, both platforms)
- `scenarios/scenario-2-backward-harderror.md` — Backward hard-error on A5
- `scenarios/scenario-3-pure-aclnn.md` — Pure ACLNN replacement (no legacy path)
- `scenarios/scenario-4-a5-only.md` — A5-only operator (4 forms)
- `references/examples/m-adapt/` — Real operator case archive (5 archetypes for Scenarios 1-3)
- `op-test` skill — M-adapt test design rules (`@SupportedDevices` isolation)
- `op-apidoc` skill — M-adapt doc rules (Atlas A5 product-support row)