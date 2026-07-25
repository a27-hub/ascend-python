# Modify Mode Guide

---

## M-discover: Grep & Status Report

`scripts/op_adapt.py discover --op-name <name>` → JSON with search results across all 8 file categories (`op_plugin_functions.yaml`, `derivatives.yaml`, `KernelNpuOutputSize.cpp/h`, `_meta_registrations.py`, `test_fake_tensor.py`, `torch_npu_OpApi_schema_all.json`, `allowlist_for_publicAPI.json`) plus the `op_plugin/ops/opapi/` directory, and a `derivatives_binding` fact (direction: `forward_has_backward` / `backward_referenced_by_forward`, with `name_line` + `formulas`). The script strips `_v<N>` suffixes and uses word-boundary matching so variants are found but unrelated ops (e.g. `_asymmetric`) are not. **Deterministic** — trust the grep results.

**Present findings as a status table** (format the script's `files` output — the LLM's job, not the script's):

```
operator_name status in codebase:

  ✅ op_plugin_functions.yaml            — L6097: func: npu_fast_gelu(...)
  ✅ derivatives.yaml                    — L50: forward_has_backward
  ❌ KernelNpuOutputSize.cpp/h           — not present
  ✅ _meta_registrations.py              — L2514: @impl(m, "npu_fast_gelu")
  ✅ test_fake_tensor.py                 — L1175: def test_npu_fast_gelu(self)
  ✅ torch_npu_OpApi_schema_all.json     — L967: "func: npu_fast_gelu(...)"
  ❌ allowlist_for_publicAPI.json        — not present (expected for backward ops)
  ✅ op_plugin/ops/opapi/                — FastGeluNpuOpApi.cpp
```

**CRITICAL — derivatives binding (semantic, LLM-driven)**: if `derivatives_binding` is present, use the script's direction fact to ask the user the rebind question BEFORE deriving the impact scope. Example for a new backward variant:

```
ℹ️  derivatives.yaml binding:
  forward {forward_op} ──→ backward {original_backward}

  ⚠️  New backward variant detected ({new_variant}). Rebind forward's backward to {new_variant}?
      If yes: new variant takes effect in autograd training automatically.
      If no: {new_variant} can only be called manually.
```

Fill in the actual operator names from the script output. Tailor the benefit line to the specific variant's purpose (use the modification description — e.g. simpler API, fewer tensor inputs, memory savings, performance). The user's answer determines whether `derivatives.yaml` goes into the impact scope (M-scope).

## M-scope: Derive Impact Scope

Based on the modification description + grep results, determine the change type and affected files. User confirms the scope within this step:

| Change type | Trigger | Files to modify |
|-------------|---------|-----------------|
| Add variant — forward | New forward op variant (`_v2`, `_v3`) | yaml(forward+backward) + meta + output_size + test + json + derivatives(new binding for new backward) |
| Add variant — backward | New backward op variant | yaml + meta + output_size + test + json + derivatives(rebind forward if confirmed in scope analysis) |
| Add variant — standalone | New variant not in autograd chain | yaml + meta + output_size + test + json |
| Interface change | Parameter types/count changes | yaml + meta + output_size + cpp(if hand-written) + test + json + derivatives(if bound) |
| Backward supplement | Add/modify backward binding | derivatives + yaml(if backward op not registered) + test + json |
| Bug fix | Logic correction, no interface change | cpp + test(possibly) |

**Present the derived scope for user confirmation:**

```
Change type: Interface change (params tokens Tensor → tokens_size_0 int + tokens_dtype ScalarType)

Affected files (5):
  1. op_plugin_functions.yaml         — modify: func signature replace params
  2. KernelNpuOutputSize.cpp/h        — modify: _out_size function params
  3. _meta_registrations.py            — modify: meta function params
  4. test_fake_tensor.py              — append: new test methods + boundary cases
  5. torch_npu_OpApi_schema_all.json  — append: new func entry
  (derivatives.yaml — user confirmed during scope confirmation)

Confirm scope? Any additions?
```

After user confirms, generate a lightweight `<project_root>/CHECKLIST.md` (see `references/templates/checklist-template.md` "Modify Mode" section; project-root resolution per `SKILL.md` Rules).

Check `references/examples/README.md` for cases matching the same change type or affected files. If found, present the case's key decisions and pitfalls to the user as reference. Proceed with normal workflow regardless.

**Scope Control**: ONLY modify files in the confirmed impact scope. Do NOT expand beyond what M-scope derived. If new dependencies are discovered during execution, pause and re-derive scope.

## M-gate: Gate Check

Before making any changes:
```bash
git status --porcelain
```
If there are uncommitted changes, warn: "Working directory has uncommitted changes. Recommend committing or stashing before proceeding."

## M-execute: Execute

Map each file in the confirmed impact scope to its corresponding step, then execute only those steps plus the three always-required steps:

**File → Step mapping:**

| Impact scope file | Execute Step |
|-------------------|-------------|
| `op_plugin_functions.yaml` | C-yaml |
| `derivatives.yaml` (new binding) | C-derivatives |
| `derivatives.yaml` (rebind, backward variant) | C-derivatives (rebind section) |
| `KernelNpuOutputSize.cpp/h` or `*KernelNpuOpApi.cpp` | C-cpp |
| `_meta_registrations.py` | C-meta |
| `test_fake_tensor.py` or new UT file | C-test |
| `torch_npu_OpApi_schema_all.json` / `allowlist_for_publicAPI.json` | C-baseline |
| Existing doc page changed | C-docs (remind manual update) |
| New variant (new operator name) | C-docs (op-apidoc for new page) |

**Meta Sync Decision Flow** — use this before executing any step:

```
Is _meta_registrations.py in the impact scope?
├── YES → Has _out_size changed?
│   ├── YES → C-meta must execute. After meta update, C-verify §2 must execute.
│   │         Check: signature match? shape logic match? dtype match? constraint match?
│   └── NO  → Is this a new variant?
│       ├── YES → C-meta must execute (new @impl for new variant name).
│       └── NO  → C-meta may skip, but C-verify still checks existing meta ↔ C++.
│
└── NO  → Does the change affect _out_size / shape / dtype logic?
    ├── YES → ⚠️  Operator has no meta. Ask user: add graph mode support?
    │         [YES] → add _meta_registrations.py to scope, execute C-meta.
    │         [NO]  → skip. Note in summary: "graph mode not supported."
    └── NO  → C-meta not needed.
```

**Always execute** (regardless of impact scope):
- **C-verify** — Signature verification (modified files need cross-checking).
  - §1 Signature check: always.
  - §2 Meta Shape/Dtype Logic Verification: only when `_meta_registrations.py` or `_out_size` is in scope. **This is the graph-mode correctness gate.**
- **C-build** — Build (any code change requires recompilation).
  - If meta changed → rebuild without `--disable_torchair` to include torchair for graph mode.
- **C-summary** — Summary (always generate integration summary).

C-parse, C-confirm, C-constraints are create-mode only — skip in modify mode.

Operate each step in modify mode (edit existing files / append entries). See `SKILL.md` for each step's mode-specific instructions.

## M-adapt: A5 Platform Adaptation Submode

> **M-adapt content is split across multiple files.** This section is the entry point in Modify mode. Main orchestration lives in `m-adapt-guide.md`; C++ implementation patterns in `m-adapt-cpp-patterns-guide.md`; per-scenario YAML/C++/Meta templates in `scenarios/scenario-N-*.md`.

**Quick reference for M-adapt**:

| Concern | See |
|---------|------|
| What M-adapt does / when to use | `m-adapt-guide.md` §1 Overview & Scope |
| Step 0 state pre-classification (A / B / C / D / **E**) | `m-adapt-guide.md` §2 |
| 4-condition auto-detection (State C only) | `m-adapt-guide.md` §3 |
| A5 kernel availability verification (**M-coverage**) | `m-adapt-guide.md` §4 |
| Skip/required step checklist | `m-adapt-guide.md` §5 |
| **Scenario routing** (1 standard / 2 backward / 3 pure / **4 A5-only**) | `m-adapt-guide.md` §6 |
| YAML per-scenario templates | `scenarios/scenario-1-standard.md` / `scenario-2-backward-harderror.md` / `scenario-3-pure-aclnn.md` / **`scenario-4-a5-only.md`** |
| C++ implementation (boilerplate, 3-segment, routing, runtime gating) | `m-adapt-cpp-patterns-guide.md` |
| Meta registration (mandatory rules, scenarios) | `m-adapt-guide.md` §8 + scenario files |
| Parameter Alignment Checklist (5 items) | `m-adapt-guide.md` §9 |
| Build (cgroup/MAX_JOBS, --pytorch version) | `m-adapt-guide.md` §10 |
| Test design notes | `m-adapt-guide.md` §10.4 |
| Wrap-up / case archiving | `m-adapt-guide.md` §11 |

### M-scope user confirmation

Once M-discover reports an M-adapt suggestion, the M-scope step **MUST ask the user**:

> Operator X meets the 4 M-adapt conditions. Run it as the A5 platform adaptation submode?

- User confirms → proceed with M-adapt (load `m-adapt-guide.md` §5–11 for the full workflow)
- User says "No, I want to change a feature of X" → route to plain Modify mode

### Full examples

- **`references/examples/m-adapt/npu_ciou-a5.md`**: simple single-aclnn (fp16→fp32 cast, backward hard-error)
- **`references/examples/m-adapt/npu_iou-a5.md`**: simple single-aclnn (output transpose)
- **`references/examples/m-adapt/npu_mish_backward-a5.md`**: end-to-end success (acl_op-only → op_api, with build/test/validate)
- **`references/examples/m-adapt/npu_moe_distribute_dispatch_v2-a5.md`**: multi-version aclnn + `check_aclnn_kernel_available` runtime gating
- **`references/examples/m-adapt/npu_swiglu-a5.md`**: gen_opapi structured adapter (state B negative case)

### Case archiving

After completing M-adapt, archive the integration as `references/examples/m-adapt/<op_name>-a5.md` per the template + `examples/m-adapt/README.md` case selection guide. See `m-adapt-guide.md` §11.

## Mode-Specific Differences

| Step | Create Mode | Modify Mode |
|------|-------------|-------------|
| C-parse, C-confirm | Parse .md+_def.cpp, confirm with user | **Skip** — replaced by M-discover~M-execute |
| C-yaml | New entry | Edit existing or append variant |
| C-derivatives | New binding | Update existing or add new |
| C-derivatives (rebind) | — | Rebind forward→backward when adding backward variant (see `derivatives-guide.md` Rebind section) |
| C-cpp | New .cpp file | Edit existing or add function (e.g., `_out_size` in `KernelNpuOutputSize.cpp`) |
| C-constraints | Constraint self-check | **Skip** (no .md parsing in modify mode) |
| C-meta | New @impl registration | Edit existing or append |
| C-test | New test file + class | Append methods to existing class |
| C-docs | New doc page | Remind manual update; if new variant → new page |
| C-baseline | New entries | Append new or update existing |

## Troubleshooting

(Modify mode specific issues. Accumulate from cases.)
