# Parse Guide (C-parse)

Detailed instructions for the parse step in create mode. Reference this guide when executing C-parse.

## Run the extractor — deterministic extraction + heuristic DRAFT

`scripts/op_adapt.py parse --md <path> --def <path> [--op-name <name>]` → JSON. The JSON feeds the **Parameter Information Table** in `CHECKLIST.md` (single source of truth for all later steps).

**Deterministic** (trust):
- From `_def.cpp`: inputs (name + optionality from `.ParamType()`), outputs, attrs (name + default + type hint + attr type from `.Bool()/.String()/.Int()/...`), `.Version()` markers, `DT_*` dtype constants. Per-chip config blocks (`OpAICoreConfig`, `GetKirinCoreConfig`) are automatically excluded.
- From `.md`: aclnn function name, `GetWorkspaceSize` signature params (camelCase normalized to snake_case; `workspaceSize`/`executor`/`**` pointer-to-pointer outputs filtered), constraint rows from Markdown pipe tables + HTML tables + freetext (each tagged with `is_likely_constraint` + `table_type` = constraint / error_code / param_desc / unknown).
- Cross-source: parameter reconciliation by normalized name → source (`both` / `_def.cpp only` / `.md only`), conflict list (`missing_in_md` / `missing_in_def` / `attr_type_unknown`) with suggested resolutions, operator type (`_grad` suffix → Backward), existing `derivatives.yaml` binding lookup.

**Heuristic DRAFT — LLM must verify** (the script cannot do semantic matching):

1. **False conflicts from name variants** — name normalization handles suffixes (`Optional`/`Out`/`Input`/`Output`/`Workspace`) and camelCase→snake_case, but cannot match semantically-identical params with different base names (e.g. `.md` `self`/`out` ↔ `_def.cpp` `x`/`y`). When the script flags `missing_in_md`/`missing_in_def` for such pairs, reconcile by **position + type + role**, not normalized name alone. Drop the false conflict; keep one canonical entry.
2. **Constraint logic** — the script returns raw table rows, each flagged with `is_likely_constraint` and `table_type`. Focus on rows where `is_likely_constraint=true` (error-code and pure workspace/executor/stream description rows are already flagged `false`). From those, derive the *logic*: dtype dependencies (param presence → output dtype), mutual-exclusion / coexistence, scenario branches. Record in the CHECKLIST "Constraint Logic" section.
3. **Version depth** — the script lists only the `.Version(...)` marker name. Read `_def.cpp` to (a) resolve the version constant value (e.g. `constexpr uint32_t FOO_VERSION_TWO = 2;`) so marker → version number is known, (b) group **ports** by version number — `.Version()` can attach to `Input(...)` / `Output(...)` / `Attr(...)` (e.g. `advance_step` marks V2-introduced **Inputs** `spec_token`/`accepted_num`, not attrs); ports without `.Version()` = base/V1, ports with `.Version(N)` = introduced in Vn — enumerate **all** N found, not just 2, and (c) cross-check against `.md`: does it reference versioned aclnn entry points (`aclnnXxx` / `aclnnXxxV2` / `aclnnXxxV3` / ...)? A single `_def.cpp` may pair with multiple `.md` (one per entry point). Feed results into C-confirm.
4. **Disabled params** — "not enabled" comments in `_def.cpp`; the script ignores comments. Mark these OPTIONAL.
5. **SymInt types** — the script does not detect SymInt (symbolic-shape integer) params. Scan `_def.cpp` for SymInt inputs/attrs and flag them; they require special typing in YAML/C++ (see `yaml-guide.md`).

## LLM action

1. Run `op_adapt.py parse --md <path> --def <path> [--op-name <name>]`.
2. Verify the 5 heuristic blind spots above against the raw `_def.cpp` / `.md`.
3. Apply source priority policy (see below).
4. Create `<project_root>/CHECKLIST.md` from `references/templates/checklist-template.md` (project-root resolution per `SKILL.md` Rules): record aclnn name, npu name, op type in "Parse Results"; fill the Parameter Information Table. Mark any still-unresolved conflict ⚠️ "awaiting C-confirm" — **never decide unilaterally**.
5. **Check `pitfall_alerts`** in the parse JSON (deterministic — fires on structural signals like "bool attr + ≥2 outputs", so it catches cross-cutting traps that a type/scenario case-match misses). Copy each alert verbatim into CHECKLIST.md "Pitfall Alerts". Treat a populated alert as a hard gate: don't clear it until its bound step (C-cpp / C-meta) has read the case and applied it.

## Source Priority Policy

Priority, highest to lowest:

1. User's explicit instructions in C-confirm (highest)
2. `_def.cpp` defined parameters (authoritative operator definition)
3. `.md` described parameters (aclnn API documentation)
4. Reference implementation parameters (NEVER auto-include)

The script already reconciles by normalized name and emits conflicts with suggested resolutions (`missing_in_md` → awaiting user confirmation; `missing_in_def` → exclude; `attr_type_unknown` → awaiting user confirmation). The LLM's policy call beyond the script:

- **Conflicting defaults** — the script does not extract `.md` param defaults, so it cannot detect default conflicts between sources. If the `.md` text describes a default that differs from `_def.cpp`, prioritize the `_def.cpp` Attr default; mark ⚠️ in the CHECKLIST Conflict Confirmation table. **Never decide unilaterally — actual resolution happens in C-confirm.**
- **`.md`-only params** that are NOT name-variant false conflicts → exclude (`_def.cpp` authoritative), per the script's `missing_in_def` resolution.

## Operator Type Implications

The script auto-detects Forward vs Backward from the `_grad` suffix. Implications for later steps:

| Pattern | Type | Implication |
|---------|------|-------------|
| `npu_xxx_grad` / `aclnnXxxGrad` | Backward (gradient) | C-yaml registers this op; C-derivatives does NOT fire; user may need to update the forward op's derivatives binding |
| `npu_xxx` (no `_grad` suffix) | Forward | C-yaml registers this op; if autograd, C-yaml also registers the backward op, and C-derivatives adds the binding |

## Feature-Group & Dimension Analysis (LLM semantic — script does not cover)

> These are semantic judgments the script **cannot** make: it has no parameter grouping, no output shape/dtype derivation, and no aclnn-layer version enumeration. The basic extractions they build on are already handled elsewhere — do **not** re-do them here:
> - disabled-param comments → §"Heuristic DRAFT" blind spot #4;
> - default-value conflict flagging → §"Source Priority Policy";
> - `.Version()` listing + DataType→V1/V2 mapping → blind spot #3;
> - constraint dtype/mutual-exclusion/scenario logic → blind spot #2.
>
> **When to enable the grouped/branch mode** (Feature Parameter Groups + Dimension Branch tables). Trigger vs screening are deliberately separated — a high parameter count alone does **not** trigger the mode.
>
> - **Trigger** (semantic judgment — enable the mode when either holds):
>   1. The `.md` constraints section contains ≥2 *independent* conditional rules (each rule's effect on output does not depend on the other), **or**
>   2. The output shape/dtype depends on ≥2 *independent* parameters/dimensions.
> - "Independent" and "depends on" are **semantic judgments**, not grep-able predicates. Make the judgment from the `.md` constraint text and the aclnn signature, and record the reasoning in CHECKLIST.
> - **Screening hint only** (does not trigger, just flags attention): when the operator has many optional parameters. A high optional count is a weak proxy — it raises the *suspicion* that trigger condition 1 or 2 may hold, so double-check them, but do not enable the mode on count alone. (A linear-shape operator with many tuning knobs must not be dragged into the heavy flow.)

**Parameter grouping** (only when the mode is triggered above): organize parameters into feature groups by functional semantics. Prefer the .md's own grouping structure; when no explicit grouping exists, identify groups by co-occurrence constraints (multiple parameters appearing together in the same conditional constraint) and shared semantic role. Record in CHECKLIST "Feature Parameter Groups" table.

**Shape impact annotation** (for each group): Does activating this group change the output tensor shape or dtype? If yes, describe the concrete rule in the "Shape Impact" column as derived from the operator's own .md / aclnn signature (do not assume a formula from other operators). Mark "C++ Pre-proc" as Y when the group requires shape calculation branches beyond the base path; N when parameters pass through to aclnn unchanged.

**Independent dimension identification** (when the mode is triggered, see trigger block above; specifically trigger condition 2): describe each dimension separately, do not enumerate cartesian products. For each dimension, enumerate **all** valid values found in the input files (do not assume a subset — check `_def.cpp` defaults, `.md` constraint tables, and aclnn signatures for the complete value set), record its values, affected parameters, impact on output shape/dtype, and conditional output existence. Record in CHECKLIST "Dimension Branch" table. **Tensor dimensionality**: if a tensor's accepted rank varies (e.g., 3D/4D/5D) and shape formulas differ by rank, create separate rows per rank.

**Non-orthogonal interaction check**: when one dimension *overrides* a value another dimension would otherwise set, the dimensions are not independent. Record the override rule explicitly (add a row describing how they combine) rather than treating them as orthogonal.

**aclnn-layer version variants** — `.Version()` is not the only versioning mechanism. Some operators carry version variants at the aclnn API layer instead (multiple `aclnnXxxV1/V2/...` entry points with no `.Version()` in _def.cpp). If the aclnn signature or `.md` references multiple versions but `_def.cpp` has no `.Version()` markers, enumerate the aclnn variants by reading the aclnn header files and record them for C-confirm version confirmation.

## Validation Report Template

```
Parameter Completeness Validation Report:
===================
Input Parameters:
  - x: ✅ Processed
  - weight: ✅ Processed
  - bias: ✅ Processed
  - per_token_scale: ⚠️ Defined in _def.cpp but not in YAML

Attr Parameters:
  - split_item: ✅ Processed
  - group_type: ⚠️ Default conflict (-1 vs 0)

Recommended Actions:
  1. Add per_token_scale parameter to YAML
  2. Confirm group_type default value
```

## Scope Control

**ONLY implement what is explicitly in input files or confirmed by user.**

**DO NOT add:**
- Extra parameters not in input files
- Extra function variants not requested by user
- Features from reference implementations unless they exist in input files

**Examples:**
- Input file has `dst_type` parameter → ✅ Implement it
- Input file has no `quant_mode` parameter → ❌ Do NOT add it
- Reference implementation has `quant_mode` → ❌ Still do NOT add it (unless in input files)

`_def.cpp` is authoritative over `.md`; when they conflict, ask user in C-confirm.

**Input files = `.md` + `_def.cpp` only.** CANN ATK case files (4 件套 atk.yaml /
generate.py / executor.py / node.yaml，或 3 件套 case.json / executor.py /
node.yaml) are NOT parse inputs — they belong to
the op-delimitation subagent (C-delimit). Do not read them, do not derive
parameters/constraints from them; only record the case-dir path in
CHECKLIST.md §ATK Case Dir (or leave it `not provided` → C-delimit skipped).

## Troubleshooting

(Accumulate from cases.)
