---
name: op-test
description: Generate and execute UT test cases for torch_npu operators. Use when the user asks to write, create, or generate unit tests for torch_npu operators. Auto-detects the correct output directory. Generates test files with comprehensive test coverage. Executes tests to verify correctness. Skips UT execution if NPU hardware is unavailable (FakeTensor tests do not depend on NPU and always execute).
---

# torch_npu Operator UT Generation

## Data Passing Mechanism

CHECKLIST.md is the **main data channel** between steps. Each step reads structured state from CHECKLIST.md and writes its output back. The format is defined and enforced by scripts — do not invent your own sections. Operator `.md` docs and `references/` spec files serve as supplementary reference sources. They do not replace CHECKLIST.md for passing state between steps.

## Workflow

### Step 1: Initialize UT Case Framework Info (Bash)

**Goal**: Run a script. Extract all deterministic information from CHECKLIST.md and aclnn_input/ operator files (`_def.cpp`, `.md`) in one pass. Write structured tables: "Operator Basic Info", "Source File Paths", "Input/Output Spec".

```bash
python .claude/skills/op-test/scripts/scaffold_init.py --checklist {project_root}/CHECKLIST.md --search-dir {project_root}
```
`{project_root}` is the op-adaptation working directory (where CHECKLIST.md and aclnn_input/ reside).
Retry up to 3 times on failure. Block if still abnormal.

### Step 2: Supplement Parameter Semantics (Model)

Read the `.md` operator doc. Fill the pending columns in CHECKLIST.md `### Input/Output Spec` tables — the "Constraints" column (input table) and the "Shape Rule" column (output table).

> 🔒 **Only the "Constraints" and output "Shape Rule" columns are model-writable**. The input "Type" / "Supported dtypes" and output "Dtype Rule" columns are machine-written by step 1; altering them (e.g. `Tensor`→`Tensor?`, `, `→`;`, dtype enum→prose) breaks the step 3 parser → zero parameter coverage.

Filling spec (A/B/C/D structured reading, `Interaction: <A>×<B>` anchor format, platform-restriction exclusion, output Shape Rule format) → (MUST read `references/test-coverage-guide.md` §Step 2: Constraints 列填写规范).

**Blocking gate**: missing operator name / empty input param table / empty output spec → stop. Soft-missing items: annotate and continue.

### Step 3: Coverage Matrix Construction

**Script generates baseline matrix (Bash)**:
```bash
python .claude/skills/op-test/scripts/scaffold_coverage.py [--checklist <path>] [--dry-run]
```
Retry up to 3 times on failure. Block if still abnormal. Report script error messages.

**Model finalizes the matrix**: read the "Constraints" column and reshape the baseline into its final form via 4 actions — platform-exception delete / not-applicable delete / interaction expansion add / conditional omission. Every deletion must be logged in CHECKLIST `### Finalization Add/Delete Log` using the exact format `Delete <verbatim method name>: <reason>` (or `Add <...>: <reason>`) — `reconcile.py` parses this format; natural-language summaries are forbidden.

> 🛡️ Positive row count must not drop below the script baseline; every deleted row needs a logged reason. Full finalization spec (4-action details, anti-overdeletion guardrail, required-scenario non-regression, degradation-banner resolution) → (MUST read `references/test-coverage-guide.md` §Step 3: 覆盖矩阵终定规范).

### Step 4: Generate UT Test File (Model)

Input: Operator Basic Info + Coverage Matrix (step 3 final) + Input/Output Spec. Output: `test/{target_dir}/test_npu_{operator_name}.py`.

The matrix is finalized (rows already added/deleted in step 3). This step **translates each row into code. Every row in both matrices must be implemented. No omissions**:

> 📐 **MUST read `references/codebase-style-guide.md` before writing code** — This file distills the repo's mainstream style from 227 real test cases in `test/test_custom_ops/`. Covers: assertion function selection, device placement conventions, naming / decorator / seed conventions. Generated UT MUST match the repo's existing code style.

1. **Select reference implementation**: custom → (MUST read `references/patterns/custom-ops.md`). native → (MUST read `references/patterns/native-ops.md`). Modify-mode anchor append → (MUST read `references/modify-mode-guide.md`). v1/v2 comparison → (MUST read `references/patterns/v1v2-comparison.md`).
2. **Positive test cases**: Read the "Coverage Matrix" row by row. Generate 1 `test_` method per row. Construct legal inputs per the "Constraints" column (shape / value range / dtype linkage / conditions). Derive expected output shape per the "Shape Rule" column. Constraints drive data construction rules (MUST read `references/test-coverage-guide.md`).
   > 🔒 **Assertion Completeness Iron Rule** — every golden computed by the reference MUST be consumed by `assertRtolEqual` (shape/dtype-only assertion on a computed golden = fake test). Full criteria + per-dtype rules (int4 unpack, fp8 structure-only exemption, scale always compared) → `references/test-template.md` §Assertion Completeness Iron Rule. Enforced at sub-step 6 hard gate.
   > 🔒 **dst_type enum values** — scaffold emits `dst_type=<name>`; translate name→int via the canonical table in `references/test-template.md` §dst_type Enum Value (NOT aten `ScalarType.h` ordinals; fp8/hifloat8 MUST use `torch_npu.<name>` constants, never bare literals; truth: `op_plugin/python/meta/_meta_registrations.py:TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP`). Wrong int → `<X> has not been supported` (looks like op gap, is a magic-number bug).
3. **Negative test cases**: Read the `### Negative Coverage Matrix` row by row. Add 1 `test_neg_` method per row in the positive test class. Construct inputs per the "Violation Construction" code. Assertion / degradation / skipIf / construction code mapping rules, degradation criteria error code table → (MUST read `references/test-template.md` negative case spec).
4. **Standards compliance**: UT standards (TestCase / @SupportedDevices / manual_seed / assertRtolEqual / .clone / run_tests) + repo code style (`references/codebase-style-guide.md`: device placement, assertRtolEqual, naming consistency) → (MUST read `references/test-template.md` + `references/codebase-style-guide.md`).
5. **Completeness self-check + reconcile gate**: positive matrix rows == `test_` methods; negative rows == `test_neg_` methods — any mismatch means missed rows, fill them. Then run the mandatory hard gate:
   ```bash
   python .claude/skills/op-test/scripts/reconcile.py --checklist CHECKLIST.md --test-file test/{target_dir}/test_npu_{operator_name}.py
   ```
   Exit code 0 = pass → proceed to step 6; ≠0 = **block** → follow the dual-path guidance the script prints (implement the missing `test_`/`test_neg_` method, or record `Delete <exact_method_name>: <reason>` in `### Finalization Add/Delete Log` and re-run). `[EXTRA]` warnings do not block but should be resolved.
5b. **Reference implementation coverage self-check (A4)** — every matrix row's reference (`supported_op_exec`/`cpu_op_exec`) must produce a real golden; rows with no working reference path must be fixed or Delete-logged (fake shape-only tests are defects). Criteria + common gaps → `references/test-coverage-guide.md` §Reference Implementation Coverage.
6. **Assertion completeness reconciliation (hard gate)** — Scan each `test_` method against the Assertion Completeness Iron Rule in `references/test-template.md` (7 criteria, three axes: every golden built & value-consumed / sibling cases aligned / no redundant cases). MUST pass before proceeding to Step 5.
7. **Code style hard-check (A5)** — scan the test file against `references/codebase-style-guide.md` §0 and §4 (model self-check, not scripted; violations are **defects**): §0 `.cpu()` consolidated at `*_op_exec` exit (not scattered in assertion lines); §4 every `np.random.*` preceded by `np.random.seed(0)` in-scope — positive `test_` only; `test_neg_` exempt (verifies error, no golden compared → seed irrelevant). Verify each `test_` method; failures block progression to Step 5.

Generation failure (syntax error / file write failure / insufficient info in CHECKLIST.md) → block and report the specific cause.

### Step 5: Generate FakeTensor Test (Model)

Must not be skipped. Input: Input/Output Spec + Operator Basic Info. Output: Append a test class to `test/core_tests/test_fake_tensor.py`.

Spec (meta-only, does not depend on NPU) → (MUST read `references/fake-tensor-guide.md`). Locate anchor points and append (`class Test{Name}` / `if __name__`). Do NOT read the full file and rewrite. Generation failure → block and report the cause.

### Step 6: Execute Test Cases (Bash)

> 🔒 **Hard pre-flight gate**: Before any test execution, run `reconcile.py`. Exit code ≠ 0 → **block**, do NOT proceed to test execution. Unpassed reconcile = coverage not verified = test results are unreliable.

Verify Python dependencies (`decorator`, `tornado`, `numpy`). Detect NPU hardware:
```bash
python -c "import torch; print(torch.npu.is_available())"
```
If NPU is unavailable, skip UT execution (FakeTensor does not depend on NPU, always execute).
```bash
# UT (execute only when NPU is available)
python test/{target_dir}/test_npu_{operator_name}.py

# FakeTensor (does not depend on NPU, always execute)
python test/core_tests/test_fake_tensor.py {TestClassName}
```

Exit code 0 means all pass → proceed to step 7. Non-zero means there are failures. Read the output. Locate the failing methods. Classify root cause using the three-branch triage below. Distinguish "UT test case bug" vs "adaptation source bug" vs "infra issue".

**Execution failure root cause self-check (three-branch triage)**: The UT contains an NPU-independent CPU reference implementation (custom → `supported_op_exec`; native → `cpu_op_exec`). This is the independent gold standard. For each failing test method, **run its CPU reference implementation in isolation**. Classify by three states:

1. CPU reference errors / NaN / wrong shape → The test case's own reference implementation is wrong → **UT test case bug** → Fix the test case.
2. CPU reference passes, but NPU output does not match CPU → Gold standard is correct, NPU deviates → **Adaptation source bug** (C++ kernel / meta behavior mismatch) → Report to user. Do NOT modify the UT.
3. CPU reference passes, NPU also matches (re-run passes) → **Infra issue** → Re-run.

Handle each failing method one by one. Do not roll back the entire flow. Verdict based on runtime execution (mechanically executable). Do not guess by reading code statically.

### Step 7: Report Results (Model)

On success: pass count. On failure: error messages + suggested fixes. On skip: inform user. Generate structured test report:
```markdown
  **NPU Availability**: {Available/Unavailable}
  **Test Execution Status**: {Executed/Skipped}
  **Test Results**: Passed: {N} / Failed: {N} / Skipped: {N}
  {If failed, list error messages}
```
Include in op-adaptation C-summary.
