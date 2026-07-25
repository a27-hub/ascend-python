# {OperatorName} Integration Checklist

> **Mode: CREATE** — use this template for new operator integration. For modify mode, see "Modify Mode Checklist" at the bottom.

## Parse Results

- aclnn function name: aclnn{OperatorName}
- torch_npu function name: npu_{operator_name}
- Input parameters: [list all input parameters and types]
- Output parameters: [list all output parameters and types]
- Key constraints: [list important constraints]
- Operator type: [Forward / Backward (grad) / SymInt]

## ATK Case Dir (C-delimit input, path-only)

> Record the CANN ATK case dir path if the user provided one. **Path only —
> do NOT read the files** (4 件套 atk.yaml / generate.py / executor.py /
> node.yaml，或 3 件套 case.json / executor.py / node.yaml——用例已生成);
> they are consumed by the op-delimitation subagent at C-delimit.

- ATK case dir: {path | not provided → C-delimit SKIPPED, note in C-summary}
  （有 ATK 输入时记录形态与文件，便于 C-delimit 组装 prompt，例如：
  `- ATK case dir: {path} | mode: 3file | files: case.json, executor.py, node.yaml | case_json: {path}`）

## Parameter Information Table (Cross-Skill Shared)

**C-parse generates this table. All subsequent steps (C-yaml through C-baseline) and sub-skills (op-test, op-apidoc) reference this table as the single source of truth for parameter definitions.**

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line | .md Description | Constraints |
|---|------------|--------|------|-------------|---------|----------|---------------|-----------------|-------------|
| 1 | {param} | both/.md only/_def.cpp only | Tensor | REQUIRED | - | - | L{xx} | {desc} | {constraints} |
| 2 | {param} | both/.md only/_def.cpp only | int? | OPTIONAL | 0 | OPTIONAL | L{xx} | {desc} | {constraints} |

**Source column**:
- `both` — parameter exists in both input files, consistent
- `.md only` — parameter exists in .md but not _def.cpp (= **exclude**, _def.cpp is authoritative)
- `_def.cpp only` — parameter exists in _def.cpp but not .md (= **⚠️ ask user in C-confirm**)
- `conflict` — parameter exists in both but with conflicting types/defaults (= **⚠️ ask user in C-confirm**)

## Information Required from User

- [ ] Are there aclnn version variants (V2/V3/V5)? If so, what are the trigger conditions for each version?
- [ ] Is SoC version dispatch needed (e.g., Ascend950 vs others)?
- [ ] Is a .List overload needed (a Tensor parameter that also supports int[] type)?
- [ ] Are there additional torch_npu function variants (e.g., _asymmetric, _v0)?
- [ ] Is the internal_format_opapi field needed?
- [ ] Is there special data format support (mxfp4, etc.)?
- [ ] Does this operator participate in autograd backward propagation? (If yes, derivatives.yaml configuration is needed)
- [ ] **Does this operator support graph mode (torch.compile)?**
  - [ ] If YES: execute C-meta, remove `--disable_torchair` from build, generate graph-mode test cases, include graph-mode call example in docs, C-verify must cross-verify meta shape/dtype against C++ _out_size
  - [ ] If NO: skip C-meta, keep `--disable_torchair` in build, skip graph-mode tests, omit graph-mode call example in docs, skip Meta Shape/Dtype cross-verification
- [ ] **Does this operator need ACLGraph update (手工 task-group update)?**（与上一题正交，见 confirm-guide Q2）
  - [ ] If YES:
    - [ ] Q2 门禁结论（confirm-guide）：两段式接口预检证据 = {文件+行 / 无→已停止}；workspace max 语义（CANN 文档核对，不得冒充）= {YES/NO}；预处理静态上界（代码审查）= {YES/NO}；shape/dtype/地址稳定+只影响 executor/tiling = YES（Q2=YES 蕴含，实现期核验，不成立 → 停止）
    - [ ] ⚠️ codegen-type 逐项决策：{param}: 保留 SymInt[]?（默认，IFA 形态）/ 降级 int[]?（需用户明确确认）
    - [ ] 交付：C-yaml 2 条目（`.out` + `_get_max_workspace`）+ C-cpp（helper + 2 函数，预处理限制转 TORCH_CHECK）+ C-test eager 三路径；不新增 `_infer_output`/meta/derivatives
  - [ ] If NO: skip all ACLGraph steps

### Conflict Confirmation (from Parameter Information Table)

> **CRITICAL**: Before asking the standard questions above, submit each ⚠️ item from the Parameter Information Table to the user for confirmation. Do NOT make unilateral decisions.

| # | Parameter | Conflict Description | Resolution |
|---|-----------|---------------------|------------|
| 1 | {param} | _def.cpp has this parameter but .md does not mention it | ⚠️ Awaiting user confirmation |
| 2 | {param} | Default value conflict: _def.cpp={X} vs .md={Y} | ⚠️ Awaiting user confirmation |

- [ ] All ⚠️ conflicts confirmed by user
  - [ ] {param}: ✅ Included / ❌ Excluded
  - [ ] {param} default: Use _def.cpp value ({X})

## Pitfall Alerts (from parse JSON `pitfall_alerts`)

> Deterministic structural alerts — surfaced regardless of operator type/scenario. Each binds specific steps; do NOT clear until that step read the case and applied it. Empty if parse found none.

| # | Alert id | Trigger | Must-read case | Gates steps | Applied? |
|---|----------|---------|----------------|-------------|----------|
| 1 | {id} | {trigger} | {must_read_case} | {applies_to_steps} | ☐ |

## Constraint Logic Checklist (Cross-Step Verification)

Extracted from .md constraints section (must be synchronized across all implementation steps):

| # | Constraint Rule | .md Source | C++ (C-cpp) | Meta (C-meta) | Test (C-test) | Doc (C-docs) |
|---|-----------------|------------|--------------|---------------|---------------|--------------|
| 1 | {e.g., scale tensor present → output quantized dtype} | {line num} | ☐ | ☐ | ☐ | ☐ |
| 2 | {e.g., parameter A present → parameter B required} | {line num} | ☐ | ☐ | ☐ | ☐ |

**Instructions**: Mark each checkbox after completing the corresponding step. Ensure constraint rules are consistently implemented across all layers.

## Constraint Coverage Self-Check (C-cpp+)

After C++ implementation, verify every constraint is implemented:

> For each constraint in the table above, search the C++ code for corresponding TORCH_CHECK or validation logic. If not found, add it immediately before proceeding to C-meta.

| # | Constraint Rule | C++ TORCH_CHECK Found? | Status |
|---|-----------------|----------------------|--------|
| 1 | {constraint} | Yes (line {L}) / No — added | ☐ |
| 2 | {constraint} | Yes (line {L}) / No — added | ☐ |

## Complex Operator Grouped Mode

> **Only fill this section when the grouped/branch mode is triggered** in C-parse (see `parse-guide.md` § Feature-Group & Dimension Analysis → "When to enable"). The trigger is a *semantic* judgment (≥2 independent conditional rules in `.md`, or output shape/dtype depending on ≥2 independent parameters) — **not** a parameter count. Skip this section for linear-shape operators even if they have many tuning knobs. Fill with values derived from the *current* operator's `.md` / `_def.cpp`; do not copy values from any example case without re-verifying against source.

### Feature Parameter Groups

| # | Group Name | Parameters | Intra-group Constraints | Version/Chip Differences | Shape Impact | C++ Pre-proc |
|---|-----------|------------|------------------------|------------------------|-------------|-------------|
| 1 | {group name} | {param names from _def.cpp} | {coexistence / mutex rules} | {version- or chip-specific} | {none / describe output change} | {Y / N} |
| 2 | {group name} | {param names} | {rules} | {-} | {none} | {N} |

### Dimension Branch Table

Describe each independent dimension that affects output shape/dtype on its own rows. Do not enumerate cartesian products across dimensions.

> **Non-orthogonal interaction warning**: when one dimension *overrides* a value another dimension would otherwise set (e.g. a flag that forces a specific output dtype regardless of another dimension's dtype rule), record the override rule explicitly — do not model such dimensions as independent. Add a row describing how the two combine.

| # | Dimension | Value | Affected Params/Groups | Output (name: shape/dtype) | Notes | C++ | Meta |
|---|-----------|-------|----------------------|--------------------------|-------|-----|------|
| 1 | {dim} | {value} | {params} | {out: {shape}; dtype rule} | {derivation note} | {Lxx / MISSING} | {Lxx / MISSING} |
| 2 | {dim} | {value} | {params} | {out: {shape}; dtype rule} | {note} | {Lxx / MISSING} | {Lxx / MISSING} |

## Parameter Coverage Checklist

All OPTIONAL Input parameters from _def.cpp must be covered in at least one test case (REQUIRED parameters are automatically covered in basic functionality tests):

| # | Parameter Name | Scenario | Test Case Name | Status |
|---|----------------|----------|----------------|--------|
| 1 | {param name} | {scenario} | {test_xxx} | ☐ |
| 2 | {param name} | {scenario} | {test_xxx} | ☐ |

## Table Migration Checklist

Constraint information in table format from .md must be fully migrated to API documentation:

| # | Table Content | .md Line | Doc Location | Status |
|---|---------------|----------|--------------|--------|
| 1 | {e.g., an enumerated-value parameter table from .md} | {line num} | Constraints section | ☐ |
| 2 | {e.g., quant_mode table} | {line num} | Constraints section | ☐ |

## Derivatives Configuration Checklist (C-derivatives)

> Only applicable when the operator participates in autograd (confirmed in C-confirm).

| # | Item | Details | Status |
|---|------|---------|--------|
| 1 | Forward operator identified in derivatives.yaml | Search for existing entries with same operator name | ☐ |
| 2 | Backward operator registered in op_plugin_functions.yaml | Verify backward op has a `func:` entry | ☐ |
| 3 | Derivatives entry added | Format: `{param}: backward_op(grad, result0, ...)` | ☐ |
| 4 | Placeholder correctness | `grad`=incoming gradient, `result0/1`=forward outputs, `grads[N]`=multi-gradient | ☐ |
| 5 | Version tag matches | Same version range as forward operator | ☐ |

## Compatibility Registration Checklist (C-baseline)

| # | Item | Details | Status |
|---|------|---------|--------|
| 1 | `op_api:` entry in schema JSON | Python-level signature with version tags | ☐ |
| 2 | `func:` entry in schema JSON | Underlying operator signature with version tags | ☐ |
| 3 | Version tags determined | Check current torch_npu version, use `"newest"` for new operators | ☐ |
| 4 | Entry added to allowlist | Added to `all_version` list (or specific version) | ☐ |
| 5 | Signature verified | Compare with `op_plugin_functions.yaml` func line and `inspect.signature()` | ☐ |

## Progress

- [ ] C-parse: Parse input files
- [ ] C-confirm: Confirm operator-specific information with user
  - [ ] **Graph mode support confirmed (YES/NO)**
- [ ] C-yaml: Generate YAML
- [ ] C-derivatives: Configure derivatives.yaml (if autograd applies)
- [ ] C-cpp: Implement C++
- [ ] C-constraints: Constraint self-check
- [ ] C-meta: Add Meta registration [conditional: 图模式 YES → execute, NO → skip]
  - [ ] Parameter signature matches YAML func: line (required→positional, optional→=None, attr→keyword-only)
  - [ ] Output count matches YAML -> (...) (single→direct return, multi→tuple)
  - [ ] All outputs use device='meta' or x.new_empty() (not default CPU)
  - [ ] rstd/scale/dgamma dtype fixed to torch.float32
  - [ ] INT4 packed output dtype=torch.int32 (not int8), last_dim // 8
  - [ ] Quantization dtype passed through TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP lookup
  - [ ] Communication operators: world_size <= 0 guarded
  - [ ] Optional input → corresponding gradient output is None (backward operators)
  - [ ] Backward dx shape matches corresponding forward input shape
  - [ ] C++ TORCH_CHECK (shape/dtype) → meta torch._check added where applicable
- [ ] C-verify: Signature verification [+ Meta Shape/Dtype cross-verify if 图模式 YES]
- [ ] C-build: Build [flag: 图模式 YES → remove --disable_torchair, NO → keep]
- [ ] C-test: Tests (UT + FakeTensor) [+ graph mode test cases if 图模式 YES]
- [ ] C-docs: API docs + update navigation files + update Python docstrings [+ graph mode call example if 图模式 YES]
- [ ] C-baseline: Compatibility baseline (schema JSON + allowlist)
- [ ] C-delimit: ATK cases run & locate/delimit via op-delimitation subagent [create only; conditional — skipped if no ATK input, record skip in C-summary]
- [ ] C-summary: Integration summary

---

# Modify Mode Checklist

> **Mode: MODIFY** — use this template when modifying an existing operator (not creating from scratch).

## Discovery (M-discover)

| File | Found? | Line(s) | Current Content (summary) |
|------|--------|---------|---------------------------|
| `op_plugin_functions.yaml` | ✅/❌ | L{xx} | {existing func entry} |
| `derivatives.yaml` | ✅/❌ | L{xx} | {existing binding} |
| `KernelNpuOutputSize.cpp` | ✅/❌ | L{xx} | {existing _out_size} |
| `KernelNpuOutputSize.h` | ✅/❌ | L{xx} | {existing declaration} |
| `_meta_registrations.py` | ✅/❌ | L{xx} | {existing meta} |
| `test_fake_tensor.py` | ✅/❌ | L{xx} | {existing test class} |
| `torch_npu_OpApi_schema_all.json` | ✅/❌ | L{xx} | {existing schema entry} |
| `allowlist_for_publicAPI.json` | ✅/❌ | L{xx} | {existing allowlist entry} |
| `op_plugin/ops/opapi/*.cpp` | ✅/❌ | L{xx} | {existing cpp file} |

## Impact Scope (M-scope)

- Change type: {AI derived from grep + description}
- Files to modify: {N}

| # | File | Action | Target Step |
|---|------|--------|-------------|
| 1 | {file} | {add / modify / append} | {C-yaml / C-cpp / ...} |

## Progress

**Disabled** (Create only — never execute):
C-parse, C-confirm, C-constraints

**Conditional** (based on impact scope):
- [ ] C-yaml
- [ ] C-derivatives
- [ ] C-cpp
- [ ] C-meta
- [ ] C-test
- [ ] C-docs
- [ ] C-baseline

**Always execute**:
- [ ] C-verify
- [ ] C-build
- [ ] C-summary
