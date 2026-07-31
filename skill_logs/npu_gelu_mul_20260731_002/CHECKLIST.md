# npu_gelu_mul Integration Checklist

run_dir: /home/llxxyy/op-plugin/skill_logs/npu_gelu_mul_20260731_002

> **Mode: CREATE** — new operator integration. User deleted existing npu_gelu_mul code; regenerating from input files (no git restore). Based on `after` branch.

## Project Constants

| Constant | Value | Set by | Status |
|----------|-------|--------|--------|
| `${OP_PLUGIN_ROOT_PATH}` | /home/llxxyy/op-plugin | SKILL.md §Rules (C-parse) | ☑ |
| `${OP_PLUGIN_SCRIPT_PATH}` | /home/llxxyy/.traecli/skills/op-adaptation/scripts | SKILL.md §Rules (C-parse) | ☑ |
| `${OP_PLUGIN_SKILL_HASH}` | non-git-skill-dir (skill dir is not a git repo) | bootstrap.sh | ☑ |
| `${OP_PLUGIN_SKILL_BRANCH}` | non-git-skill-dir | bootstrap.sh | ☑ |
| `${OP_PLUGIN_BUILD_PATH}` | /home/llxxyy/op-plugin/build | bootstrap.sh (285G free) | ☑ |
| `${CCACHE_DIR}` | (default ~/.ccache, 6332G free — OK) | bootstrap.sh | ☑ |
| `${MAX_JOBS}` | TBD | pre_build_check.sh (C-build) | ☐ |
| `${PYTORCH_TAG}` | TBD | pre_build_check.sh (C-build) | ☐ |

## Parse Results

- aclnn function name: aclnnGeluMul
- torch_npu function name: npu_gelu_mul
- Input parameters: input (Tensor, REQUIRED)
- Attr parameters: approximate (str, OPTIONAL, default="none")
- Output parameters: output (Tensor, REQUIRED) — .md uses name `out` (false-conflict name variant; merged by position+role)
- Key constraints:
  - input: 最后一维偶数且 ≤1024；其他维度乘积 ≤200000；dtype BFLOAT16/FLOAT16/FLOAT；格式 ND；维度 2-8；支持空Tensor
  - approximate: 只支持 "none" 和 "tanh"，空指针为 "none"
  - output: dtype 与 input 一致；shape 其他维度与 input 一致；最后一维 = input 最后一维 / 2；dtype BF16/FLOAT16/FLOAT；ND；维度 2-8
  - 确定性计算：默认确定性
  - 产品支持：Ascend 950PR/950DT、A3、A2 支持；310b/310p/910 不支持
- Operator type: Forward (no `_grad` suffix)
- _def.cpp chips: ascend910b, ascend910_93, ascend950, kirinx90, kirin9030
- Version markers: none (no `.Version()`)
- SymInt params: none
- Grouped mode: NOT triggered (single shape rule: output last dim = input last dim / 2; output dtype follows input — no ≥2 independent conditional rules, no ≥2 independent dimensions)

## ATK Case Dir (C-delimit input, path-only)

- ATK case dir: not provided → C-delimit SKIPPED, note in C-summary

## Parameter Information Table (Cross-Skill Shared)

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line | .md Description | Constraints |
|---|------------|--------|------|-------------|---------|----------|---------------|-----------------|-------------|
| 1 | input | both | Tensor | REQUIRED | - | - | L24 | 输入张量，公式中的input | 最后一维偶数且≤1024；其他维乘积≤200000；dtype BF16/F16/FLOAT；ND；维度2-8；支持空Tensor |
| 2 | approximate | both | str | OPTIONAL | "none" | OPTIONAL | L31 | Gelu计算模式 | 只支持"none"/"tanh"；空指针为"none" |
| 3 | output | both (name variant: .md `out` ↔ _def.cpp `output`) | Tensor | REQUIRED | - | - | L33 | 输出张量 | dtype与input一致；shape其他维与input一致；最后一维=input最后一维/2；dtype BF16/F16/FLOAT；ND；维度2-8 |

**Conflict resolution (C-parse LLM judgment)**:
- `out` (.md) ↔ `output` (_def.cpp): name-variant false conflict — merged by position (output slot) + type (Tensor) + role (operator output). Single canonical entry `output`.

## Standard Questions Tracker (C-confirm MANDATORY gate)

| # | Question | Asked? | User Answer |
|---|----------|--------|-------------|
| 1 | CANN version dispatch (V2/V3/V5)? | ☐ | |
| 2 | SoC-specific dispatch (e.g., Ascend950 vs others)? | ☐ | |
| 3 | .List overload needed? | ☐ | |
| 4 | Additional torch_npu function variants? | ☐ | |
| 5 | internal_format_opapi field needed? | ☐ | |
| 6 | Special data format support (mxfp4, etc.)? | ☐ | |
| 7 | Autograd requirements? | ☐ | |
| 8 | SymInt type parameters? | ☐ | |
| 9 | Graph mode support (Q1: torch.compile + Q2: ACLGraph)? | ☐ | |

## Information Required from User

- [ ] CANN version variants? (none detected in _def.cpp / .md — confirm)
- [ ] SoC dispatch? (.md lists 950/A3/910b support; _def.cpp adds kirin — confirm whether dispatch needed)
- [ ] .List overload? (input is single Tensor — likely no)
- [ ] Additional variants? (_out/_v0 etc.)
- [ ] internal_format_opapi?
- [ ] Special data format (mxfp4 etc.)?
- [ ] Autograd? (Forward op — does it need backward?)
- [ ] SymInt? (none detected)
- [ ] Graph mode Q1 (torch.compile) + Q2 (ACLGraph)?

### Conflict Confirmation (from Parameter Information Table)

| # | Parameter | Conflict Description | Resolution |
|---|-----------|---------------------|------------|
| 1 | out/output | name-variant false conflict (.md `out` ↔ _def.cpp `output`) | ✅ Resolved at C-parse: merged by position+type+role → canonical `output` |
| 2 | approximate nullability | OPTIONAL attr `str`; sibling convention needs check (gelu family) | ⚠️ Awaiting C-confirm — default: mirror sibling gelu (`str` non-nullable with default) |

- [ ] All ⚠️ conflicts confirmed by user

## Pitfall Alerts (from parse JSON `pitfall_alerts`)

> Empty — parse found no structural alerts.

| # | Alert id | Trigger | Must-read case | Gates steps | Applied? |
|---|----------|---------|----------------|-------------|----------|
| - | (none)   | -       | -              | -           | -        |

## Constraint Logic Checklist (Cross-Step Verification)

| # | Constraint Rule | .md Source | C++ (C-cpp) | Meta (C-meta) | Test (C-test) | Doc (C-docs) |
|---|-----------------|------------|--------------|---------------|---------------|--------------|
| 1 | input 最后一维偶数且 ≤1024 | L115 | ☐ | ☐ | ☐ | ☐ |
| 2 | input 其他维度乘积 ≤200000 | L115 | ☐ | ☐ | ☐ | ☐ |
| 3 | input dtype ∈ {BF16, F16, FLOAT} | L115 | ☐ | ☐ | ☐ | ☐ |
| 4 | input 维度 2-8 | L115 | ☐ | ☐ | ☐ | ☐ |
| 5 | approximate ∈ {"none", "tanh"} | L125 | ☐ | ☐ | ☐ | ☐ |
| 6 | output dtype = input dtype | L135 | ☐ | ☐ | ☐ | ☐ |
| 7 | output 最后一维 = input 最后一维 / 2 | L135 | ☐ | ☐ | ☐ | ☐ |
| 8 | output 其他维度 = input 其他维度 | L135 | ☐ | ☐ | ☐ | ☐ |
| 9 | 支持空Tensor | L115 | ☐ | ☐ | ☐ | ☐ |

## Constraint Coverage Self-Check (C-cpp+)

| # | Constraint Rule | C++ TORCH_CHECK Found? | Status |
|---|-----------------|----------------------|--------|
| 1 | input 最后一维偶数且 ≤1024 | TBD | ☐ |
| 2 | input 其他维度乘积 ≤200000 | TBD | ☐ |
| 3 | input dtype ∈ {BF16, F16, FLOAT} | TBD | ☐ |
| 4 | input 维度 2-8 | TBD | ☐ |
| 5 | approximate ∈ {"none", "tanh"} | TBD | ☐ |
| 6 | output dtype = input dtype | TBD | ☐ |
| 7 | output 最后一维 = input 最后一维 / 2 | TBD | ☐ |
| 8 | output 其他维度 = input 其他维度 | TBD | ☐ |
| 9 | 支持空Tensor | TBD | ☐ |

## Parameter Coverage Checklist

| # | Parameter Name | Scenario | Test Case Name | Status |
|---|----------------|----------|----------------|--------|
| 1 | approximate="none" (default) | default path | TBD | ☐ |
| 2 | approximate="tanh" | tanh path | TBD | ☐ |
| 3 | input BF16/F16/FLOAT | dtype coverage | TBD | ☐ |
| 4 | input 空Tensor | empty tensor | TBD | ☐ |
| 5 | input 2D-8D | rank coverage | TBD | ☐ |

## Derivatives Configuration Checklist (C-derivatives)

> Only applicable when autograd confirmed in C-confirm.

| # | Item | Details | Status |
|---|------|---------|--------|
| - | (pending C-confirm Q7) | - | ☐ |

## Compatibility Registration Checklist (C-baseline)

| # | Item | Details | Status |
|---|------|---------|--------|
| 1 | `op_api:` entry in schema JSON | TBD | ☐ |
| 2 | `func:` entry in schema JSON | TBD | ☐ |
| 3 | Version tags determined | TBD | ☐ |
| 4 | Entry added to allowlist | TBD | ☐ |
| 5 | Signature verified | TBD | ☐ |

## Progress

- [x] C-parse: Parse input files
- [ ] C-confirm: Confirm operator-specific information with user
  - [ ] **Standard Questions Tracker: ALL 9 rows ☑ (gate — block if any ☐)**
  - [ ] **Graph mode support confirmed (Q1: YES/NO, Q2: YES/NO)**
- [ ] C-yaml: Generate YAML
- [ ] C-derivatives: Configure derivatives.yaml (if autograd applies)
- [ ] C-cpp: Implement C++
- [ ] C-constraints: Constraint self-check
- [ ] C-meta: Add Meta registration [conditional: graph mode YES → execute, NO → skip]
- [ ] C-verify: Signature verification [+ Meta Shape/Dtype cross-verify if graph mode YES]
- [ ] C-codegen-check: codegen dry-run + signature cross-check
- [ ] C-build: Build [flag: graph mode YES → remove --disable_torchair, NO → keep]
- [ ] C-test: Tests (UT + FakeTensor) [+ graph mode test cases if graph mode YES]
- [ ] C-docs: API docs + update navigation files + update Python docstrings [+ graph mode call example if graph mode YES]
- [ ] C-baseline: Compatibility baseline (schema JSON + allowlist)
- [ ] C-delimit: SKIPPED (no ATK input — record in C-summary)
- [ ] C-summary: Integration summary
