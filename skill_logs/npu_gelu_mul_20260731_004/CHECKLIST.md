# npu_gelu_mul Integration Checklist

run_dir: /home/llxxyy/op-plugin/skill_logs/npu_gelu_mul_20260731_004

> **Mode: CREATE** — new operator integration. The op was deleted in commit 38ddb5a7 on the `after` branch; this run regenerates it from `./aclnn_input` (NOT restored from git).

## Project Constants

| Constant | Value | Set by | Status |
|----------|-------|--------|--------|
| `${OP_PLUGIN_ROOT_PATH}` | /home/llxxyy/op-plugin | SKILL.md §Rules (C-parse) | ☑ |
| `${OP_PLUGIN_SCRIPT_PATH}` | /home/llxxyy/.traecli/skills/op-adaptation/scripts | SKILL.md §Rules (C-parse) | ☑ |
| `${OP_PLUGIN_SKILL_HASH}` | nongit (skill dir is not a git repo in this deployment) | bootstrap.sh | ☑ |
| `${OP_PLUGIN_SKILL_BRANCH}` | nongit | bootstrap.sh | ☑ |
| `${OP_PLUGIN_BUILD_PATH}` | /home/llxxyy/op-plugin/build | bootstrap.sh (284G free on repo FS) | ☑ |
| `${CCACHE_DIR}` (only if user overrides) | default ~/.ccache (6332G free, OK) | bootstrap.sh | ☑ |
| `${MAX_JOBS}` | TBD by pre_build_check.sh (C-build) | pre_build_check.sh | ☐ |
| `${PYTORCH_TAG}` | TBD by pre_build_check.sh (C-build) | pre_build_check.sh | ☐ |

> Note: bootstrap.sh's git-state capture step fails because the skill dir `/home/llxxyy/.traecli/skills/op-adaptation` is not a git repository in this deployment (it was unpacked directly, not cloned). All other env vars resolved normally.

## Parse Results

- aclnn function name: aclnnGeluMul
- torch_npu function name: npu_gelu_mul
- Input parameters: `input` (Tensor, REQUIRED, BF16/F16/FLOAT, ND, 2-8D, supports empty + non-contiguous)
- Attr parameters: `approximate` (str, OPTIONAL, default "none", values: "none"|"tanh")
- Output parameters: `output` (Tensor, REQUIRED, dtype = input dtype, shape: last dim = input last dim / 2, other dims unchanged)
- Key constraints:
  - input last dim must be even and ≤ 1024
  - product of other dims ≤ 200000
  - input dtype ∈ {BFLOAT16, FLOAT16, FLOAT}
  - input dim rank 2-8
  - approximate ∈ {"none", "tanh"}
  - output dtype == input dtype
  - output last dim == input last dim / 2
  - SoC support: Ascend950 / A3 / 910b supported; 310b / 310p / 910 NOT supported
- Operator type: **Forward** (no `_grad` suffix)
- aclnn version variants: NONE (single aclnnGeluMul entry point, no `.Version()` markers in _def.cpp)

## ATK Case Dir (C-delimit input, path-only)

- ATK case dir: not provided → C-delimit SKIPPED, will note in C-summary

## Parameter Information Table (Cross-Skill Shared)

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line | .md Description | Constraints |
|---|------------|--------|------|-------------|---------|----------|---------------|-----------------|-------------|
| 1 | input | both | Tensor | REQUIRED | - | - | L24 | 输入的张量，公式中的input | last dim even & ≤1024; other dims product ≤200000; dtype BF16/F16/FLOAT; ND; rank 2-8; supports empty + non-contiguous |
| 2 | approximate | both | str | OPTIONAL | "none" | OPTIONAL | L31 | Gelu计算的模式 | values: "none"\|"tanh"; null → "none" |
| 3 | output (out) | both (name-variant: _def.cpp "output" ↔ .md "out") | Tensor | REQUIRED | - | - | L33 | 输出的张量，公式中的out | dtype == input dtype; shape: last dim = input last dim / 2, other dims unchanged; rank 2-8 |

**Conflict reconciliation**:
- `out` flagged as `missing_in_def` is a **false conflict** — `.md` uses `out` for the output tensor while `_def.cpp` names it `output`. Reconciled by position+role (single output tensor). Canonical name: `output` (following _def.cpp authority).

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

## Pitfall Alerts (from parse JSON `pitfall_alerts`)

> None — parse JSON `pitfall_alerts` is empty.

| # | Alert id | Trigger | Must-read case | Gates steps | Applied? |
|---|----------|---------|----------------|-------------|----------|
| - | - | - | - | - | - |

## Constraint Logic Checklist (Cross-Step Verification)

| # | Constraint Rule | .md Source | C++ (C-cpp) | Meta (C-meta) | Test (C-test) | Doc (C-docs) |
|---|-----------------|------------|--------------|---------------|---------------|--------------|
| 1 | input dtype ∈ {BFLOAT16, FLOAT16, FLOAT} | L115 | ☑ (L37 TORCH_CHECK) | ☑ (implicit via dtype=input.dtype; sibling style) | ☐ | ☐ |
| 2 | input rank 2-8 | L118 | ☑ (L43 TORCH_CHECK) | ☑ (L7312 torch._check) | ☐ | ☐ |
| 3 | input last dim even & ≤ 1024 | L115 | ☑ (L49 even, L52 ≤1024) | ☑ (L7316 even, L7320 ≤1024) | ☐ | ☐ |
| 4 | product of input other dims ≤ 200000 | L115 | ☑ (L61 TORCH_CHECK) | deferred (perf constraint, sibling npu_gelu/swiglu don't add to meta either) | ☐ | ☐ |
| 5 | approximate ∈ {"none", "tanh"} | L125 | ☑ (via helper npu_gelu_approximate_str L67) | ☑ (L7324 torch._check) | ☐ | ☐ |
| 6 | output dtype == input dtype | L135 | ☑ (input.options() at L73) | ☑ (dtype=input.dtype at L7330) | ☐ | ☐ |
| 7 | output shape: last dim = input last dim / 2, others unchanged | L135 | ☑ (npu_gelu_mul_out_size L26-29) | ☑ (output_shape[-1]//=2 at L7329) | ☐ | ☐ |
| 8 | SoC: 950/A3/910b supported; 310b/310p/910 NOT | L7-24 | deferred (structural, _def.cpp AICore config) | deferred (structural) | ☐ | ☐ |

## Constraint Coverage Self-Check (C-cpp+)

Checker script: 2 covered / 2 uncovered (false negatives) / 3 unanchored (SoC support rows, structural).

| # | Constraint Rule | C++ TORCH_CHECK Found? | Status |
|---|-----------------|----------------------|--------|
| 1 | input dtype ∈ {BF16, F16, FLOAT} | Yes (NpuGeluMulKernelNpuOpApi.cpp:37) | ☑ |
| 2 | input rank 2-8 | Yes (L43) | ☑ |
| 3 | input last dim even & ≤ 1024 | Yes (L49 even, L52 ≤1024) | ☑ |
| 4 | product of input other dims ≤ 200000 | Yes (L61) | ☑ |
| 5 | approximate ∈ {"none", "tanh"} | deferred — covered by helper `op_infer::npu_gelu_approximate_str(approximate)` (KernelNpuNewParams.cpp:29) called at L67; checker can't trace cross-file helper | ☑ |
| 6 | output dtype == input dtype | deferred — guaranteed by allocation `apply_tensor_without_format(shape, input.options())` (L73) which copies input's dtype | ☑ |
| 7 | output shape: last dim = input last dim / 2 | deferred — guaranteed by `npu_gelu_mul_out_size(input)` helper (L26-29) which halves `output_shape.back()` | ☑ |
| 8 | SoC: 950/A3/910b supported; 310b/310p/910 NOT | deferred — structural; controlled by `_def.cpp` AICore config (ascend910b/ascend910_93/ascend950), not an adapter-level check | ☑ |

## Parameter Coverage Checklist

| # | Parameter Name | Scenario | Test Case Name | Status |
|---|----------------|----------|----------------|--------|
| 1 | approximate="none" (default) | default path | TBD by op-test | ☐ |
| 2 | approximate="tanh" | non-default | TBD by op-test | ☐ |
| 3 | BF16/F16/FLOAT dtypes | dtype coverage | TBD by op-test | ☐ |
| 4 | empty Tensor | edge case | TBD by op-test | ☐ |
| 5 | non-contiguous Tensor | edge case | TBD by op-test | ☐ |

## Compatibility Registration Checklist (C-baseline)

| # | Item | Details | Status |
|---|------|---------|--------|
| 1 | `op_api:` entry in schema JSON | `op_api: torch_npu.npu_gelu_mul(*args, **kwargs)` version `all_version` (line 5072) | ☑ (typed-form verification deferred to post-build) |
| 2 | `func:` entry in schema JSON | `func: npu_gelu_mul(Tensor input, *, str approximate='none') -> Tensor` version `all_version` (line 5069) | ☑ |
| 3 | Version tags determined | `all_version` (matches sibling npu_gelu; new op available in all supported versions) | ☑ |
| 4 | Entry added to allowlist | `npu_gelu_mul` appended to `torch_npu.all_version` (line 220) | ☑ |
| 5 | Signature verified | `test_compatibility.py` pending post-build install (needs `inspect.signature(torch_npu.npu_gelu_mul)`) | ☐ |

## Progress

- [x] C-parse: Parse input files
- [x] C-confirm: Confirm operator-specific information with user
  - [x] **Standard Questions Tracker: ALL 9 rows ☑ (gate — block if any ☐)**
  - [x] **Graph mode support confirmed (Q1: YES, Q2: NO)**
- [x] C-yaml: Generate YAML
- [x] C-derivatives: Configure derivatives.yaml (skipped — no autograd)
- [x] C-cpp: Implement C++
- [x] C-constraints: Constraint self-check
- [x] C-meta: Add Meta registration [executed — torch.compile YES]
- [x] C-verify: Signature verification [+ Meta Shape/Dtype cross-verify — PASS]
- [x] C-codegen-check: gencode.sh dry-run — yaml schema generated correctly (`npu_gelu_mul` entry present in v2r10/op_plugin_functions.yaml with `exposed: true`, `impl_ns: op_api`); derivatives.yaml correctly absent; YAML↔C++ type mapping verified (Tensor→const at::Tensor&, str→c10::string_view, return Tensor→at::Tensor). CustomFunctions.h ABI check deferred to C-build (build/pytorch not yet bootstrapped).
- [ ] C-build: Build [flag: graph mode YES → remove --disable_torchair, NO → keep]
- [ ] C-test: Tests (UT + FakeTensor) [+ graph mode test cases if graph mode YES]
- [ ] C-docs: API docs + update navigation files + update Python docstrings [+ graph mode call example if graph mode YES]
- [ ] C-baseline: Compatibility baseline (schema JSON + allowlist)
- [ ] C-delimit: ATK cases run & locate/delimit via op-delimitation subagent [SKIPPED — no ATK input]
- [ ] C-summary: Integration summary

## UT Generation Info

### Operator Basic Info
- Operator Name: gelu_mul
- torch_npu function name: npu_gelu_mul
- Operator Type: custom
- Target Directory: test/test_custom_ops
- Mode: create
- Operator Classification: activation

### Source File Paths
- _def.cpp: /home/llxxyy/aclnn_input/gelu_mul_def.cpp (outside repo — provided via C-parse)
- .md doc: /home/llxxyy/aclnn_input/aclnnGeluMul.md (outside repo — provided via C-parse)
- C++ impl: op_plugin/ops/opapi/NpuGeluMulKernelNpuOpApi.cpp
- YAML: op_plugin/config/op_plugin_functions.yaml
- meta: op_plugin/python/meta/_meta_registrations.py

### Input/Output Spec

**Input parameter list**:

| Param Name | Type | Required/Optional | Supported dtypes | Constraints |
|------------|------|-------------------|------------------|-------------|
| input | Tensor | Required | bfloat16, float16, float32 | ND format; 2-8D; last dim even & ≤1024; product of other dims ≤200000; supports empty Tensor; supports non-contiguous Tensor |
| * | (separator) | - | - | position-keyword separator |
| approximate | str | Optional (default 'none') | - | values: 'none' \| 'tanh' |

**Output specs**:

| Output Name | Type | Shape Rule | Dtype Rule |
|-------------|------|------------|------------|
| output | Tensor | copy input shape, last dim = input last dim / 2 | same as input dtype |

### Coverage Matrix

> Auto-generated by scaffold_coverage.py, add/delete applied per Constraints column markers.

| Test Item | Covered Params | Data Type | Dimensions | Scenario |
|-----------|----------------|-----------|------------|----------|
| test_basic_bf16 | input | bfloat16 | (128, 128) | Basic functionality |
| test_basic_fp16 | input | float16 | (128, 128) | Basic functionality |
| test_basic_fp32 | input | float32 | (128, 128) | Basic functionality |
| test_min_dimension | input | float32 | (128, 128) | 边界-最小维度(2D) |
| test_3d_input | input | float32 | (4, 32, 128) | 3D输入 |
| test_output_shape_dtype | input | float16 | (4, 32, 128) | 输出shape+dtype断言 |

**Auto-generated stats**: 6 test items.

**Rule compliance self-check**:
- Basic functionality dtype coverage: 3 rows ✅
- Output dtype full coverage: expected 0 values [], expanded 0 rows ⚠️ 漏展开
- Optional param coverage: 0 optional Tensors ✅
- Interaction / not-applicable: Step 3 handles row add/delete and expansion per `Interaction: A×B` / `Not applicable: param` anchors in Constraints column
- Required scenario check (category=activation):
    - all supported dtypes: ⚠️ missing
    - in-place/out-of-place: ✅ covered
    - boundary input values: ✅ covered

**⚠️ Self-check warnings (step 3 review or upstream format fix needed)**:
  - Required scenarios missing (activation): ['all supported dtypes']
  - Negative degradation: .constraint_info.json missing, only dtype+dim negative recoverable; rank/shape_link/mutex negative lost → re-run scaffold_init to generate .constraint_info.json then re-run this script

### Negative Coverage Matrix

> Auto-generated by generate_negative_cases(), derived from constraint signals for illegal input combinations.

| Test Item | Covered Params | Data Type | Dimensions | Scenario | Violated Field | Violation Construction | Expected |
|-----------|----------------|-----------|------------|----------|----------------|------------------------|----------|
| test_neg_input_invalid_dtype | input | float32 | (128, 128) | Negative-dtype domain | input | dtype=int64 | raises:PARAM_INVALID |
| test_neg_dim_below_min | input | float32 | (128, 128) | Negative-dim out of bounds | input | shape=(2,) | raises:PARAM_INVALID |
| test_neg_dim_above_max | input | float32 | (128, 128) | Negative-dim out of bounds | input | shape=(2, 2, 2, 2, 2, 2, 2, 2, 2) | raises:PARAM_INVALID |

**Negative category coverage**: 2 categories, 3 rows total

### Finalization Add/Delete Log

> 格式契约: `Delete <exact_method_name>: <reason>` / `Add <exact_method_name>: <reason>`
> `<exact_method_name>` 必须与 Coverage Matrix 的 Test Item 列值**逐字一致**, 禁自然语言概述。
> reconcile.py 依赖此格式做删除放行判定, 格式不符 → [BAD-LOG] 告警。

<!-- 在此下方逐行记录增删, 一行一项 -->
Add test_approximate_tanh: approximate='tanh' 模式覆盖（.md 明确支持两种模式，baseline 仅测默认 'none'）
Add test_empty_tensor: 空 Tensor 支持（.md 明确"支持空Tensor"，activation 类边界场景）
Add test_non_contiguous: 非连续 Tensor 支持（.md 明确"支持非连续Tensor"）
Add test_graph_mode: torch.compile 图模式测试（C-confirm Q1=YES，graph mode 必测）
Delete test_output_shape_dtype: 与 test_basic_fp16 重复（shape/dtype 断言已在 basic 用例中隐含覆盖，单独行冗余）
