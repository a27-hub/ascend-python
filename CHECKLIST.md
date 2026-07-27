# npu_gelu_mul Integration Checklist

> **Mode: CREATE** — new operator integration (operator code deleted, regenerating from input files; no git restore).

## Parse Results

- aclnn function name: aclnnGeluMul
- torch_npu function name: npu_gelu_mul
- Input parameters: `input` (Tensor, REQUIRED)
- Output parameters: `output` (Tensor, REQUIRED) — shape = input shape with last dim halved
- Attr: `approximate` (str, OPTIONAL, default `"none"`) — `"none"`(erf) / `"tanh"`
- Key constraints:
  - dtypes: BFLOAT16, FLOAT16, FLOAT (output dtype == input dtype)
  - rank: 2–8 dims (ND format)
  - last dim must be **even** and **<= 1024**
  - product of other dims <= 200000
  - supports empty Tensor
  - supported SoC: Ascend950PR/950DT (A5), Atlas A3, Atlas A2 (910b); NOT 310b/310p/910
  - `_def.cpp` AICore configs: ascend910b, ascend910_93, ascend950 (+ kirin)
- Operator type: Forward
- Formula: split input last dim into x1 (left half), x2 (right half); out = GELU(x1) * x2

## ATK Case Dir (C-delimit input, path-only)

- ATK case dir: `/home/skill_lxy/aclnn_input` | mode: 4file-like | files: atk_gelu_mul.yaml, generator_npu_gelu_mul.py, executor_gelu_mul.py, nodes_accuracy.yaml
  （gelu_mul 相关 ATK 输入存在；C-delimit 由 subagent 消费这些路径）

## Parameter Information Table (Cross-Skill Shared)

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line | .md Description | Constraints |
|---|------------|--------|------|-------------|---------|----------|---------------|-----------------|-------------|
| 1 | input | both | Tensor | REQUIRED | - | - | L24 | 输入张量 | BF16/F16/FLOAT, ND, 2-8D, last dim even & <=1024, other dims product <=200000 |
| 2 | approximate | both | str | OPTIONAL | "none" | OPTIONAL | L31 | Gelu 模式 | only "none"/"tanh"; nullptr→"none" |
| 3 | out | .md only | aclTensor* | (output) | - | - | (md L72) | 输出张量 | output last dim = input last dim / 2; dtype == input dtype |

**Conflict `out`**: `.md only` aclnn output param → standard exclude (it is the aclnn-side output tensor, materialized in C++ as `output`). Not a real conflict.

## Information Required from User

- [ ] aclnn version variants? — none in _def.cpp (no `.Version()`); single aclnnGeluMul entry. Assume NO.
- [ ] SoC version dispatch needed? — opapi registers normally; no per-SoC branch in C++.
- [ ] .List overload? — NO (input is single Tensor).
- [ ] additional variants (_asymmetric/_v0)? — NO.
- [ ] internal_format_opapi? — NO.
- [ ] special data format (mxfp4...)? — NO.
- [ ] autograd backward? — **ask user** (no _grad def provided).
- [ ] graph mode (torch.compile)? — **ask user**.

### Conflict Confirmation

| # | Parameter | Conflict | Resolution |
|---|-----------|----------|------------|
| 1 | out | `.md only` (aclnn output tensor) | Exclude (materialized as `output` in C++); not a real conflict |

## Pitfall Alerts

None from parse JSON (`pitfall_alerts` empty).

## Constraint Logic Checklist

| # | Constraint Rule | .md Source | C++ | Meta | Test | Doc |
|---|-----------------|------------|-----|------|------|-----|
| 1 | dtype ∈ {BF16,F16,FLOAT} | L116 | ☐ | ☐ | ☐ | ☐ |
| 2 | rank 2–8 | L118 | ☐ | ☐ | ☐ | ☐ |
| 3 | last dim even & <=1024 | L115 | ☐ | ☐ | ☐ | ☐ |
| 4 | other-dims product <=200000 | L115 | ☐ | ☐ | ☐ | ☐ |
| 5 | approximate ∈ {none, tanh} | L125 | ☐ | ☐ | ☐ | ☐ |
| 6 | output last dim = input last dim / 2 | L135 | ☐ | ☐ | ☐ | ☐ |
| 7 | output dtype == input dtype | L135 | ☐ | ☐ | ☐ | ☐ |

## Progress

- [x] C-parse
- [x] C-confirm (graph mode=YES, autograd=NO)
- [x] C-yaml (func entry, op_api+exposed [v2.1,newest])
- [x] C-derivatives (skipped — autograd NO)
- [x] C-cpp (GeluMulKernelNpuOpApi.cpp, uses npu_gelu_mul_output_size)
- [x] C-constraints (all 7 covered by TORCH_CHECK)
- [x] C-meta (npu_gelu_mul_meta, last dim halved, mirrors C++)
- [x] C-verify (0 mismatches; meta shape/dtype cross-verified)
- [x] C-baseline (schema x2 all_version + allowlist all_version)
- [x] C-build (rc=0, whl built; fixed ErrCode::DIM→VALUE; torchair enabled)
- [x] C-test (NPU UT 13/13 pass; FakeTensor 2/2 pass)
- [x] C-docs (page + nav + docstring done)
- [x] C-delimit (ATK 500 cases all SUCCESS, 0 accuracy fail; 475 valid match + 25 inf/nan invalid)
- [x] C-summary (notes/summary_2026-07-25.md)

## UT Generation Info

### Operator Basic Info
- Operator Name: gelu_mul
- torch_npu function name: npu_gelu_mul
- Operator Type: custom
- Target Directory: test/test_custom_ops
- Mode: create
- Operator Classification: activation

### Source File Paths
- _def.cpp: /home/skill_lxy/aclnn_input/gelu_mul_def.cpp
- .md doc: /home/skill_lxy/aclnn_input/aclnnGeluMul.md

### Input/Output Spec

**Input parameter list**:

| Param Name | Type | Required/Optional | Supported dtypes | Constraints |
|------------|------|-------------------|------------------|-------------|
| input | Tensor | Required | bfloat16, float32, float16 | rank 2-8; last dim even and <= 1024; product of other dims <= 200000; supports empty tensor |
| approximate | int (Attr) | Optional | — | str attr in {"none","tanh"}; default "none"; "none"=erf mode, "tanh"=tanh mode |

**Output specs**:

| Output Name | Type | Shape Rule | Dtype Rule |
|-------------|------|------------|------------|
| output | Tensor | same shape as input except last dim = input last dim / 2 | bfloat16, float32, float16 |
- 维度支持范围: 2-8

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
| test_approximate_tanh | input | float32 | (128, 128) | approximate='tanh' 模式 (Interaction: approximate×output, none→erf; tanh→tanh) |
| test_input_8d | input | float32 | (2, 2, 2, 2, 2, 2, 2, 16) | 边界-最大维度(8D) |

**Auto-generated stats**: 8 test items. (+2 added, -2 deleted = 8 final)

**Rule compliance self-check**:
- Basic functionality dtype coverage: 3 rows ✅
- Output dtype full coverage: expected 2 values ['bfloat16', 'float16'], expanded 2 rows ✅
- Optional param coverage: 0 optional Tensors ✅
- Interaction / not-applicable: Step 3 handles row add/delete and expansion per `Interaction: A×B` / `Not applicable: param` anchors in Constraints column
- Required scenario check (category=activation):
    - all supported dtypes: ⚠️ missing
    - in-place/out-of-place: ✅ covered
    - boundary input values: ✅ covered

**⚠️ Self-check warnings (step 3 review or upstream format fix needed)**:
  - Required scenarios missing (activation): ['all supported dtypes']

### Negative Coverage Matrix

> Auto-generated by generate_negative_cases(), derived from constraint signals for illegal input combinations.

| Test Item | Covered Params | Data Type | Dimensions | Scenario | Violated Field | Violation Construction | Expected |
|-----------|----------------|-----------|------------|----------|----------------|------------------------|----------|
| test_neg_input_invalid_dtype | input | float32 | (128, 128) | Negative-dtype domain | input | dtype=int64 | raises:PARAM_INVALID |
| test_neg_dim_below_min | input | float32 | (128, 128) | Negative-dim out of bounds | input | shape=(2,) | raises:PARAM_INVALID |
| test_neg_dim_above_max | input | float32 | (128, 128) | Negative-dim out of bounds | input | shape=(2, 2, 2, 2, 2, 2, 2, 2, 2) | raises:PARAM_INVALID |
| test_neg_last_dim_odd | input | float32 | (128, 127) | Negative-shape | input | last dim odd (127) | raises:PARAM_INVALID |
| test_neg_approximate_invalid | input | float32 | (128, 128) | Negative-value | approximate | approximate="relu" | raises:PARAM_INVALID |

**Negative category coverage**: 3 categories, 5 rows total

### Finalization Add/Delete Log

- Add test_approximate_tanh: covers the approximate='tanh' branch (Interaction: approximate×output) — the default 'none' path is covered by all basic cases, tanh was untested.
- Add test_input_8d: boundary max rank (8D) — rank 2-8 is a key constraint, only 2D/3D were covered.
- Add test_neg_last_dim_odd: last dim even is a hard constraint; previously only rank-bounds negatives existed.
- Add test_neg_approximate_invalid: approximate ∈ {none,tanh} is a hard constraint; previously untested.
- Resolve 'all supported dtypes' warning: false alarm — test_basic_bf16/fp16/fp32 already cover all 3 supported dtypes.
- Delete test_output_dtype_bf16: not-applicable — output dtype always equals input dtype (no independent output dtype control); bf16 output is already covered transitively by test_basic_bf16.
- Delete test_output_dtype_fp16: not-applicable — same reason; fp16 output covered by test_basic_fp16.
> 格式契约: `Delete <exact_method_name>: <reason>` / `Add <exact_method_name>: <reason>`
> `<exact_method_name>` 必须与 Coverage Matrix 的 Test Item 列值**逐字一致**, 禁自然语言概述。
> reconcile.py 依赖此格式做删除放行判定, 格式不符 → [BAD-LOG] 告警。

<!-- 在此下方逐行记录增删, 一行一项 -->
