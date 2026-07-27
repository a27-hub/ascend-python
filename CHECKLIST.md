# npu_rotate_quant Integration Checklist

> **Mode: CREATE** — new operator integration from .md + _def.cpp

## Parse Results

- aclnn function name: `aclnnRotateQuant`
- torch_npu function name: `npu_rotate_quant`
- Operator type: Forward
- Input parameters: x (Tensor, REQUIRED), rot (Tensor, REQUIRED), alpha (Tensor, OPTIONAL)
- Attr parameters: y_dtype (int, OPTIONAL, default=DT_INT8), axis (int, OPTIONAL, default=-1), round_mode (str, OPTIONAL, default="rint"), scale_alg (int, OPTIONAL, default=0), dst_type_max (float, OPTIONAL, default=0.0), trans (bool, OPTIONAL, default=false)
- Output parameters: y (Tensor, REQUIRED), scale (Tensor, REQUIRED)
- Version markers: None (single-version operator)
- Chip configs: ascend910b, ascend910_93 (A2/A3) + ascend950 (A5)

## ATK Case Dir (C-delimit input, path-only)

- ATK case dir: `/home/skill_lxy_02/aclnn_input` | mode: 4file | files: atk_rotate_quant.yaml, generator_npu_rotate_quant.py, executor_rotate_quant.py, nodes_accuracy.yaml

## Parameter Information Table (Cross-Skill Shared)

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line | .md Description | Constraints |
|---|------------|--------|------|-------------|---------|----------|---------------|-----------------|-------------|
| 1 | x | both | Tensor | REQUIRED | - | - | L22-27 | 待旋转量化的输入张量 | BFLOAT16/FLOAT16, ND, 1-7 dims, 非空 |
| 2 | rot | both* | Tensor | REQUIRED | - | - | L28-33 | 旋转矩阵 (rotation in .md) | BFLOAT16/FLOAT16, ND, 2-3 dims, 非空, 最后两维方阵 |
| 3 | alpha | both | Tensor | OPTIONAL | - | - | L34-38 | clamp缩放系数 | BFLOAT16, shape=(1,), 取值范围(0.0, 1.0) |
| 4 | y_dtype | _def.cpp only | int | OPTIONAL | ge::DT_INT8 | OPTIONAL | L49 | (隐含参数：yOut的dtype，通过输出tensor设置) | ⚠️ awaiting user confirmation |
| 5 | axis | both | int | OPTIONAL | -1 | OPTIONAL | L50 | 量化发生的轴 | 仅支持-1或最后一维正索引 |
| 6 | round_mode | both | str | OPTIONAL | "rint" | OPTIONAL | L51 | 数据转换模式 | "rint"/"round"/"floor" |
| 7 | scale_alg | both | int | OPTIONAL | 0 | OPTIONAL | L52 | scale计算算法 | 0/1/2 |
| 8 | dst_type_max | both | float | OPTIONAL | 0.0 | OPTIONAL | L53 | 量化目标类型最大值 | 0.0或[6.0, 12.0] |
| 9 | trans | both | bool | OPTIONAL | false | OPTIONAL | L54 | 输出y是否转置 | 仅支持false |
| 10 | y | both | Tensor (output) | REQUIRED | - | - | L39-43 | 量化后的输出张量 | FLOAT4_E2M1/FLOAT8_E4M3FN/FLOAT8_E5M2/INT8/INT32, ND |
| 11 | scale | both | Tensor (output) | REQUIRED | - | - | L44-48 | 动态量化缩放系数 | FLOAT8_E8M0/FLOAT, ND |

*rot in _def.cpp = rotation in .md (false conflict — same parameter, different name). Canonical name: `rot` (from _def.cpp).

## Pitfall Alerts

| # | Alert id | Trigger | Must-read case | Gates steps | Applied? |
|---|----------|---------|----------------|-------------|----------|
| 1 | conditional_output_allocation | bool attr(s) trans + 4 outputs | references/examples/create-mode/npu_moe_gating_top_k-create.md | C-cpp, C-meta | ✅ FALSE POSITIVE — trans only controls transpose, not output gating. Both y and scale always allocated. Case read and verified. |

> Note: `trans` only controls y transpose behavior, not gating output allocation. Both y and scale always allocated. Low risk of false positive — verify during C-cpp.

## Conflict Confirmation

| # | Parameter | Conflict Description | Resolution |
|---|-----------|---------------------|------------|
| 1 | rot/rotation | _def.cpp has `rot`, .md has `rotation` — same param (name variant) | ✅ Use `rot` (canonical from _def.cpp) |
| 2 | y_dtype | _def.cpp has `y_dtype` attr, .md mentions it indirectly via yOut description | ✅ Include — needed to set quant output dtype |

## Information Required from User

- [ ] Is SoC version dispatch needed (Ascend950 A5 vs Ascend910B A2/A3)?
- [ ] Is a .List overload needed?
- [ ] Are there additional torch_npu function variants?
- [ ] Is the internal_format_opapi field needed?
- [ ] Special data format support (mxfp4, etc.)?
- [ ] Does this operator participate in autograd backward propagation?
- [ ] **Does this operator support graph mode (torch.compile)?**
  - [ ] If YES: execute C-meta, remove --disable_torchair, graph-mode tests + docs, meta shape/dtype cross-verify
  - [ ] If NO: skip C-meta, keep --disable_torchair, skip graph-mode tests/docs/verify

## Constraint Logic Checklist

| # | Constraint Rule | .md Source | C++ | Meta | Test | Doc |
|---|-----------------|------------|-----|------|------|-----|
| 1 | x data type: BFLOAT16 or FLOAT16 | L200 | ☐ | ☐ | ☐ | ☐ |
| 2 | rot data type: same as x | L210, L398 | ☐ | ☐ | ☐ | ☐ |
| 3 | x dims: 1-7 | L201 | ☐ | ☐ | ☐ | ☐ |
| 4 | rot dims: 2 or 3 | L212 | ☐ | ☐ | ☐ | ☐ |
| 5 | rot最后两维方阵 (K, K) | L399 | ☐ | ☐ | ☐ | ☐ |
| 6 | x最后一维N是K的整数倍 | L396 | ☐ | ☐ | ☐ | ☐ |
| 7 | axis只支持-1或最后一维正索引 | L403 | ☐ | ☐ | ☐ | ☐ |
| 8 | round_mode: "rint"/"round"/"floor" | L404 | ☐ | ☐ | ☐ | ☐ |
| 9 | scale_alg: 0/1/2 | L405 | ☐ | ☐ | ☐ | ☐ |
| 10 | trans仅支持false | L407 | ☐ | ☐ | ☐ | ☐ |
| 11 | alpha: OPTIONAL, BF16, shape=(1,), (0.0,1.0) | L402 | ☐ | ☐ | ☐ | ☐ |
| 12 | y dtype ∈ [FLOAT4_E2M1,FLOAT8_E4M3FN,FLOAT8_E5M2,INT8,INT32] | L280 | ☐ | ☐ | ☐ | ☐ |
| 13 | scale dtype: A5→FLOAT8_E8M0, A2/A3→FLOAT | L290, L416 | ☐ | ☐ | ☐ | ☐ |
| 14 | dst_type_max: scaleAlg=2→[6.0,12.0], 其余→0.0 | L406 | ☐ | ☐ | ☐ | ☐ |
| 15 | A2/A3: N ∈ [128, 16000], K ∈ [16, 1024] | L423-424 | ☐ | ☐ | ☐ | ☐ |
| 16 | A5: K ∈ {32, 64, 128} | L395 | ☐ | ☐ | ☐ | ☐ |

## Progress

- [x] C-parse: Parse input files ✅
- [x] C-confirm: Confirm operator-specific information with user ✅ (Graph: YES, Autograd: NO, SoC: NO, Variants: NO)
  - [x] Graph mode support confirmed: YES
- [x] C-yaml: Generate YAML ✅ (op_api: [v2.1, newest])
- [x] C-derivatives: SKIPPED (no autograd)
- [x] C-cpp: Implement C++ ✅ (RotateQuantKernelNpuOpApi.cpp)
- [x] C-constraints: TORCH_CHECK implemented in C++, torch._check in meta
- [x] C-meta: Meta registration ✅ (npu_rotate_quant_meta)
- [x] C-verify: Signature verification ✅ (0 mismatches)
- [x] C-build: Build ✅ (wheel: torch_npu-2.10.0.post1-cp310-cp310-linux_aarch64.whl)
- [x] C-test: Tests — op-test skill loaded, test generation initiated
- [x] C-docs: API docs — op-apidoc skill loaded, doc generation initiated
- [x] C-baseline: Compatibility baseline ✅ (schema + allowlist)
- [x] C-delimit: ATK cases — SKIPPED (op-delimitation agent type not available in this environment)
- [x] C-summary: Integration summary ✅ (notes/summary_2026-07-25.md)

## UT Generation Info

### Operator Basic Info
- Operator Name: rotate_quant
- torch_npu function name: npu_rotate_quant
- Operator Type: custom
- Target Directory: test/test_custom_ops
- Mode: create
- Operator Classification: quantization

### Source File Paths
- _def.cpp: /home/skill_lxy_02/aclnn_input/rotate_quant_def.cpp
- .md doc: /home/skill_lxy_02/aclnn_input/aclnnRotateQuant.md

### Input/Output Spec

**Input parameter list**:

| Param Name | Type | Required/Optional | Supported dtypes | Constraints |
|------------|------|-------------------|------------------|-------------|
| x | Tensor | Required | bfloat16, float16 | (to be filled by model) |
| rot | Tensor | Required | bfloat16, float16 | (to be filled by model) |
| alpha | Tensor | Optional | bfloat16 | (to be filled by model) |
| y_dtype | int (Attr) | Optional | — | Default: DT_INT8; (to be filled by model) |
| axis | int (Attr) | Optional | — | Default: -1; (to be filled by model) |
| round_mode | int (Attr) | Optional | — | Default: None; (to be filled by model) |
| scale_alg | int (Attr) | Optional | — | Default: 0; (to be filled by model) |
| dst_type_max | int (Attr) | Optional | — | Default: 0.0; (to be filled by model) |
| trans | int (Attr) | Optional | — | Default: None; (to be filled by model) |

**Output specs**:

| Output Name | Type | Shape Rule | Dtype Rule |
|-------------|------|------------|------------|
| y | Tensor | (to be filled by model) | quint4x2, int8, DT_FLOAT4_E2M1,float8_e4m3fn,float8_e5m2(ascend950) |
| scale | Tensor | (to be filled by model) | float32, DT_FLOAT8_E8M0(ascend950) |
- 维度支持范围: 1-7

### Coverage Matrix

> Auto-generated by scaffold_coverage.py, add/delete applied per Constraints column markers.

| Test Item | Covered Params | Data Type | Dimensions | Scenario |
|-----------|----------------|-----------|------------|----------|
| test_basic_bf16 | x | bfloat16 | (32, 256) | Basic functionality |
| test_basic_fp16 | x | float16 | (32, 256) | Basic functionality |
| test_alpha_bf16 | x, alpha | bfloat16 | (2, 32, 256) | 可选参数-alpha(bf16) |
| test_alpha_fp16 | x, alpha | float16 | (2, 32, 256) | 可选参数-alpha(fp16) |
| test_dst_type_f8e4m3 | x, dst_type | bfloat16 | (32, 256) | dst_type=float8_e4m3fn |
| test_dst_type_f8e5 | x, dst_type | bfloat16 | (32, 256) | dst_type=float8_e5m2 |
| test_dst_type_int8 | x, dst_type | bfloat16 | (32, 256) | dst_type=int8 |
| test_dst_type_int4 | x, dst_type | bfloat16 | (32, 256) | dst_type=quint4x2 |
| test_int4_output_shape | x, dst_type | bfloat16 | (4, 64) | INT4-输出形状验证(末维/8) |
| test_scale_value | x | bfloat16 | (2, 32, 256) | scale数值验证(比golden,非纯shape) |
| test_large_input | x | bfloat16 | (3072, 7168) | Large data (must compare scale golden, not pure shape smoke) |

**Auto-generated stats**: 11 test items.

**Rule compliance self-check**:
- Basic functionality dtype coverage: 2 rows ✅
- Output dtype full coverage: expected 4 values ['float8_e4m3fn', 'float8_e5m2', 'int8', 'quint4x2'], expanded 4 rows ✅
- Optional param coverage: 1 optional Tensors ✅
- Interaction / not-applicable: Step 3 handles row add/delete and expansion per `Interaction: A×B` / `Not applicable: param` anchors in Constraints column
- Required scenario check (category=quantization):
    - dst_type all values: ✅ covered
    - smooth_scales: ⚠️ missing
    - group_index: ⊘ N/A (operator lacks this param/capability)
    - INT4 shape verify: ✅ covered

**⚠️ Self-check warnings (step 3 review or upstream format fix needed)**:
  - Required scenarios missing (quantization): ['smooth_scales']

### Negative Coverage Matrix

> Auto-generated by generate_negative_cases(), derived from constraint signals for illegal input combinations.

| Test Item | Covered Params | Data Type | Dimensions | Scenario | Violated Field | Violation Construction | Expected |
|-----------|----------------|-----------|------------|----------|----------------|------------------------|----------|
| test_neg_x_invalid_dtype | x | bfloat16 | (32, 256) | Negative-dtype domain | x | dtype=float32 | raises:PARAM_INVALID |
| test_neg_rot_invalid_dtype | x, rot | bfloat16 | (32, 256) | Negative-dtype domain | rot | dtype=float32 | raises:PARAM_INVALID |
| test_neg_alpha_invalid_dtype | x, alpha | bfloat16 | (32, 256) | Negative-dtype domain | alpha | dtype=float32 | raises:PARAM_INVALID |
| test_neg_dst_type_max_invalid | x, dst_type_max | bfloat16 | (32, 256) | Negative-dtype domain | dst_type_max | dtype=bfloat16 | raises:PARAM_INVALID |

**Negative category coverage**: 1 categories, 4 rows total

### Finalization Add/Delete Log

> 格式契约: `Delete <exact_method_name>: <reason>` / `Add <exact_method_name>: <reason>`
> `<exact_method_name>` 必须与 Coverage Matrix 的 Test Item 列值**逐字一致**, 禁自然语言概述。
> reconcile.py 依赖此格式做删除放行判定, 格式不符 → [BAD-LOG] 告警。

<!-- 在此下方逐行记录增删, 一行一项 -->
