# npu_swi_glu_quant_v2 Integration Checklist

> **Mode: CREATE** — re-integrating `aclnnSwiGluQuantV2` (source operator was deleted; regenerated from parse inputs `.md` + `_def.cpp`, NOT git-restored). Function name `npu_swi_glu_quant_v2` confirmed via build-tree test artifact + standard convention `aclnn{SwiGluQuantV2} → npu_{swi_glu_quant_v2}`.

## Parse Results

- aclnn function name: `aclnnSwiGluQuantV2` (+ `aclnnSwiGluQuantV2GetWorkspaceSize`)
- torch_npu function name: `npu_swi_glu_quant_v2`
- Input parameters: `x` (REQUIRED Tensor, FP16/BF16/FP32); `smooth_scales`/`offsets`/`group_index` (OPTIONAL Tensor)
- Output parameters: `y` (REQUIRED Tensor, INT8/INT4), `scale` (REQUIRED Tensor, FP32)
- Attrs: `activate_left`(bool,false), `quant_mode`(str,"dynamic"), `group_list_type`(int,0), `dst_type`(int,2)
- Key constraints: x.dim()>1; x last dim %2==0; x last dim<=8192; dst_type=29→x last dim %4==0; quant_mode∈{dynamic,static}; group_list_type∈{0,1}; dst_type∈{2,29}
- Operator type: **Forward** (no `_grad` suffix, no autograd)

## ATK Case Dir (C-delimit input, path-only)

- ATK case dir: `/home/c00618179/aclnnSwiGluQuantV2/op-plugin/aclnnSwiGluQuantV2` | mode: 4file (non-standard names) | files: `atk_swiglu_quant.yaml`, `executor_swiglu_quant.py`, `generator_npu_swiglu_quant.py`, `nodes_accuracy.yaml`
  - **Path only — main flow does NOT read these files; consumed by op-delimitation subagent at C-delimit.**

## Parameter Information Table (Cross-Skill Shared)

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line | .md Description | Constraints |
|---|------------|--------|------|-------------|---------|----------|---------------|-----------------|-------------|
| 1 | x | both | Tensor | REQUIRED | - | - | L24 | 输入待处理数据，公式中的x | last dim %2==0; dim>1; last dim<=8192; dst_type=29→last dim%4==0; dtype FP16/BF16/FP32 |
| 2 | smooth_scales | both | Tensor (Tensor?) | OPTIONAL | None | - | L30 | 量化的smooth_scales | FLOAT; 不支持空Tensor; shape [G,N] 或 [G] |
| 3 | offsets | both | Tensor (Tensor?) | OPTIONAL | None | - | L36 | 公式中的offsets | FLOAT; 动态量化传空; per_channel [G,N], per_tensor [G] |
| 4 | group_index | both | Tensor (Tensor?) | OPTIONAL | None | - | L42 | MoE分组group_index | INT32; shape [G]; 元素非递减; max ≤ x除末维外所有维度之积 |
| 5 | activate_left | both | bool | OPTIONAL | False | OPTIONAL(Bool) | L60 | 左矩阵是否参与运算 | 用户必须传参 |
| 6 | quant_mode | both | str | OPTIONAL | "dynamic" | OPTIONAL(String) | L61 | static/dynamic/dynamic_msd | 仅支持 "dynamic" / "static" |
| 7 | group_list_type | both | int | OPTIONAL | 0 | OPTIONAL(Int, V2) | L62 | 0=cumsum / 1=count | 仅 0 / 1 |
| 8 | dst_type | both | int | OPTIONAL | 2 | OPTIONAL(Int, V2) | L63 | 2=int8 / 29=int4 | 仅 2 / 29；默认 2 |
| 9 | y (output) | both | Tensor | REQUIRED | - | - | L48 | yOut | INT8/INT4; last dim = x last dim/2 (int8), 或 /4 packed (int4) |
| 10 | scale (output) | both | Tensor | REQUIRED | - | - | L54 | scaleOut | FLOAT; shape = x 去最后一维 |

**Version depth**: `.Version(SWI_GLU_QUANT_VERSION_TWO=2)` marks attrs `group_list_type` (L62) + `dst_type` (L63). Base(V1) attrs: activate_left, quant_mode. The supplied `.md` is the V2 entry point (`aclnnSwiGluQuantV2`) covering base+V2 → implement **V2 (full)**.

**No conflicts** (parse `conflicts: []`). All params reconciled `both`.

## Information Required from User

(See C-confirm batch — answered below after confirmation.)

- [x] aclnn version variants: V2 only (single .md covers it)
- [x] SoC: AICore configs ascend910b / ascend910_93 / ascend950 / kirinx90 / kirin9030; tests target Ascend910B
- [x] .List overload: NO
- [x] additional variants: NO
- [x] internal_format_opapi: NO
- [x] special data format: INT4 (dst_type=29) packing handled in C++
- [x] autograd: NO (forward-only quant op)
- [x] **Graph mode (torch.compile): YES** → C-meta executes, remove `--disable_torchair`, graph test cases, graph doc example, Meta cross-verify
- [x] **ACLGraph update (Q2): NO** (not attention/dynamic-seq-len)
  - [x] Q2 gate: N/A (NO)

## Pitfall Alerts (from parse JSON `pitfall_alerts`)

(empty — parse surfaced no structural alerts)

## Manual Pitfall Alerts (from V1 reference case npu_swi_glu_quant-create.md — version-independent)

> Applied during C-cpp / C-test. Reference case is V1 (`npu_swi_glu_quant`/str), current is V2 (`npu_swi_glu_quant_v2`/str+int attrs). Pitfalls below are version-independent.

| # | Pitfall | Applied at | Applied? |
|---|---------|-----------|----------|
| 1 | `quant_mode` string → pass as `const char*` via `std::string(...).c_str()` (null-terminated safe), not `string_view::data()` | C-cpp | ☐ |
| 2 | SwiGLU A/B order: `activate_left=False` → swish on **second half** (B=first half). Reference impl got this backwards. | C-test (golden) | ☐ |
| 3 | scale semantics (dynamic): NPU returns quant scale = dst_max/row_max (127/max_abs for int8, 7/max_abs for int4), NOT dequant scale. | C-test (golden) | ☐ |
| 4 | group_index = cumulative boundaries (cumsum mode, group_list_type=0): [2,4] = rows 0:2 and 2:4. NOT per-group lengths. | C-test (golden) | ☐ |
| 5 | INT4 (dst_type=29): y dtype container + last dim = x_last_dim/4 (2-per-byte). x last dim must be %4==0. | C-cpp + C-meta | ☐ |

## Constraint Logic Checklist (Cross-Step Verification)

| # | Constraint Rule | .md Source | C++ (C-cpp) | Meta (C-meta) | Test (C-test) | Doc (C-docs) |
|---|-----------------|------------|--------------|---------------|---------------|--------------|
| 1 | x.dim() > 1 | L125 | ☐ | ☐ | ☐ | ☐ |
| 2 | x last dim % 2 == 0 | L125 | ☐ | ☐ | ☐ | ☐ |
| 3 | x last dim <= 8192 | L125 | ☐ | ☐ | ☐ | ☐ |
| 4 | dst_type=29 (INT4) → x last dim % 4 == 0 | L125 | ☐ | ☐ | ☐ | ☐ |
| 5 | quant_mode ∈ {"dynamic","static"} | L175 | ☐ | ☐ | ☐ | ☐ |
| 6 | group_list_type ∈ {0,1} | L185 | ☐ | ☐ | ☐ | ☐ |
| 7 | dst_type ∈ {2,29} | L195 | ☐ | ☐ | ☐ | ☐ |
| 8 | y shape: last dim = x_last/2 (int8) ; dtype INT8/INT4 | L205 | ☐ | ☐ | ☐ | ☐ |
| 9 | scale shape: x without last dim; dtype FP32 | L215 | ☐ | ☐ | ☐ | ☐ |
| 10 | input dtype: x FP16/BF16/FP32; smooth_scales/offsets FP32; group_index INT32 | L126,135,145,156 | ☐ | ☐ | ☐ | ☐ |

## Constraint Coverage Self-Check (C-cpp+)

(filled at C-constraints)

## Complex Operator Grouped Mode

**Not triggered.** Output shape is a single linear derivation (y = x with last_dim halved/halved-again-for-int4; scale = x drop last dim). dtype branches on a single param (dst_type). No ≥2 independent conditional rules, no multi-dimension branching. (dst_type INT8/INT4 is one dimension with an override note below, not 2+ independent dimensions.)

## Parameter Coverage Checklist

(filled at C-test)

## Table Migration Checklist

| # | Table Content | .md Line | Doc Location | Status |
|---|---------------|----------|--------------|--------|
| 1 | 产品支持情况表 (950PR/950DT, A3, A2 √; others ×) | L7-14 | 产品支持 | ☐ |
| 2 | GetWorkspaceSize 参数说明表 | L99-242 | 参数说明 | ☐ |
| 3 | 错误码表 (161001/161002) | L250-285 | 约束/错误码 | ☐ |

## Compatibility Registration Checklist (C-baseline)

(filled at C-baseline)

## Progress

- [x] C-parse: Parse input files
- [x] C-confirm: Confirm operator-specific information with user
  - [x] **Graph mode support confirmed (YES)**
- [x] C-yaml: Generate YAML
- [x] C-derivatives: Configure derivatives.yaml (autograd = NO → skip)
- [x] C-cpp: Implement C++
- [x] C-constraints: Constraint self-check
- [x] C-meta: Add Meta registration [graph mode YES → execute]
- [x] C-verify: Signature verification [+ Meta Shape/Dtype cross-verify]
- [x] C-build: Build [graph mode YES → remove --disable_torchair]
- [x] C-test: Tests (UT + FakeTensor) [+ graph mode test cases] — FakeTensor 2/2, UT 32/32 PASS
- [x] C-docs: API docs + nav + docstrings [+ graph mode call example]
- [x] C-baseline: Compatibility baseline (schema JSON + allowlist) — test_compatibility PASS
- [x] C-delimit: ATK cases via op-delimitation subagent — 50/50 bit-exact SUCCESS, acc_pass_result:Pass
- [x] C-summary: Integration summary

## UT Generation Info

### Operator Basic Info
- Operator Name: swi_glu_quant_v2
- torch_npu function name: npu_swi_glu_quant_v2
- Operator Type: custom
- Target Directory: test/test_custom_ops
- Mode: create
- Operator Classification: quantization

### Source File Paths
- _def.cpp: aclnn_input/swi_glu_quant_v2_def.cpp
- .md doc: aclnn_input/aclnnSwiGluQuantV2.md

### Input/Output Spec

**Input parameter list**:

| Param Name | Type | Required/Optional | Supported dtypes | Constraints |
|------------|------|-------------------|------------------|-------------|
| x | Tensor | Required | bfloat16, float32, float16 | dim>1; last dim %2==0; last dim<=8192; dst_type=29 → last dim %4==0; non-contiguous unsupported |
| smooth_scales | Tensor | Optional | float32 | not empty; shape [G,N] or [G]; G=group count, N=x_last/2 |
| offsets | Tensor | Optional | float32 | ignored in dynamic (pass None); static: per_channel [G,N], per_tensor [G] (same dtype/shape as smooth_scales) |
| group_index | Tensor | Optional | int32 | shape [G]; elements non-decreasing; max ≤ prod(x dims except last); MoE vs non-MoE (None) |
| activate_left | bool (Attr) | Optional | — | bool; True=swish on first half, False=swish on second half; Default False |
| quant_mode | str (Attr) | Optional | — | str; "dynamic" \| "static" (dynamic_msd unsupported); Default "dynamic" |
| group_list_type | int (Attr) | Optional | — | int; 0=cumsum \| 1=count; Default 0; Interaction: group_list_type × group_index (cumsum boundaries vs count) |
| dst_type | int (Attr) | Optional | — | int; 2=int8 \| 29=int4; Default 2; Interaction: dst_type × y dtype & shape (int8: last/2; int4: last/4 packed) |

**Output specs**:

| Output Name | Type | Shape Rule | Dtype Rule |
|-------------|------|------------|------------|
| y | Tensor | shape = x.shape with last dim changed: INT8(dst_type=2) → x_last/2; INT4(dst_type=29) → x_last/4 (2-per-byte packed in uint8) | int8 (dst_type=2); uint8 (dst_type=29, logical INT4) |
| scale | Tensor | shape = x.shape without the last dim | float32 |

**Constraints extracted from operator doc**:
- INT4 (dst_type=29): 2-per-byte packing — y dtype container = uint8, last dim = x_last/4 (NOT quint4x2/int32 /8). Confirmed by meta-guide (aclnnSwiGluQuantV2 uses 2-per-byte) + build-tree test (npu_y.dtype==uint8, shape x_last/4).

### Coverage Matrix

> Auto-generated by scaffold_coverage.py, add/delete applied per Constraints column markers.

| Test Item | Covered Params | Data Type | Dimensions | Scenario |
|-----------|----------------|-----------|------------|----------|
| test_basic_bf16 | x | bfloat16 | (32, 256) | Basic functionality |
| test_basic_fp16 | x | float16 | (32, 256) | Basic functionality |
| test_basic_fp32 | x | float32 | (32, 256) | Basic functionality |
| test_smooth_scales_offsets_group_index_bf16 | x, smooth_scales, offsets, group_index | bfloat16 | (2, 32, 256) | 可选参数-smooth_scales+offsets+group_index(bf16) |
| test_smooth_scales_offsets_group_index_fp16 | x, smooth_scales, offsets, group_index | float16 | (2, 32, 256) | 可选参数-smooth_scales+offsets+group_index(fp16) |
| test_smooth_scales_offsets_bf16 | x, smooth_scales, offsets | bfloat16 | (2, 32, 256) | 可选参数-smooth_scales+offsets(bf16) |
| test_smooth_scales_offsets_fp16 | x, smooth_scales, offsets | float16 | (2, 32, 256) | 可选参数-smooth_scales+offsets(fp16) |
| test_smooth_scales_group_index_bf16 | x, smooth_scales, group_index | bfloat16 | (2, 32, 256) | 可选参数-smooth_scales+group_index(bf16) |
| test_smooth_scales_group_index_fp16 | x, smooth_scales, group_index | float16 | (2, 32, 256) | 可选参数-smooth_scales+group_index(fp16) |
| test_smooth_scales_bf16 | x, smooth_scales | bfloat16 | (2, 32, 256) | 可选参数-smooth_scales(bf16) |
| test_smooth_scales_fp16 | x, smooth_scales | float16 | (2, 32, 256) | 可选参数-smooth_scales(fp16) |
| test_offsets_group_index_bf16 | x, offsets, group_index | bfloat16 | (2, 32, 256) | 可选参数-offsets+group_index(bf16) |
| test_offsets_group_index_fp16 | x, offsets, group_index | float16 | (2, 32, 256) | 可选参数-offsets+group_index(fp16) |
| test_offsets_bf16 | x, offsets | bfloat16 | (2, 32, 256) | 可选参数-offsets(bf16) |
| test_offsets_fp16 | x, offsets | float16 | (2, 32, 256) | 可选参数-offsets(fp16) |
| test_group_index_bf16 | x, group_index | bfloat16 | (2, 32, 256) | 可选参数-group_index(bf16) |
| test_group_index_fp16 | x, group_index | float16 | (2, 32, 256) | 可选参数-group_index(fp16) |
| test_dst_type_int8 | x, dst_type | float32 | (32, 256) | dst_type=int8(2), golden compare |
| test_dst_type_int4 | x, dst_type | float32 | (32, 256) | dst_type=int4(29): uint8 packed, last=x_last/4 (structure/dtype check) |
| test_int4_output_shape | x, dst_type | float32 | (4, 64) | INT4 输出形状验证 (末维/4, uint8 2-per-byte) |
| test_scale_value | x | float32 | (2, 32, 256) | scale数值验证 (比golden, 非纯shape) |
| test_large_input | x | float32 | (8, 8192) | Large/boundary data: last dim=8192 (max), golden compare |
| test_activate_left_true | x, activate_left | float32 | (32, 256) | activate_left=True (swish on first half), golden compare |
| test_quant_mode_static | x, smooth_scales, offsets, quant_mode | float32 | (32, 256) | quant_mode=static (per_tensor, with offsets) |
| test_group_list_type_count | x, smooth_scales, group_index, group_list_type | float32 | (2, 32, 256) | group_list_type=1 (count mode) |
| test_graph_dynamic_int8 | x, smooth_scales | float32 | (32, 256) | Graph mode torch.compile(npugraph_ex), dynamic int8 |
| test_graph_static | x, smooth_scales, offsets, quant_mode | float32 | (16, 128) | Graph mode torch.compile(npugraph_ex), static |

**Auto-generated stats**: 22 test items.

**Rule compliance self-check**:
- Basic functionality dtype coverage: 3 rows ✅
- Output dtype full coverage: expected 2 values ['int8', 'quint4x2'], expanded 2 rows ✅
- Optional param coverage: 3 optional Tensors ✅
- Interaction / not-applicable: Step 3 handles row add/delete and expansion per `Interaction: A×B` / `Not applicable: param` anchors in Constraints column
- Required scenario check (category=quantization):
    - dst_type all values: ✅ covered
    - smooth_scales: ✅ covered
    - group_index: ✅ covered
    - INT4 shape verify: ✅ covered

### Negative Coverage Matrix

> Auto-generated by generate_negative_cases(), derived from constraint signals for illegal input combinations.

| Test Item | Covered Params | Data Type | Dimensions | Scenario | Violated Field | Violation Construction | Expected |
|-----------|----------------|-----------|------------|----------|----------------|------------------------|----------|
| test_neg_x_invalid_dtype | x | float32 | (32, 256) | Negative-dtype domain | x | dtype=int64 | raises:PARAM_INVALID |
| test_neg_smooth_scales_invalid_dtype | x, smooth_scales | float32 | (32, 256) | Negative-dtype domain | smooth_scales | dtype=int32 | raises:PARAM_INVALID |
| test_neg_offsets_invalid_dtype | x, offsets | float32 | (32, 256) | Negative-dtype domain | offsets | dtype=int32 | raises:PARAM_INVALID |
| test_neg_group_index_invalid_dtype | x, group_index | float32 | (32, 256) | Negative-dtype domain | group_index | dtype=float32 | raises:PARAM_INVALID |
| test_neg_dst_type_invalid | x, dst_type | float32 | (32, 256) | Negative-value domain | dst_type | value=99 (not in {2,29}) | raises:PARAM (msg "dst_type") |

**Negative category coverage**: 2 categories, 5 rows total

### Finalization Add/Delete Log

- Delete test_neg_scale_shape_mismatch: scale is an OUTPUT of this op (returns (y, scale)), not an input — no input scale param to violate; row not applicable.
- Add test_activate_left_true: cover activate_left=True (swish on first half) — attr value coverage.
- Add test_quant_mode_static: cover quant_mode="static" path (with offsets) — attr value coverage.
- Add test_group_list_type_count: cover group_list_type=1 (count mode) — attr value coverage.
- Add test_graph_dynamic_int8: graph mode (torch.compile npugraph_ex) dynamic int8 — graph mode YES.
- Add test_graph_static: graph mode static — graph mode YES.
- Fix test_dst_type_int4 / test_int4_output_shape: scaffold used wrong INT4 convention (quint4x2 /8); corrected to this op's 2-per-byte uint8 /4 (per meta-guide + build-tree test).

> 格式契约: `Delete <exact_method_name>: <reason>` / `Add <exact_method_name>: <reason>`
> `<exact_method_name>` 必须与 Coverage Matrix 的 Test Item 列值**逐字一致**, 禁自然语言概述。
> reconcile.py 依赖此格式做删除放行判定, 格式不符 → [BAD-LOG] 告警。

<!-- 在此下方逐行记录增删, 一行一项 -->
