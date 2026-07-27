# SwiGluQuantV2 Integration Checklist

> **Mode: CREATE** — new operator integration of `aclnnSwiGluQuantV2` (SwiGlu + 量化融合算子, V2). V1 reference case: `references/examples/create-mode/npu_swi_glu_quant-create.md`.

## Parse Results

- aclnn function name: aclnnSwiGluQuantV2
- torch_npu function name: npu_swi_glu_quant_v2 (script canonical camelCase split; matches V1 case `npu_swi_glu_quant` naming style — to confirm in C-confirm)
- Input parameters: x, smooth_scales, offsets, group_index (4 inputs, 3 optional)
- Output parameters: y (INT8/INT4), scale (FLOAT) — 2 outputs, both REQUIRED
- Key constraints: x last dim multiple-of-2 (int8) / multiple-of-4 (int4); x dims>1; x last dim ≤8192; dst_type ∈ {2(int8),29(int4)}; group_list_type ∈ {0,1}; quant_mode ∈ {dynamic,static}; group_index non-decreasing & bounded; y shape = halve x last dim; scale shape = x[:-1]
- Operator type: Forward (no `_grad` suffix); quantization op (likely non-differentiable)

## ATK Case Dir (C-delimit input, path-only)

> Path only — NOT read by main flow. Consumed by op-delimitation subagent at C-delimit.

- ATK case dir: /home/c00618179/swi_opencode/op-plugin/aclnnSwiGluQuantV2 | mode: 4file (variant names) | files: atk_swiglu_quant.yaml, generator_npu_swiglu_quant.py, executor_swiglu_quant.py, nodes_accuracy.yaml
  - atk.yaml-equiv: /home/c00618179/swi_opencode/op-plugin/aclnnSwiGluQuantV2/atk_swiglu_quant.yaml
  - generate.py-equiv: /home/c00618179/swi_opencode/op-plugin/aclnnSwiGluQuantV2/generator_npu_swiglu_quant.py
  - executor.py-equiv: /home/c00618179/swi_opencode/op-plugin/aclnnSwiGluQuantV2/executor_swiglu_quant.py
  - node.yaml-equiv: /home/c00618179/swi_opencode/op-plugin/aclnnSwiGluQuantV2/nodes_accuracy.yaml

## Parameter Information Table (Cross-Skill Shared)

**C-parse generates this table. All subsequent steps reference this as the single source of truth.**

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line | .md Description | Constraints |
|---|------------|--------|------|-------------|---------|----------|---------------|-----------------|-------------|
| 1 | x | both | Tensor | REQUIRED | - | - | L24 | 输入待处理的数据，公式中的x | last dim 多2的倍数; dims>1; last dim ≤8192; dstType=29时last dim多4的倍数; dtype FLOAT16/BF16/FLOAT |
| 2 | smooth_scales | both | Tensor | OPTIONAL | - | - | L30 | 量化的smooth_scales | 不支持空Tensor; shape [G,N]或[G,]; G=group_index分组数, N=x last dim/2; dtype FLOAT |
| 3 | offsets | both | Tensor | OPTIONAL | - | - | L36 | 公式中的offsets | 动态量化不生效(传空); 静态量化FLOAT; per_channel [G,N]; per_tensor [G,] 且 dtype+shape 与 smooth_scales 一致 |
| 4 | group_index | both | Tensor | OPTIONAL | - | - | L42 | MoE分组的group_index | shape [G,]; 元素非递减; max ≤ ∏(x dims 除最后一维); G ≤ 同前; dtype INT32 |
| 5 | y (output) | both | Tensor | REQUIRED | - | - | L48 | 输出张量 yOut | shape last dim = x last dim/2, 其余同x; dtype INT8/INT4 |
| 6 | scale (output) | both | Tensor | REQUIRED | - | - | L54 | 输出张量 scaleOut | shape = x[:-1] (无最后一维), 其余同x; dtype FLOAT |
| 7 | activate_left | both | bool | OPTIONAL | false | OPTIONAL (Bool) | L60 | 表示左矩阵是否参与运算，用户必须传参 | - |
| 8 | quant_mode | both | str | OPTIONAL | "dynamic" | OPTIONAL (String) | L61 | 量化模式 | ∈ {"static","dynamic","dynamic_msd"}; 当前仅支持 dynamic/static; 静态仅支持 per_tensor/per_channel |
| 9 | group_list_type | both | int | OPTIONAL | 0 | OPTIONAL (Int, Version TWO) | L62 | groupListType 模式 | ∈ {0(cumsum), 1(count)}; 用户必须传参 |
| 10 | dst_type | both | int | OPTIONAL | ge::DT_INT8 | OPTIONAL (Int, Version TWO) | L63 | 量化输出dtype控制 | ∈ {2(int8), 29(int4)}; 默认2 |

**Source column**: all `both` — no conflicts (parse `conflicts: []`).

## Information Required from User

**C-confirm RESOLVED (2026-07-27)**:
- [x] aclnn version variants: `.Version(SWI_GLU_QUANT_VERSION_TWO=2)` on `group_list_type` + `dst_type`; .md = ONLY `aclnnSwiGluQuantV2`. → **implement V2 only** (base V1 not provided; scope control).
- [x] **Operator name** = `npu_swi_glu_quant_v2` (user chose script canonical camelCase split of aclnnSwiGluQuantV2; NOT deleted real-lib `npu_swiglu_quant`). C++ file: `SwiGluQuantV2KernelNpuOpApi.cpp` (align with repo `...KernelNpuOpApi.cpp` convention, e.g. DequantSwigluQuant uses `...OpApi.cpp` — verify exact in C-cpp).
- [x] **quant_mode type** = `str` (user chose strict input-files scope control; pass string→char* to aclnn via `std::string` + `.c_str()` per V1 case pitfall #1 — safer than string_view).
- [x] SoC version dispatch = **all_version op_api** (no torch_npu-layer SoC branching; only one aclnn `aclnnSwiGluQuantV2`, unlike DequantSwigluQuant which branches V1/V2 by SoC). _def.cpp kernel configs (ascend910b/910_93/950/kirin) are kernel-level, not torch_npu-level.
- [x] .List overload = NO (no int[] tensor params; dst_type/group_list_type are scalar int).
- [x] Additional torch_npu variants = NO.
- [x] internal_format_opapi = NO (ND format).
- [x] Special data format = NO mxfp4; INT4 output representation = **`c10::ScalarType::Char` (int8-backed buffer) with halved shape for both dst_type=2(int8) and dst_type=29(int4)**, pass dst_type int to aclnn (matches sibling DequantSwigluQuant pattern + .md "yOut shape 末维/2 for both int8/int4"). Meta y dtype = Char.
- [x] **Autograd** = NO (quant cast non-differentiable; V1 case had none; no backward aclnn in inputs). → SKIP C-derivatives.
- [x] **Q1 torch.compile graph capture** = **YES** (user confirmed; consistent with V1 case having `npu_swi_glu_quant_meta`). → EXECUTE C-meta, remove `--disable_torchair` in build, add graph test cases + graph doc example, C-verify Meta Shape/Dtype cross-verify.
- [x] **Q2 ACLGraph update** = NO (auto-determined: not attention-class dynamic seq-len; no SymInt[]/actual_seq_*/context_lens/prefix params; not FA/IFA/MLA/paged-attention family). → skip all ACLGraph steps.

### Conflict Confirmation (from Parameter Information Table)

> No ⚠️ conflicts — parse `conflicts: []`, all params source=both.

| # | Parameter | Conflict Description | Resolution |
|---|-----------|---------------------|------------|
| - | (none) | - | - |

- [x] All ⚠️ conflicts confirmed by user (N/A — none)

## Pitfall Alerts (from parse JSON `pitfall_alerts`)

| # | Alert id | Trigger | Must-read case | Gates steps | Applied? |
|---|----------|---------|----------------|-------------|----------|
| 1 | conditional_output_allocation | bool attr(s) ['activate_left'] + 2 outputs → flag may gate conditional output; allocating output as {0}/zero on flag=False can trip kernel tiling CheckOutShape (err 561002) | references/examples/create-mode/npu_moe_gating_top_k-create.md | C-cpp, C-meta | ☐ (read before C-cpp) |

> Note: for THIS op both y/scale are REQUIRED outputs per _def.cpp (always allocated). `activate_left` controls SwiGlu left-matrix participation, not output existence. Alert is structurally triggered but likely false-positive — STILL must read the case before C-cpp/C-meta per hard-gate rule, and decide allocation shape from this op's aclnn header + .md (never copy sibling's {0}).

## Constraint Logic Checklist (Cross-Step Verification)

| # | Constraint Rule | .md Source | C++ (C-cpp) | Meta (C-meta) | Test (C-test) | Doc (C-docs) |
|---|-----------------|------------|--------------|---------------|---------------|--------------|
| 1 | x last dim 多2的倍数; dims>1; last dim ≤8192 | L125 | ☐ | ☐ | ☐ | ☐ |
| 2 | dst_type=29 → x last dim 多4的倍数 | L125 | ☐ | ☐ | ☐ | ☐ |
| 3 | x dtype ∈ {FLOAT16,BF16,FLOAT} | L126 | ☐ | ☐ | ☐ | ☐ |
| 4 | smooth_scales 不支持空Tensor; shape [G,N]/[G,]; dtype FLOAT | L135 | ☐ | ☐ | ☐ | ☐ |
| 5 | offsets: 动态传空; 静态FLOAT; per_channel [G,N]; per_tensor [G,] 且与 smooth_scales 一致 | L145 | ☐ | ☐ | ☐ | ☐ |
| 6 | group_index: shape [G,]; 元素非递减; max & G ≤ ∏(x dims 除最后一维); dtype INT32 | L155 | ☐ | ☐ | ☐ | ☐ |
| 7 | activate_left: bool 必传 | L164 | ☐ | ☐ | ☐ | ☐ |
| 8 | quant_mode ∈ {static,dynamic,dynamic_msd}; 当前仅 dynamic/static; 静态仅 per_tensor/per_channel | L175 | ☐ | ☐ | ☐ | ☐ |
| 9 | group_list_type ∈ {0,1} 必传 | L185 | ☐ | ☐ | ☐ | ☐ |
| 10 | dst_type ∈ {2,29} 默认2 | L195 | ☐ | ☐ | ☐ | ☐ |
| 11 | y shape = halve x last dim, 其余同x; dtype INT8/INT4 | L205 | ☐ | ☐ | ☐ | ☐ |
| 12 | scale shape = x[:-1]; dtype FLOAT | L215 | ☐ | ☐ | ☐ | ☐ |

## Constraint Coverage Self-Check (C-cpp+)

> C-constraints done (2026-07-27). check-constraints script: 12 constraints, 10 covered / 2 uncovered (both FALSE POSITIVES — LLM-verified). Real TORCH_CHECKs at L37-74.

| # | Constraint Rule | C++ TORCH_CHECK Found? | Status |
|---|-----------------|----------------------|--------|
| 1 | x dims>1 / last dim even / ≤8192 | L37, L39, L40 | ✅ |
| 2 | dst_type=29 → x last dim multiple of 4 | L44 | ✅ |
| 3 | x dtype ∈ {f16,bf16,f32} | L52 | ✅ |
| 4 | smooth_scales 非空 + FLOAT (if provided) | L62, L64 | ✅ |
| 5 | offsets FLOAT (if provided) | L69 | ✅ (shape per_channel/per_tensor left to aclnn) |
| 6 | group_index INT32 (if provided) | L74 | ✅ (non-decreasing/bound left to aclnn) |
| 7 | activate_left bool 必传 | signature param (default False) | ✅ N/A (bool always valid) — false-positive uncovered |
| 8 | quant_mode ∈ {dynamic,static} | L47 | ✅ |
| 9 | group_list_type ∈ {0,1} | L49 | ✅ |
| 10 | dst_type ∈ {2,29} | L41 | ✅ |
| 11 | y shape = halve x last dim (int8: x/2; int4: x/4 physical) | y_size computation L92-99 | ✅ structural |
| 12 | scale shape = x[:-1], FLOAT | scale_size + kFloat L101 | ✅ |

## Parameter Coverage Checklist

> All OPTIONAL inputs (smooth_scales, offsets, group_index) must be covered in ≥1 test case.

| # | Parameter Name | Scenario | Test Case Name | Status |
|---|----------------|----------|----------------|--------|
| 1 | smooth_scales | MoE dynamic quant | - | ☐ |
| 2 | offsets | static per_tensor / per_channel | - | ☐ |
| 3 | group_index | MoE scene | - | ☐ |
| 4 | (none) group_index=None | non-MoE scene | - | ☐ |
| 5 | dst_type=29 (INT4) | int4 quant | - | ☐ |

## Table Migration Checklist

| # | Table Content | .md Line | Doc Location | Status |
|---|---------------|----------|--------------|--------|
| 1 | 产品支持情况表 | L7 | Doc product-support section | ☐ |
| 2 | 参数说明表 (GetWorkspaceSize) | L99 | Doc params section | ☐ |
| 3 | 返回码/错误码表 | L250 | Doc constraints/errors | ☐ |

## Derivatives Configuration Checklist (C-derivatives)

> Only applicable when autograd confirmed YES. Likely NO for quant op.

| # | Item | Details | Status |
|---|------|---------|--------|
| - | (pending C-confirm) | - | ☐ |

## Compatibility Registration Checklist (C-baseline)

| # | Item | Details | Status |
|---|------|---------|--------|
| 1 | op_api: entry in schema JSON | - | ☐ |
| 2 | func: entry in schema JSON | - | ☐ |
| 3 | Version tags | "newest" for new op | ☐ |
| 4 | allowlist entry | all_version | ☐ |
| 5 | Signature verified | - | ☐ |

## Progress

- [x] C-parse: Parse input files (2026-07-27)
- [x] C-confirm: Confirm operator-specific information with user (2026-07-27)
  - [x] **Graph mode support confirmed: YES (Q1); ACLGraph Q2=NO**
- [x] C-yaml: Generate YAML (2026-07-27)
- [x] C-derivatives: SKIPPED (autograd=NO per user) (2026-07-27)
- [x] C-cpp: Implement C++ (2026-07-27) — `op_plugin/ops/opapi/SwiGluQuantV2KernelNpuOpApi.cpp`
- [x] C-constraints: Constraint self-check (2026-07-27) — 12/12 covered (2 false-positive uncovered)
- [x] C-meta: Add Meta registration [图模式 YES → executed] (2026-07-27) — `npu_swi_glu_quant_v2_meta` in `_meta_registrations.py`; ge-convention dst_type 手动映射 (2→int8, 29→uint8+末维//2 for 2-per-byte, per meta-guide "aclnnSwiGluQuantV2 uses this")
  - [x] Parameter signature matches YAML func: line (required x positional, rest keyword-only)
  - [x] Output count matches YAML -> (Tensor, Tensor) → return (y, scale)
  - [x] All outputs use x.new_empty (device-correct)
  - [x] INT4 packed output dtype=torch.uint8 (2-per-byte), last_dim // 2 (per meta-guide; NOT int32//8)
  - [x] Quantization dst_type mapped (ge convention: 2→int8, 29→uint8; NOT TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP which is aten-ordinal)
  - [x] C++ TORCH_CHECK (shape/dtype) → meta torch._check added (dim>1/even/≤8192/int4-mult4/dst_type/quant_mode/group_list_type/x-dtype)
- [x] C-verify: Signature verification [+ Meta Shape/Dtype cross-verify] (2026-07-27) — verify script `passed:true`/0 warnings; Meta Shape/Dtype Logic Verification PASS (8/8 dims+dtypes match, sentinel branches equivalent)
- [ ] C-build: Build [图模式 YES → remove --disable_torchair]
- [x] C-test: Tests (UT + FakeTensor) [+ graph mode test cases if 图模式 YES] (2026-07-27)
  - **NPU Availability**: Unavailable (no local Ascend hardware; torch.npu.is_available()=False)
  - **Test Execution Status**: UT Skipped (no NPU); FakeTensor Executed
  - **Test Results**: FakeTensor Passed 3/3 (int8 meta, int4 meta, multi-dtype) in 0.025s — meta registration validated (graph mode gate); UT: Skipped (no NPU to run aclnn). reconcile.py PASS (21 positive + 6 negative methods, 0 missing).
- [x] C-docs: API docs + nav + docstrings [+ graph mode call example] (2026-07-27) — `docs/zh/custom_APIs/torch_npu/torch_npu-npu_swi_glu_quant_v2.md` + nav (torch_npu_list.md, menu_Pytorch_API.md) + docstring (_op_plugin_docs.py); graph mode example included (Q1=YES)
- [x] C-baseline: Compatibility baseline (schema JSON + allowlist) (2026-07-27) — 2 schema entries (func + op_api, all_version) + allowlist npu_swi_glu_quant_v2→all_version; build-verified (wheel built)
- [x] C-delimit: ATK cases run & locate/delimit via op-delimitation subagent (2026-07-27) — ATK 4-file adapted (V1→V2 executor rewrite: op name/quant_mode int→str/dst_type dtype→raw int 2/29, independently confirming C-confirm decisions); 2100 cases generated; `atk task -e 1` executed but FAILED at device init (aclInit 507899, no NPU), operator never reached; xlsx honestly recorded FAILED-environment (NOT fabricated PASS); op-level conclusion PENDING NPU access. Artifacts in `npu_swi_glu_quant_v2_pta_atk/`.
- [x] C-summary: Integration summary (2026-07-27) — `notes/summary_2026-07-27.md`

## UT Generation Info

### Operator Basic Info
- Operator Name: swi_glu_quant_v2
- torch_npu function name: npu_swi_glu_quant_v2
- Operator Type: custom
- Target Directory: test/test_custom_ops
- Mode: create
- Operator Classification: quantization

### Source File Paths
- _def.cpp: /home/c00618179/swi_opencode/op-plugin/aclnn_input/swi_glu_quant_v2_def.cpp
- .md doc: /home/c00618179/swi_opencode/op-plugin/aclnn_input/aclnnSwiGluQuantV2.md

### Input/Output Spec

**Input parameter list**:

| Param Name | Type | Required/Optional | Supported dtypes | Constraints |
|------------|------|-------------------|------------------|-------------|
| x | Tensor | Required | bfloat16, float32, float16 | dims > 1; last dim multiple of 2; last dim ≤ 8192; when dst_type=29 last dim multiple of 4; dtype FLOAT16/BF16/FLOAT |
| smooth_scales | Tensor | Optional | float32 | does not support empty tensor; shape [G,N] or [G,] (G=group_index group count, N=x last dim/2); dtype FLOAT; per_tensor mode: offsets dtype+shape must match smooth_scales |
| offsets | Tensor | Optional | float32 | Interaction: quant_mode×offsets, dynamic→offsets ignored (pass null); static→effective; per_channel shape [G,N]; per_tensor shape [G,] matching smooth_scales; dtype FLOAT |
| group_index | Tensor | Optional | int32 | Interaction: group_index×scene, present→MoE; None→non-MoE; shape [G,]; elements non-decreasing; max & G ≤ product of x dims except last; dtype INT32 |
| activate_left | bool (Attr) | Optional | — | whether left matrix participates in SwiGlu; user must pass; default False |
| quant_mode | str (Attr) | Optional | — | ∈ {"dynamic","static"} (dynamic_msd listed but not currently supported); static only per_tensor/per_channel; default "dynamic" |
| group_list_type | int (Attr) | Optional | — | ∈ {0 (cumsum), 1 (count)}; user must pass; default 0 |
| dst_type | int (Attr) | Optional | — | ge-convention: ∈ {2 (int8), 29 (int4)}; default 2; Interaction: dst_type×y, 2→y dtype int8 shape last=x_last/2; 29→y dtype uint8 (2-per-byte) physical shape last=x_last/4, x last dim multiple of 4 |

**Output specs**:

| Output Name | Type | Shape Rule | Dtype Rule |
|-------------|------|------------|------------|
| y | Tensor | dims[0..-2] same as x; last dim = x_last/2 (int8); for int4 (dst_type=29) physical last dim = x_last/4 (2-per-byte uint8) | int8 (dst_type=2), uint8 (dst_type=29, 2-per-byte) |
| scale | Tensor | shape = x.shape[:-1] (x without last dim); dims same as x except no last dim | float32 |

**Constraints extracted from operator doc**:
- INT4(2-per-byte): output y last dim = input x last dim / 4 (SwiGlu halves → /2, then 2-per-byte packing → /2 again; aclnnSwiGluQuantV2 uses 2-per-byte per meta-guide, NOT quint4x2 int32-packed)

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
| test_dst_type_int8 | x, dst_type | bfloat16 | (32, 256) | dst_type=int8 |
| test_dst_type_int4 | x, dst_type | bfloat16 | (32, 256) | dst_type=int4 (29): y dtype uint8 (2-per-byte), physical last dim = x_last/4 |
| test_scale_value | x | bfloat16 | (2, 32, 256) | scale数值验证(比golden,非纯shape) |
| test_large_input | x | bfloat16 | (3072, 7168) | Large data (must compare scale golden, not pure shape smoke) |

**Auto-generated stats**: 20 test items.

**Rule compliance self-check**:
- Basic functionality dtype coverage: 3 rows ✅
- Output dtype full coverage: expected 1 values ['int8'], expanded 1 rows ✅
- Optional param coverage: 3 optional Tensors ✅
- Interaction / not-applicable: Step 3 handles row add/delete and expansion per `Interaction: A×B` / `Not applicable: param` anchors in Constraints column
- Required scenario check (category=quantization):
    - dst_type all values: ✅ covered
    - smooth_scales: ✅ covered
    - group_index: ✅ covered
    - INT4 shape verify: ⊘ N/A (operator lacks this param/capability)

### Negative Coverage Matrix

> Auto-generated by generate_negative_cases(), derived from constraint signals for illegal input combinations.

| Test Item | Covered Params | Data Type | Dimensions | Scenario | Violated Field | Violation Construction | Expected |
|-----------|----------------|-----------|------------|----------|----------------|------------------------|----------|
| test_neg_x_invalid_dtype | x | bfloat16 | (32, 256) | Negative-dtype domain | x | dtype=int64 | raises:PARAM_INVALID |
| test_neg_smooth_scales_invalid_dtype | x, smooth_scales | bfloat16 | (32, 256) | Negative-dtype domain | smooth_scales | dtype=int64 | raises:PARAM_INVALID |
| test_neg_offsets_invalid_dtype | x, offsets | bfloat16 | (32, 256) | Negative-dtype domain | offsets | dtype=int64 | raises:PARAM_INVALID |
| test_neg_group_index_invalid_dtype | x, group_index | bfloat16 | (32, 256) | Negative-dtype domain | group_index | dtype=float32 | raises:PARAM_INVALID |
| test_neg_scale_shape_mismatch | x, scale | bfloat16 | (32, 256) | Negative-shape link | scale | shape=(32, 512) | raises:PARAM_INVALID |
| test_neg_dst_type_invalid | x, dst_type | bfloat16 | (32, 256) | Negative-dtype domain | dst_type | dtype=bfloat16 | raises:PARAM_INVALID |

**Negative category coverage**: 2 categories, 6 rows total

### Finalization Add/Delete Log

> 格式契约: `Delete <exact_method_name>: <reason>` / `Add <exact_method_name>: <reason>`
> `<exact_method_name>` 必须与 Coverage Matrix 的 Test Item 列值**逐字一致**, 禁自然语言概述。
> reconcile.py 依赖此格式做删除放行判定, 格式不符 → [BAD-LOG] 告警。

<!-- 在此下方逐行记录增删, 一行一项 -->
Add test_dst_type_int4: Interaction: dst_type×y int4 branch — script baseline only had test_dst_type_int8; int4 (dst_type=29) branch missing per `Interaction: dst_type×y` anchor (29→y uint8 2-per-byte, physical shape last=x_last/4); required-scenario INT4 shape verify was mis-reported N/A by script.
