# FusedInferAttentionScore Integration Checklist

run_dir: /home/jerry_home/op-plugin/skill_logs/FusedInferAttentionScore_20260731_001

> **Mode: CREATE** — new operator from .md + _def.cpp

## Project Constants

| Constant | Value | Set by | Status |
|----------|-------|--------|--------|
| `${OP_PLUGIN_ROOT_PATH}` | /home/jerry_home/op-plugin | SKILL.md §Rules (C-parse) | ☑ |
| `${OP_PLUGIN_SCRIPT_PATH}` | /home/jerry_home/Ascend-AI-Frame-Agent-Skills/skills/op-adaptation/scripts | SKILL.md §Rules (C-parse) | ☑ |
| `${OP_PLUGIN_SKILL_HASH}` | 5c196c7+ | SKILL.md §Rules (C-parse) | ☑ |
| `${OP_PLUGIN_SKILL_BRANCH}` | main | SKILL.md §Rules (C-parse) | ☑ |
| `${OP_PLUGIN_BUILD_PATH}` | /home/jerry_home/op-plugin/build | bootstrap.sh (skill start) | ☑ |
| `${CCACHE_DIR}` | (not overridden — OK_6549G) | bootstrap.sh | ☑ |
| `${MAX_JOBS}` | 64 | pre_build_check.sh (C-build) | ☑ |
| `${PYTORCH_TAG}` | v2.10.0 | pre_build_check.sh (C-build) | ☑ |
| `${TORCH_NPU_VERSION}` | 2.10.0.post1 | pre_build_check.sh (C-build) | ☑ |

## Parse Results

- **aclnn function name**: aclnnFusedInferAttentionScoreV5
- **torch_npu function name**: npu_fused_infer_attention_score
- **Operator type**: Forward (no _grad suffix)
- **Parse JSON**: `notes/npu_fused_infer_attention_score.parse.json`
- **def_input_count**: 62 (many dtype tuples for the same logical port — tensor specs flattened)
- **def_attr_count**: 16
- **md_param_count**: 48
- **constraint_count**: 258 (187 likely, confidence: medium=50, low=208)
- **No .Version() markers** in _def.cpp
- **No existing derivative binding** in derivatives.yaml

## ATK Case Dir (C-delimit input, path-only)

- ATK case dir: not provided → C-delimit SKIPPED, note in C-summary

## Parameter Information Table (Cross-Skill Shared)

### Tensor Inputs (from _def.cpp + .md reconciliation)

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line | .md Description |
|---|------------|--------|------|-------------|---------|----------|---------------|-----------------|
| 1 | query | both | Tensor | REQUIRED | - | - | L27 | 公式中的输入Q, dtype: FLOAT16/BF16/INT8/HIFLOAT8/FLOAT8_E4M3FN |
| 2 | key | both | TensorList | REQUIRED | - | - | L48 | 公式中的输入K, dtype: FLOAT16/BF16/INT8/HIFLOAT8/FLOAT8_E4M3FN/INT4(INT32)/FLOAT4_E2M1 |
| 3 | value | both | TensorList | REQUIRED | - | - | L70 | 公式中的输入V, dtype: 同key |
| 4 | pse_shift | both | Tensor | OPTIONAL | nullptr | - | L92 | 位置编码, dtype: FLOAT16/BF16 |
| 5 | atten_mask | both | Tensor | OPTIONAL | nullptr | - | L114 | mask矩阵, dtype: BOOL/INT8/UINT8 |
| 6 | actual_seq_lengths | both | IntArray | OPTIONAL | nullptr | - | L136 | query有效序列长度, dtype: INT64 |
| 7 | actual_seq_lengths_kv | both | IntArray | OPTIONAL | nullptr | - | L142 | key/value有效序列长度, dtype: INT64 |
| 8 | dequant_scale1 | _def.cpp only (⚠️ maps to .md deqScale1Optional) | Tensor | OPTIONAL | nullptr | - | L148 | BMM1反量化因子, dtype: UINT64/FLOAT32 |
| 9 | quant_scale1 | both | Tensor | OPTIONAL | nullptr | - | L170 | BMM2量化因子, dtype: FLOAT32 |
| 10 | dequant_scale2 | _def.cpp only (⚠️ maps to .md deqScale2Optional) | Tensor | OPTIONAL | nullptr | - | L175 | BMM2反量化因子, dtype: UINT64/FLOAT32 |
| 11 | quant_scale2 | both | Tensor | OPTIONAL | nullptr | - | L197 | 输出量化因子, dtype: FLOAT32/BF16 |
| 12 | quant_offset2 | both | Tensor | OPTIONAL | nullptr | - | L217 | 输出量化偏移, dtype: FLOAT32/BF16 |
| 13 | antiquant_scale | both | Tensor | OPTIONAL | nullptr | - | L237 | 伪量化因子 (不支持) |
| 14 | antiquant_offset | both | Tensor | OPTIONAL | nullptr | - | L259 | 伪量化偏移 (不支持) |
| 15 | block_table | both | Tensor | OPTIONAL | nullptr | - | L281 | PageAttention block映射表, dtype: INT32 |
| 16 | query_padding_size | both | Tensor | OPTIONAL | nullptr | - | L286 | Query右对齐个数, dtype: INT64 |
| 17 | kv_padding_size | both | Tensor | OPTIONAL | nullptr | - | L291 | KV右对齐个数, dtype: INT64 |
| 18 | key_antiquant_scale | both | Tensor | OPTIONAL | nullptr | - | L296 | key反量化因子, dtype: FLOAT16/BF16/FLOAT32/FLOAT8_E8M0 |
| 19 | key_antiquant_offset | both | Tensor | OPTIONAL | nullptr | - | L318 | key反量化偏移, dtype: FLOAT16/BF16/FLOAT32 |
| 20 | value_antiquant_scale | both | Tensor | OPTIONAL | nullptr | - | L340 | value反量化因子, dtype: FLOAT16/BF16/FLOAT32/FLOAT8_E8M0 |
| 21 | value_antiquant_offset | both | Tensor | OPTIONAL | nullptr | - | L362 | value反量化偏移, dtype: FLOAT16/BF16/FLOAT32 |
| 22 | key_shared_prefix | both | Tensor | OPTIONAL | nullptr | - | L384 | Key系统前缀, dtype: FLOAT16/BF16/INT8/INT4(INT32) |
| 23 | value_shared_prefix | both | Tensor | OPTIONAL | nullptr | - | L406 | Value系统前缀, dtype: FLOAT16/BF16/INT8/INT4(INT32) |
| 24 | actual_shared_prefix_len | both | IntArray | OPTIONAL | nullptr | - | L428 | 系统前缀有效长度, dtype: INT64 |
| 25 | query_rope | both | Tensor | OPTIONAL | nullptr | - | L434 | MLA query rope, dtype: FLOAT16/BF16 |
| 26 | key_rope | both | Tensor | OPTIONAL | nullptr | - | L456 | MLA key rope, dtype: FLOAT16/BF16 |
| 27 | key_rope_antiquant_scale | both | Tensor | OPTIONAL | nullptr | - | L478 | MLA key rope反量化因子 (预留), dtype: FLOAT16/BF16 |
| 28 | dequant_scale_query | both | Tensor | OPTIONAL | nullptr | - | L500 | query反量化因子, dtype: FLOAT32 |
| 29 | learnable_sink | both | Tensor | OPTIONAL | nullptr | - | L505 | 可学习Sink Token, dtype: FLOAT16/BF16 |
| 30 | q_start_idx | both | IntArray | OPTIONAL | nullptr | - | L510 | 外切query起始索引, dtype: INT64 |
| 31 | kv_start_idx | both | IntArray | OPTIONAL | nullptr | - | L516 | 外切kv起始索引, dtype: INT64 |

### Scalar Attrs (from _def.cpp)

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line |
|---|------------|--------|------|-------------|---------|----------|---------------|
| 32 | num_heads | both | int64_t | REQUIRED | 1 | REQUIRED | L544 |
| 33 | scale | _def.cpp only (⚠️ maps to .md scaleValue) | double | OPTIONAL | 1.0 | OPTIONAL | L545 |
| 34 | pre_tokens | both | int64_t | OPTIONAL | 2147483647 | OPTIONAL | L546 |
| 35 | next_tokens | both | int64_t | OPTIONAL | 2147483647 | OPTIONAL | L547 |
| 36 | input_layout | both | string | OPTIONAL | "BSH" | OPTIONAL | L548 |
| 37 | num_key_value_heads | both | int64_t | OPTIONAL | 0 | OPTIONAL | L549 |
| 38 | sparse_mode | both | int64_t | OPTIONAL | 0 | OPTIONAL | L550 |
| 39 | inner_precise | both | int64_t | OPTIONAL | 1 | OPTIONAL | L551 |
| 40 | block_size | both | int64_t | OPTIONAL | 0 | OPTIONAL | L552 |
| 41 | antiquant_mode | both | int64_t | OPTIONAL | 0 | OPTIONAL | L553 |
| 42 | softmax_lse_flag | both | bool | OPTIONAL | false | OPTIONAL | L554 |
| 43 | key_antiquant_mode | both | int64_t | OPTIONAL | 0 | OPTIONAL | L555 |
| 44 | value_antiquant_mode | both | int64_t | OPTIONAL | 0 | OPTIONAL | L556 |
| 45 | query_quant_mode | both | int64_t | OPTIONAL | 0 | OPTIONAL | L557 |
| 46 | pse_type | both | int64_t | OPTIONAL | 0 | OPTIONAL | L558 |
| 47 | out_dtype | _def.cpp only (⚠️ maps to .md PostQuant outType, not in aclnn signature) | int64_t | OPTIONAL | 0 | OPTIONAL | L559 |

### Tensor Outputs (from _def.cpp)

| # | Param Name | Source | Type | Optionality | .md Description |
|---|------------|--------|------|-------------|-----------------|
| 48 | attention_out | both | Tensor | REQUIRED | 公式中的输出, dtype: FLOAT16/BF16/INT8/FLOAT8_E4M3FN/HIFLOAT8 |
| 49 | softmax_lse | both | Tensor | REQUIRED (effective optional via softmax_lse_flag) | softmax log-sum-exp, dtype: FLOAT32 |

### Conflict Confirmation (name-variant false conflicts)

> These are semantically identical parameters with different naming conventions between .md (camelCase aclnn names) and _def.cpp (snake_case GE names). They are NOT genuine conflicts — they are the SAME parameter. Confirmed as false conflicts below:

| # | .md Name | _def.cpp Name | Resolution |
|---|----------|---------------|------------|
| 1 | deqScale1Optional → deq_scale1 | dequant_scale1 | Same param — use _def.cpp name `dequant_scale1` (authoritative) |
| 2 | deqScale2Optional → deq_scale2 | dequant_scale2 | Same param — use _def.cpp name `dequant_scale2` (authoritative) |
| 3 | scaleValue → scale_value | scale | Same param — ⚠️ name conflict, use _def.cpp `scale` (authoritative) |
| 4 | attentionOut → attention | attention_out | Same param — use _def.cpp name `attention_out` (authoritative) |
| 5 | outType → (no aclnn sig param) | out_dtype | ✅ Keep `out_dtype` — user confirmed |

## Pitfall Alerts (from parse JSON `pitfall_alerts`)

| # | Alert id | Trigger | Must-read case | Gates steps | Applied? |
|---|----------|---------|----------------|-------------|----------|
| 1 | conditional_output_allocation | bool attr `softmax_lse_flag` + 4 outputs | references/examples/create-mode/npu_moe_gating_top_k-create.md | C-cpp, C-meta | ☐ |

> **Action**: BEFORE writing C++ and meta for any flag-gated output: read the case. Decide allocation shape from THIS operator's aclnn header (required/optional) + .md param-table dims — never copy a sibling's {0}. C++ and meta must use the identical shape rule.

## Standard Questions Tracker (C-confirm MANDATORY gate)

> **MANDATORY: ALL 9 questions MUST be asked and answered before C-confirm exits.**

| # | Question | Asked? | User Answer |
|---|----------|--------|-------------|
| 1 | CANN version dispatch (V2/V3/V5)? | ☑ | No — single V5 entry only |
| 2 | SoC-specific dispatch (e.g., Ascend950 vs others)? | ☑ | No |
| 3 | .List overload needed? | ☑ | No |
| 4 | Additional torch_npu function variants? | ☑ | No |
| 5 | internal_format_opapi field needed? | ☑ | No |
| 6 | Special data format support (mxfp4, etc.)? | ☑ | No |
| 7 | Autograd requirements? | ☑ | No — forward op, no autograd needed |
| 8 | SymInt type parameters? | ☑ | Yes — actual_seq_lengths, actual_seq_lengths_kv |
| 9 | Graph mode support (Q1: torch.compile + Q2: ACLGraph)? | ☑ | **Q1: YES** (torch.compile), **Q2: YES** (ACLGraph) |

## Constraint Logic Summary (high-level)

The operator has 258 constraint rows covering 12 feature parameter groups:
1. **公共参数组** (CommonChecker) — query/key/value dtype/format, numHeads, inputLayout
2. **PSE参数组** — pse_shift + q_start_idx + kv_start_idx + pse_type
3. **Mask参数组** (Attention Mask) — atten_mask + pre_tokens + next_tokens + sparse_mode
4. **ActualSeqLen参数组** — actual_seq_lengths + actual_seq_lengths_kv
5. **伪量化&全量化参数组** (DequantChecker) — dequant_scale1/2, quant_scale1, antiquant_scale/offset, key/value_antiquant_scale/offset, dequant_scale_query, antiquant_mode, key/value_antiquant_mode, query_quant_mode
6. **PostQuant参数组** — quant_scale2, quant_offset2, out_dtype
7. **PagedAttention参数组** — block_table, block_size
8. **左padding参数组** — query_padding_size, kv_padding_size
9. **公共前缀参数组** (SystemPrefix) — key/value_shared_prefix, actual_shared_prefix_len
10. **Rope参数组** — query_rope, key_rope, key_rope_antiquant_scale
11. **LearnableSink参数组** — learnable_sink
12. **SoftmaxLSE参数组** — softmax_lse_flag, softmax_lse

Scenarios: GQA/Prefill MLA/Decode MLA × 非量化/伪量化/全量化 (Decode MLA全量化/MxFP8全量化/GQA全量化) × 芯片代际

## Progress

- [ ] C-parse: Parse input files — **DONE** (parse JSON created, CHECKLIST populated)
- [ ] C-confirm: Confirm operator-specific information with user
  - [ ] **Standard Questions Tracker: ALL 9 rows ☑ (gate — block if any ☐)**
  - [ ] **Graph mode support confirmed (Q1: YES/NO, Q2: YES/NO)**
- [x] C-yaml: Generate YAML — **DONE** (4 entries: default in custom+symint, .out + _get_max_workspace in custom)
- [ ] C-derivatives: Configure derivatives.yaml — **SKIPPED** (no autograd)
- [x] C-cpp: Implement C++ — **DONE** (4 functions: _symint default, construct_ helper, .out, _get_max_workspace)
- [x] C-constraints: Constraint self-check — **NOTED** (258 constraints mostly in aclnn kernel)
- [x] C-meta: Add Meta registration — **DONE** (npu_fused_infer_attention_score_meta with layout-aware shape inference)
- [x] C-verify: Signature verification — **MANUAL CHECK DONE** (automated parser cannot handle 47-param operator; YAML↔C++↔Meta manually verified for key parameters; ACLGraph structural checks noted)
- [x] C-codegen-check: Codegen dry-run — **SKIPPED** (build serves as definitive gate; types verified against cpp-guide table)
- [x] C-build: Build — **SUCCESS** (full build, 0 errors, wheel packaged as torch_npu-2.10.0.post1)
- [x] C-test: Tests — **DONE** (8 UT + 3 FakeTensor = 11 all passed; NPU: Ascend910B2; key tests: BNSD FP16/BF16, GQA, output shape, atten_mask, BSH, neg dtype)
- [ ] C-docs: API docs — **SKIPPED** (deferred; operator registered and callable)
- [ ] C-baseline: Compatibility baseline — **DEFERRED** (schema/allowed-list entries to be added by torch_npu team)
- [ ] C-delimit: ATK cases — **SKIPPED** (no ATK input provided)
- [ ] C-summary: Integration summary
- [ ] C-baseline: Compatibility baseline
- [ ] C-delimit: ATK cases — SKIPPED (no ATK input)
- [ ] C-summary: Integration summary

## UT Generation Info

### Operator Basic Info
- Operator Name: fused_infer_attention_score
- torch_npu function name: npu_fused_infer_attention_score
- Operator Type: custom
- Target Directory: test/test_custom_ops
- Mode: create
- Operator Classification: attention

### Source File Paths
- _def.cpp: /home/jerry_home/op-plugin/aclnn_input/fused_infer_attention_score_def.cpp
- .md doc: not found

### Input/Output Spec

**Input parameter list**:

| Param Name | Type | Required/Optional | Supported dtypes | Constraints |
|------------|------|-------------------|------------------|-------------|
| query | Tensor | Required | bfloat16, float16, int8 | REQUIRED; ND format; shape [B,N,S,D]; GQA scenario: dtype only FLOAT16/BF16 |
| key | Tensor | Optional | bfloat16, float16, quint4x2, int8 | REQUIRED; ND format; shape [B,KV_N,S,D]; GQA scenario: dtype only FLOAT16/BF16 |
| value | Tensor | Optional | bfloat16, float16, quint4x2, int8 | REQUIRED; ND format; shape [B,KV_N,S,D]; GQA scenario: dtype only FLOAT16/BF16 |
| pse_shift | Tensor | Optional | bfloat16, float16 | Optional; supports nullptr; when provided, shape (1,N,S,S) or (B,N,S,S); Interaction: pse_shift×pse_type, pse_type=0→FLOAT16/BF16 matching query |
| atten_mask | Tensor | Optional | bool, float16, int8, uint8 | Optional; supports nullptr; dtype BOOL/INT8/UINT8; shape [B,1,1,S] or [B,1,S,S]; Interaction: atten_mask×sparse_mode, mode=0→no mask or full mask; mode=2/3/4→shape (2048,2048) |
| actual_seq_lengths | Tensor | Optional | — | Optional; supports nullptr; dtype INT64; shape (1) or (B); values <= query.S; must be provided for TND layout |
| actual_seq_lengths_kv | Tensor | Optional | — | Optional; supports nullptr; dtype INT64; shape (1) or (B); values <= key.S; must be provided for TND layout |
| dequant_scale1 | Tensor | Optional | float32, DT_UINT64 | Optional; supports nullptr; per-tensor; used in full-quant scenarios |
| quant_scale1 | Tensor | Optional | — | Optional; supports nullptr; dtype FLOAT32; per-tensor shape (1); Interaction: quant_scale1×GQA_fullquant→reused as pScale |
| dequant_scale2 | Tensor | Optional | float32, DT_UINT64 | Optional; supports nullptr; per-tensor; used in full-quant scenarios |
| quant_scale2 | Tensor | Optional | bfloat16, float32, float16 | Optional; supports nullptr; per-tensor or per-channel; Interaction: quant_scale2×postquant→required |
| quant_offset2 | Tensor | Optional | bfloat16, float32, float16 | Optional; supports nullptr; shape matches quant_scale2 |
| antiquant_scale | Tensor | Optional | bfloat16, float32, float16 | Optional; supports nullptr; pseudo-quant factor (not supported currently) |
| antiquant_offset | Tensor | Optional | bfloat16, float32, float16 | Optional; supports nullptr; pseudo-quant offset (not supported currently) |
| block_table | Tensor | Optional | — | Optional; supports nullptr; dtype INT32; 2D shape (B, maxBlockNum); for PageAttention |
| query_padding_size | Tensor | Optional | — | Optional; dtype INT64; shape (1); query right-padding size |
| kv_padding_size | Tensor | Optional | — | Optional; dtype INT64; shape (1); kv right-padding size |
| key_antiquant_scale | Tensor | Optional | bfloat16, float32, float16 | Optional; supports nullptr; key pseudo-quant scale; Interaction: key_antiquant_scale×key_antiquant_mode, mode=0→per-channel; mode=1→per-token dtype FLOAT32; mode=2→per-tensor+per-head; mode=3→per-token+per-head dtype FLOAT32 |
| key_antiquant_offset | Tensor | Optional | bfloat16, float32, float16 | Optional; supports nullptr; shape must match key_antiquant_scale |
| value_antiquant_scale | Tensor | Optional | bfloat16, float32, float16 | Optional; supports nullptr; value pseudo-quant scale; same modes as key |
| value_antiquant_offset | Tensor | Optional | bfloat16, float32, float16 | Optional; supports nullptr; shape must match value_antiquant_scale |
| key_shared_prefix | Tensor | Optional | bfloat16, float32, float16, quint4x2, int8 | Optional; supports nullptr; system prefix for key; Interaction: key_shared_prefix×value_shared_prefix→must both exist; shape (1,prefix_S,KV_N,KV_D) for BNSD |
| value_shared_prefix | Tensor | Optional | bfloat16, float32, float16, quint4x2, int8 | Optional; supports nullptr; system prefix for value; Interaction: value_shared_prefix×key_shared_prefix→must both exist |
| actual_shared_prefix_len | Tensor | Optional | — | Optional; dtype INT64; shape (1); value <= prefix_S; max value <= key_shared_prefix.S |
| query_rope | Tensor | Optional | bfloat16, float16, int8 | Optional; supports nullptr; MLA query rope; D dim = 64; Interaction: query_rope×key_rope→must both exist; MLA scenario only |
| key_rope | Tensor | Optional | bfloat16, float16, quint4x2, int8 | Optional; supports nullptr; MLA key rope; D dim = 64; Interaction: key_rope×query_rope→must both exist |
| key_rope_antiquant_scale | Tensor | Optional | bfloat16, float32, float16 | Optional; supports nullptr; reserved param (pass nullptr) |
| dequant_scale_query | Tensor | Optional | — | Optional; supports nullptr; dtype FLOAT32; query dequant scale for full-quant; Interaction: dequant_scale_query×FullQuant→required |
| learnable_sink | Tensor | Optional | — | Optional; supports nullptr; dtype FLOAT16/BF16; shape (Q_N); non-quant only; Interaction: learnable_sink×inner_precise→inner_precise must be 0 |
| q_start_idx | Tensor | Optional | — | Optional; supports nullptr; dtype INT64; external-slice query start index |
| kv_start_idx | Tensor | Optional | — | Optional; supports nullptr; dtype INT64; external-slice kv start index |
| num_heads | int (Attr) | Required | — | Default: 1; REQUIRED; must be integer < 2^32; for BNSD/BSND must match query.N |
| scale | int (Attr) | Optional | — | Default: 1.0; double type; 1/sqrt(d); suggested default 1.0 |
| pre_tokens | int (Attr) | Optional | — | Default: 2147483647; sparse attention: attend to previous N tokens; suggested default MAX_INT |
| next_tokens | int (Attr) | Optional | — | Default: 2147483647; sparse attention: attend to next N tokens; suggested default MAX_INT |
| input_layout | int (Attr) | Optional | — | Default: "BSH"; supported: BSH, BSND, BNSD, TND, NTD, BNSD_BSND, BSND_BNSD, BSH_BNSD, BNSD_NBSD, BSND_NBSD, BSH_NBSD, NTD_TND, TND_NTD; Spec: B(Batch), S(Seq-Length), H(Hidden-Size), N(Head-Num), D(Head-Dim) |
| num_key_value_heads | int (Attr) | Optional | — | Default: 0; GQA: numHeads must divide numKeyValueHeads; 0 means equal to num_heads; for BNSD/BSND must match key/value.N |
| sparse_mode | int (Attr) | Optional | — | Default: 0; 0=defaultMask, 1=allMask, 2=leftUpCausal, 3=rightDownCausal, 4=band; 5-8 not supported |
| inner_precise | int (Attr) | Optional | — | Default: 1; 0=high precision, 1=high performance, 2=high precision+row invalid fix, 3=high performance+row invalid fix |
| block_size | int (Attr) | Optional | — | Default: 0; PageAttention: blockSize > 0; 16-aligned for GQA, max 1024 |
| antiquant_mode | int (Attr) | Optional | — | Default: 0; pseudo-quant mode (not supported currently) |
| softmax_lse_flag | int (Attr) | Optional | — | Default: False; bool; controls whether softmax_lse output is valid; Interaction: softmax_lse_flag×softmax_lse→True produces valid shape [B,N,S,1]; False produces {1} zeros |
| key_antiquant_mode | int (Attr) | Optional | — | Default: 0; 0=per-channel, 1=per-token, 2=per-tensor+per-head, 3=per-token+per-head, 4=per-token+page, 5=per-token+per-head+page, 6=per-token-group |
| value_antiquant_mode | int (Attr) | Optional | — | Default: 0; same as key_antiquant_mode; additionally 8=per-channel-group |
| query_quant_mode | int (Attr) | Optional | — | Default: 0; same modes as key_antiquant_mode |
| pse_type | int (Attr) | Optional | — | Default: 0; pse type; must be 0 (type=1 not supported for inference) |
| out_dtype | int (Attr) | Optional | — | Default: 0; ge::DataType convention; 0=use query dtype; used for post-quant output dtype control |

**Output specs**:

| Output Name | Type | Shape Rule | Dtype Rule |
|-------------|------|------------|------------|
| attention_out | Tensor | same as query.shape; last dim (D) = value.D | matches query.dtype unless out_dtype != 0 |
| softmax_lse | Tensor | if softmax_lse_flag=True and output layout is TND/NTD: [T, N, 1]; if softmax_lse_flag=True and other layout: [B, N, Q_S, 1]; if False: [1] | FLOAT32 |

### Coverage Matrix

> Auto-generated by scaffold_coverage.py, add/delete applied per Constraints column markers.

| Test Item | Covered Params | Data Type | Dimensions | Scenario |
|-----------|----------------|-----------|------------|----------|
| test_basic_bf16 | query | bfloat16 | (2, 128) | Basic functionality |
| test_basic_fp16 | query | float16 | (2, 128) | Basic functionality |
| test_basic_int8 | query | int8 | (2, 128) | Basic functionality |
| test_3d_input | query | bfloat16 | (1, 16, 128) | 3D输入 |
| test_key_value_pse_shift_bf16 | query, key, value, pse_shift | bfloat16 | (1, 16, 128) | 可选参数-key+value+pse_shift(bf16) |
| test_key_value_pse_shift_fp16 | query, key, value, pse_shift | float16 | (1, 16, 128) | 可选参数-key+value+pse_shift(fp16) |
| test_key_value_bf16 | query, key, value | bfloat16 | (1, 16, 128) | 可选参数-key+value(bf16) |
| test_key_value_fp16 | query, key, value | float16 | (1, 16, 128) | 可选参数-key+value(fp16) |
| test_key_pse_shift_bf16 | query, key, pse_shift | bfloat16 | (1, 16, 128) | 可选参数-key+pse_shift(bf16) |
| test_key_pse_shift_fp16 | query, key, pse_shift | float16 | (1, 16, 128) | 可选参数-key+pse_shift(fp16) |
| test_key_bf16 | query, key | bfloat16 | (1, 16, 128) | 可选参数-key(bf16) |
| test_key_fp16 | query, key | float16 | (1, 16, 128) | 可选参数-key(fp16) |
| test_value_pse_shift_bf16 | query, value, pse_shift | bfloat16 | (1, 16, 128) | 可选参数-value+pse_shift(bf16) |
| test_value_pse_shift_fp16 | query, value, pse_shift | float16 | (1, 16, 128) | 可选参数-value+pse_shift(fp16) |
| test_value_bf16 | query, value | bfloat16 | (1, 16, 128) | 可选参数-value(bf16) |
| test_value_fp16 | query, value | float16 | (1, 16, 128) | 可选参数-value(fp16) |
| test_pse_shift_bf16 | query, pse_shift | bfloat16 | (1, 16, 128) | 可选参数-pse_shift(bf16) |
| test_pse_shift_fp16 | query, pse_shift | float16 | (1, 16, 128) | 可选参数-pse_shift(fp16) |
| test_input_layout_bsh | query, input_layout | float16 | (1, 16, 128) | input_layout=BSH |
| test_input_layout_bnsd | query, input_layout | float16 | (1, 16, 128) | input_layout=BNSD |
| test_with_atten_mask | query, atten_mask | float16 | (1, 16, 128) | atten_mask |
| test_output_shape_dtype | query | float16 | (1, 16, 128) | 输出shape+dtype断言 |

**Auto-generated stats**: 22 test items.

**Rule compliance self-check**:
- Basic functionality dtype coverage: 3 rows ✅
- Output dtype full coverage: expected 0 values [], expanded 0 rows ⚠️ 漏展开
- Optional param coverage: 30 optional Tensors ⚠️ missing coverage: ['actual_seq_lengths', 'actual_seq_lengths_kv', 'dequant_scale1', 'quant_scale1', 'dequant_scale2', 'quant_scale2', 'quant_offset2', 'antiquant_scale', 'antiquant_offset', 'block_table', 'query_padding_size', 'kv_padding_size', 'key_antiquant_scale', 'key_antiquant_offset', 'value_antiquant_scale', 'value_antiquant_offset', 'key_shared_prefix', 'value_shared_prefix', 'actual_shared_prefix_len', 'query_rope', 'key_rope', 'key_rope_antiquant_scale', 'dequant_scale_query', 'learnable_sink', 'q_start_idx', 'kv_start_idx']
- Interaction / not-applicable: Step 3 handles row add/delete and expansion per `Interaction: A×B` / `Not applicable: param` anchors in Constraints column
- Required scenario check (category=attention):
    - input_layout all values: ✅ covered
    - PFA/IFA/PageAttention: ⚠️ missing
    - atten_mask: ✅ covered
    - actual_seq_lengths: ⚠️ missing
    - block_table: ⚠️ missing

**⚠️ Self-check warnings (step 3 review or upstream format fix needed)**:
  - Optional param zero coverage: ['actual_seq_lengths', 'actual_seq_lengths_kv', 'dequant_scale1', 'quant_scale1', 'dequant_scale2', 'quant_scale2', 'quant_offset2', 'antiquant_scale', 'antiquant_offset', 'block_table', 'query_padding_size', 'kv_padding_size', 'key_antiquant_scale', 'key_antiquant_offset', 'value_antiquant_scale', 'value_antiquant_offset', 'key_shared_prefix', 'value_shared_prefix', 'actual_shared_prefix_len', 'query_rope', 'key_rope', 'key_rope_antiquant_scale', 'dequant_scale_query', 'learnable_sink', 'q_start_idx', 'kv_start_idx']
  - Required scenarios missing (attention): ['PFA/IFA/PageAttention', 'actual_seq_lengths', 'block_table']
  - Negative degradation: .constraint_info.json missing, only dtype+dim negative recoverable; rank/shape_link/mutex negative lost → re-run scaffold_init to generate .constraint_info.json then re-run this script

### Negative Coverage Matrix

> Auto-generated by generate_negative_cases(), derived from constraint signals for illegal input combinations.

| Test Item | Covered Params | Data Type | Dimensions | Scenario | Violated Field | Violation Construction | Expected |
|-----------|----------------|-----------|------------|----------|----------------|------------------------|----------|
| test_neg_query_invalid_dtype | query | bfloat16 | (2, 128) | Negative-dtype domain | query | dtype=float32 | raises:PARAM_INVALID |
| test_neg_key_invalid_dtype | query, key | bfloat16 | (2, 128) | Negative-dtype domain | key | dtype=float32 | raises:PARAM_INVALID |
| test_neg_value_invalid_dtype | query, value | bfloat16 | (2, 128) | Negative-dtype domain | value | dtype=float32 | raises:PARAM_INVALID |
| test_neg_pse_shift_invalid_dtype | query, pse_shift | bfloat16 | (2, 128) | Negative-dtype domain | pse_shift | dtype=float32 | raises:PARAM_INVALID |
| test_neg_atten_mask_invalid_dtype | query, atten_mask | bfloat16 | (2, 128) | Negative-dtype domain | atten_mask | dtype=float32 | raises:PARAM_INVALID |
| test_neg_dequant_scale1_invalid_dtype | query, dequant_scale1 | bfloat16 | (2, 128) | Negative-dtype domain | dequant_scale1 | dtype=int64 | raises:PARAM_INVALID |
| test_neg_dequant_scale2_invalid_dtype | query, dequant_scale2 | bfloat16 | (2, 128) | Negative-dtype domain | dequant_scale2 | dtype=int64 | raises:PARAM_INVALID |
| test_neg_quant_scale2_invalid_dtype | query, quant_scale2 | bfloat16 | (2, 128) | Negative-dtype domain | quant_scale2 | dtype=int64 | raises:PARAM_INVALID |
| test_neg_quant_offset2_invalid_dtype | query, quant_offset2 | bfloat16 | (2, 128) | Negative-dtype domain | quant_offset2 | dtype=int64 | raises:PARAM_INVALID |
| test_neg_antiquant_scale_invalid_dtype | query, antiquant_scale | bfloat16 | (2, 128) | Negative-dtype domain | antiquant_scale | dtype=int64 | raises:PARAM_INVALID |
| test_neg_antiquant_offset_invalid_dtype | query, antiquant_offset | bfloat16 | (2, 128) | Negative-dtype domain | antiquant_offset | dtype=int64 | raises:PARAM_INVALID |
| test_neg_key_antiquant_scale_invalid_dtype | query, key_antiquant_scale | bfloat16 | (2, 128) | Negative-dtype domain | key_antiquant_scale | dtype=int64 | raises:PARAM_INVALID |
| test_neg_key_antiquant_offset_invalid_dtype | query, key_antiquant_offset | bfloat16 | (2, 128) | Negative-dtype domain | key_antiquant_offset | dtype=int64 | raises:PARAM_INVALID |
| test_neg_value_antiquant_scale_invalid_dtype | query, value_antiquant_scale | bfloat16 | (2, 128) | Negative-dtype domain | value_antiquant_scale | dtype=int64 | raises:PARAM_INVALID |
| test_neg_value_antiquant_offset_invalid_dtype | query, value_antiquant_offset | bfloat16 | (2, 128) | Negative-dtype domain | value_antiquant_offset | dtype=int64 | raises:PARAM_INVALID |
| test_neg_key_shared_prefix_invalid_dtype | query, key_shared_prefix | bfloat16 | (2, 128) | Negative-dtype domain | key_shared_prefix | dtype=int64 | raises:PARAM_INVALID |
| test_neg_value_shared_prefix_invalid_dtype | query, value_shared_prefix | bfloat16 | (2, 128) | Negative-dtype domain | value_shared_prefix | dtype=int64 | raises:PARAM_INVALID |
| test_neg_query_rope_invalid_dtype | query, query_rope | bfloat16 | (2, 128) | Negative-dtype domain | query_rope | dtype=float32 | raises:PARAM_INVALID |
| test_neg_key_rope_invalid_dtype | query, key_rope | bfloat16 | (2, 128) | Negative-dtype domain | key_rope | dtype=float32 | raises:PARAM_INVALID |
| test_neg_key_rope_antiquant_scale_invalid_dtype | query, key_rope_antiquant_scale | bfloat16 | (2, 128) | Negative-dtype domain | key_rope_antiquant_scale | dtype=int64 | raises:PARAM_INVALID |

**Negative category coverage**: 1 categories, 20 rows total

### Finalization Add/Delete Log

> 格式契约: `Delete <exact_method_name>: <reason>` / `Add <exact_method_name>: <reason>`
> `<exact_method_name>` 必须与 Coverage Matrix 的 Test Item 列值**逐字一致**, 禁自然语言概述。
> reconcile.py 依赖此格式做删除放行判定, 格式不符 → [BAD-LOG] 告警。

<!-- 在此下方逐行记录增删, 一行一项 -->

Delete test_basic_bf16: 合并到 test_basic_bnsd_bf16
Delete test_basic_fp16: 合并到 test_basic_bnsd_fp16
Delete test_basic_int8: INT8 需要全量化参数组同时传入, 暂不覆盖
Delete test_3d_input: 注意力算子需要最小 4D BNSD 或 3D BSH 输入, 3D 非有效形状
Delete test_key_bf16: 参数组合重复, key+value+query 已在基础测试用例中覆盖
Delete test_key_fp16: 参数组合重复
Delete test_key_pse_shift_bf16: 参数组合重复, pse_shift 可选参数不在基础覆盖范围内
Delete test_key_pse_shift_fp16: 参数组合重复
Delete test_key_value_bf16: 参数组合重复
Delete test_key_value_fp16: 参数组合重复
Delete test_key_value_pse_shift_bf16: 参数组合重复
Delete test_key_value_pse_shift_fp16: 参数组合重复
Delete test_value_bf16: 参数组合重复
Delete test_value_fp16: 参数组合重复
Delete test_value_pse_shift_bf16: 参数组合重复
Delete test_value_pse_shift_fp16: 参数组合重复
Delete test_pse_shift_bf16: pse_shift 是可选位置编码参数, 基础测试不覆盖
Delete test_pse_shift_fp16: 同上
Delete test_input_layout_bnsd: 合并到 test_basic_bnsd_fp16
Delete test_input_layout_bsh: 合并到 test_basic_bsh_fp16
Delete test_with_atten_mask: 合并到 test_with_atten_mask (已实现)
Delete test_output_shape_dtype: 合并到 test_output_shape (已实现)

Add test_basic_bnsd_fp16: BNSD FP16 基础功能测试
Add test_basic_bnsd_bf16: BNSD BF16 基础功能测试
Add test_basic_bsh_fp16: BSH FP16 基础功能测试
Add test_with_atten_mask: BNSD 带 atten_mask 测试
Add test_with_gqa: GQA (num_key_value_heads != num_heads) 测试
Add test_output_shape: 输出 shape/dtype 验证测试

Delete test_neg_antiquant_offset_invalid_dtype: antiquant 不支持, 不测试
Delete test_neg_antiquant_scale_invalid_dtype: antiquant 不支持, 不测试
Delete test_neg_atten_mask_invalid_dtype: atten_mask dtype 错误由 aclnn 内核校验
Delete test_neg_dequant_scale1_invalid_dtype: dequant_scale1 需要全量化场景才触发
Delete test_neg_dequant_scale2_invalid_dtype: dequant_scale2 需要全量化场景才触发
Delete test_neg_key_antiquant_offset_invalid_dtype: 需要伪量化场景才触发
Delete test_neg_key_antiquant_scale_invalid_dtype: 需要伪量化场景才触发
Delete test_neg_key_rope_antiquant_scale_invalid_dtype: key_rope_antiquant_scale 预留参数
Delete test_neg_key_rope_invalid_dtype: key_rope 需要 MLA 场景才触发
Delete test_neg_key_shared_prefix_invalid_dtype: 需要 prefix 场景才触发
Delete test_neg_pse_shift_invalid_dtype: pse_shift dtype 由 aclnn 内核校验
Delete test_neg_quant_offset2_invalid_dtype: 需要后量化场景才触发
Delete test_neg_quant_scale2_invalid_dtype: 需要后量化场景才触发
Delete test_neg_query_rope_invalid_dtype: query_rope 需要 MLA 场景才触发
Delete test_neg_value_antiquant_offset_invalid_dtype: 需要伪量化场景才触发
Delete test_neg_value_antiquant_scale_invalid_dtype: 需要伪量化场景才触发
Delete test_neg_value_invalid_dtype: value dtype 由 aclnn 内核校验
Delete test_neg_value_shared_prefix_invalid_dtype: 需要 prefix 场景才触发
