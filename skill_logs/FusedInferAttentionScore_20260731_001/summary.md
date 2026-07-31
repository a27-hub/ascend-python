# FusedInferAttentionScore V5 Operator Integration Summary

**Date**: 2026-07-31
**Operator Name**: aclnnFusedInferAttentionScoreV5
**torch_npu Function Name**: npu_fused_infer_attention_score
**Mode**: CREATE

## 1. Operator Functionality

适配 decode & prefill 场景的 FlashAttention 算子 (V5)，支持 GQA / Prefill MLA / Decode MLA 三种计算场景。相比于 V4 版本新增 qStartIdxOptional、kvStartIdxOptional、pseType 参数。

- 12 个特性参数组：公共、PSE、Mask、ActualSeqLen、伪量化&全量化、PostQuant、PagedAttention、左padding、公共前缀、Rope、LearnableSink、SoftmaxLSE
- 产品支持：仅 Ascend 950PR/Ascend 950DT
- 258 条约束

## 2. Integration Steps

### C-parse
- 输入: `aclnnFusedInferAttentionScoreV5.md` + `fused_infer_attention_score_def.cpp`
- 参数: 47 (31 可选 tensor 输入 + 2 tensor 输出 + 16 scalar attrs)
- 约束: 258 extracted (187 likely)
- 8 冲突（均为命名约定差异的假冲突）

### C-confirm
- Q1-Q7: All NO
- Q8: SymInt — actual_seq_lengths, actual_seq_lengths_kv
- Q9a: torch.compile = YES → C-meta, torchair 编译
- Q9b: ACLGraph = YES → .out + _get_max_workspace

### C-yaml
- 4 entries: `.default` (custom+symint) + `.out` (custom) + `_get_max_workspace` (custom)
- `op_api: [v2.7, newest]`, `exposed:` on `.default`, `device_check: NoCheck`

### C-derivatives
- Skipped (no autograd)

### C-cpp
- `FusedInferAttentionScoreKernelNpuOpApi.cpp` — 4 functions
- Key: TensorList转换, SymInt→int, uppercase layout, workspace management

### C-constraints
- Noted (258 constraints validated by aclnn kernel)

### C-meta
- `_meta_registrations.py` — `npu_fused_infer_attention_score_meta()`
- Layout-aware shape inference for softmax_lse

### C-verify
- Manual cross-verification (automated parser can't handle 47-param operator)

### C-build
- **SUCCESS**: full build, 0 errors, wheel `torch_npu-2.10.0.post1`
- Fixes: lvalue binding for &workspace_size/&executor, named stream variable, Graph.cpp stub

### C-test
- SKIPPED (no NPU HW)

### C-delimit
- SKIPPED: 无 ATK 输入，未执行 ATK 用例验证

## 3. Files Changed

| File | Action |
|------|--------|
| `op_plugin/config/op_plugin_functions.yaml` | Added 4 YAML entries |
| `op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp` | **New** — C++ implementation |
| `op_plugin/python/meta/_meta_registrations.py` | Added meta registration |
| `ci/build.sh` | Removed `--disable_torchair` |

## 4. Known Issues

1. **Graph.cpp stub**: `_npu_fused_infer_attention_score_out_graph` throws runtime error — needs torch_npu team update
2. **No SoC gate**: A5-only per .md but no runtime check added
3. **Compatibility baseline deferred**: schema JSON + allowlist entries pending

## 5. Verification Results

- ✅ Build successful
- ⏭️ UT tests skipped (no NPU HW)
- ⚠️ ATK case validation skipped (无 ATK 输入)
