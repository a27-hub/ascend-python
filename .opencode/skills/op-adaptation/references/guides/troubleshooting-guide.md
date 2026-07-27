# Troubleshooting Guide

Cross-cutting issues that don't belong to a single step guide. Step-specific troubleshooting lives in each guide's `## Troubleshooting` section.

## Build Issues

**undefined reference to op_api::npu_xxx**
- YAML function signature must match C++ implementation
- Check generated signature in `build/pytorch/third_party/op-plugin/op_plugin/OpApiInterface.h`
- For type mismatches, see the canonical type mapping table in `cpp-guide.md` § Critical Requirements.

## Complex Operator Pitfalls

**Parameters present in signature but features unusable**
- Symptom: output shape/dtype wrong for certain parameter combinations
- Root cause: shape calculation only covers base paths, missing branches identified in the Dimension Branch Table (some rows left `MISSING`)
- Fix: add shape/dtype branches for each MISSING row; check Feature Parameter Groups entries marked "C++ Pre-proc: Y". Re-derive the concrete formula from the operator's own `.md` / aclnn signature — do not copy a formula from another operator or a case file without verifying against source.

**Build error worked around instead of fixed**
- Symptom: diff shows edits that delete/skip functionality or reference non-existent symbols purely to make it compile
- Root cause: treating "compiles" as equivalent to "correctly implemented"
- Fix: determine whether the missing function should be implemented (then implement it) or the caller is stale/wrong (then fix the caller). Neither is a "workaround" — both are real fixes when matched to the actual root cause.
- **Not a workaround** (do not flag as wrong): version dispatch — selecting between multiple correct aclnn variants (`aclnnXxxV2` vs `aclnnXxxV3`) by CANN/chip version. This is a legitimate pattern used by many complex operators.

## Meta Issues

**Meta registration not loaded**
- Symptom: FakeTensor test raises `UnsupportedOperatorException`
- Cause: torch_npu._C fails to import → Meta module never loaded
- Workaround: manually register via `Library('npu', 'IMPL', 'Meta')` + `@impl(m, 'op_name')` for testing

## Graph Mode Issues

### torch.compile silently falls back to eager (no error, no graph)
- **Cause**: `ci/build.sh` default flag `--disable_torchair` excluded torchair from build.
- **Fix**: Remove `--disable_torchair` from `ci/build.sh`, rebuild. Verify: `[83%] Built target _torchair` in build log.

### torch.compile fails but eager mode works
- **Cause**: Meta registration is missing or has incorrect shape/dtype inference.
- **Check**: does `_meta_registrations.py` have `@impl(m, "npu_xxx")` for this operator?
- **Check**: does meta shape derivation match C++ `_out_size` / `apply_tensor_without_format`?
- **Fix**: see `verify-guide.md` Meta Shape/Dtype Logic Verification for the cross-check procedure.

### torch.compile produces wrong output (eager correct)
- **Cause**: meta shape/dtype inference is inconsistent with C++ implementation.
- **Symptom**: downstream operators silently reshape or reinterpret tensors with wrong shapes.
- **Fix**: cross-check meta shape derivation against C++ `_out_size`. See `verify-guide.md`.

### Graph mode compiles but runs slower than eager
- **Cause**: FakeTensor tracing overhead on first invocation, or Dynamo graph breaks causing recompilation.
- **Check**: `TORCH_LOGS="graph_breaks"` to identify where Dynamo falls back to eager. Re-run to amortize compilation.
- **Fix**: Reduce graph breaks by ensuring all operators in the call chain have meta registrations.

### `backend="npugraph_ex"` vs `backend="npu"` - which to use?
- `npugraph_ex`: pre-initialized backend, faster first invocation, always available. **Recommended** for operator testing.
- `npu`: lazy torchair load, slower first call, requires `--disable_torchair` removed from build.
- For operator meta verification: always use `npugraph_ex` to avoid lazy-loading overhead. Both go through the same meta functions.

### Multi-card graph mode: communication operator shapes
- **Symptom**: `torch.compile` with `npu_mm_reduce_scatter_base` or `npu_all_gather_base_mm` produces wrong shapes in multi-card training.
- **Cause**: Dynamo traces on a single device with `world_size=1`. The meta function must handle `world_size <= 0` -> default to 1 for tracing to succeed.
- **Fix**: Meta must use defensive `if world_size <= 0: world_size = 1` guard. Use `math.floor(self.size(0) / world_size)` for division. See `npu_mm_reduce_scatter_base_meta` for the reference pattern.

## ACLGraph Auxiliary Interface False Positives

`verify` 已内置 ACLGraph 处理：辅助接口（`.out` / `_get_max_workspace` / `_infer_output`）meta 缺失为**预期**，结构检查由 verify 承担（见 `verify-guide.md` §ACLGraph Structural Verification）。若仍出现误报，先查接口是否被识别为辅助接口（命名 `_npu_*_get_max_workspace` / `.out` overload）。

- **Rule**: do not add meta or rename functions to silence verify reports on these interfaces.
- **Graph update 排障顺序**：先分离 eager 三路径（functional / `.out(workspace=None)` / `.out(workspace=max_ws)`，eager 失败不进 graph 排障）→ 确认新旧 host 参数 eager 结果有区分度且 replay 对齐新参数 → 查 workspace/输出 `data_ptr()` 地址稳定性与 host 参数生命周期（CopyTypesV2）。
- 历史记录（2026-07-21，脚本修复前）：IFA 4 条目 verify 全部 `passed: false`，均为脚本局限误报。
