# Verify Guide (C-verify)

Detailed instructions for the signature verification step. Reference this guide when executing C-verify in either create or modify mode.

> **Script**: `scripts/op_adapt.py verify --op-name <name>` → cross-checks YAML `func:` ↔ C++ signature ↔ Meta `@impl` ↔ Derivatives `name:`+formulas (param names/types/order, backward-op registration, formula-body identifiers). Exit code 0 = pass, 1 = mismatch.

---

## Run the verifier — deterministic checks + heuristic DRAFT

The script does the four-way cross-check. Most checks are deterministic; two are heuristic and the LLM must interpret them.

**Deterministic** (trust): param name-set diffs, param order, backward-op-registered check, `_dummy` suffix normalization, file finding.

**Heuristic — LLM must verify** (the script can false-alarm):

1. **Type compatibility** (`_types_compatible`) — substring matching + a fixed YAML→C++ mapping table + a loose fallback (`yaml_type` minus `?` appears in the C++ type). An "incompatible" report is a **warning** (does not fail `passed`) and may be a false alarm (e.g. `Tensor(b!)` vs `at::Tensor&`, which affects ~52 ops). Read the reported type pair and judge whether it is a real mismatch.
2. **Derivatives formula-body identifiers** — regex-extracted identifiers minus a hand-maintained keyword list; "suspicious unknown identifiers" are **warnings** and may flag method calls / types not in the removal list. Confirm they are real typos before fixing.
3. **Signature parsing fragility** — regex parsing of C++/Python can mis-parse complex signatures (`std::tuple`, templates, multi-line). A "params mismatch" may be a parse error, not a real inconsistency; spot-check the parsed signature against the source.

**Severity & `passed`**: only **hard** mismatches flip `passed` to false. `warning` (param order, intentional derivatives name differences like `fast_gelu` vs `npu_fast_gelu`, C++ file not found for gen_opapi operators) and `known` (see below) do **not** fail — so `re-run until passed: true` is always reachable for ops with legitimate warnings.

**Known-inconsistent allowlist**: `scripts/tests/known_inconsistent.json` lists operators with accepted historical mismatches. The script consults it at runtime (shared with the test suite): if the operator is listed, all its mismatches are marked `severity: "known"`, the report carries `known_inconsistent: true`, and `passed` stays true. Do not "fix" these.

## LLM action

1. Run `op_adapt.py verify --op-name <name>`.
2. For each mismatch, classify real vs heuristic false-alarm per above; ignore `known` entries.
3. Fix only real hard mismatches (in YAML / C++ / Meta / Derivatives), then re-run until `passed: true`.

5. **Constraint coverage check** (assertion-type constraints only):
   - **Scope boundary**: this step covers *assertion-type* constraints only — mutual-exclusion/coexistence rules, value-range checks, dtype-combination restrictions. These are the constraints implemented as `TORCH_CHECK` / `DO_COMPATIBILITY` in C++.
   - Constraints that are *shape/dtype derivation rules* (output shape or dtype changes by branch) are **out of scope here** — they belong to step #6 (Dimension Branch Table). Do not double-count a rule in both #5 and #6; if a rule is logged as a Dimension Branch row, #5 skips it.
   - This step is a *verification* of the C-constraints step: read the CHECKLIST "Constraint Coverage Self-Check" table (filled during C-constraints) and confirm each assertion-type row is marked done with a `TORCH_CHECK` location. If any is uncovered, return to C-constraints and fix before build.

6. **Branch reachability check** (mandatory when Dimension Branch Table exists):
   - Covers *shape/dtype derivation rules* (the complement of #5's assertion-type constraints). For each row in CHECKLIST Dimension Branch Table, verify "C++" and "Meta" columns are filled (not MISSING). A row's branch may be implemented as `TORCH_CHECK`, `if/else`, `switch`, or a helper function — any of these counts as covered.
   - For any MISSING: implement the branch before proceeding to C-build
   - Output: `Branch Reachability: {M}/{N} covered. MISSING: {list rows}. VERDICT: PASS/BLOCK`
   - BLOCK = any row missing C++ or Meta implementation

7. **Type compatibility check**:
   - Extract parameter declared types from YAML `func:` line and actual C++ types from the function signature.
   - Verify against the **canonical type mapping table** in `cpp-guide.md` § Critical Requirements (the single source for this table — do not maintain a copy here). Report type mismatches.


## Meta Shape/Dtype Logic Verification

Beyond parameter name matching, verify that the meta function's shape/dtype inference logic is consistent with the C++ implementation. Parameter name checks alone will NOT catch logic errors - and logic errors silently produce wrong results in graph mode.

### Walkthrough Example: `npu_gelu_mul`

This A2 operator (activation with last_dim halved) demonstrates the full verification flow.

**Step 1: Extract both implementations**

C++ (`GeluMulKernelNpuOpApi.cpp`):
```cpp
c10::SmallVector<int64_t, SIZE> npu_gelu_mul_out_size(const at::Tensor& input) {
    auto shape = input.sizes();
    shape[shape.size() - 1] = shape[shape.size() - 1] / 2;   // last dim halved
    return shape;
}
```

Meta (`_meta_registrations.py`):
```python
def npu_gelu_mul_meta(input_tensor, *, approximate="none"):
    output_shape = list(input_tensor.shape)
    last_dim = input_tensor.shape[-1]
    output_shape[-1] = last_dim // 2                            # last dim halved
    return torch.empty(tuple(output_shape), dtype=input_tensor.dtype, device='meta')
```

**Step 2: Compare dimension-by-dimension**

| Output # | Dim | C++ Value | Meta Value | Match? |
|----------|-----|-----------|------------|--------|
| out[0] | dim[0..-2] | `shape[0..-2]` (unchanged) | `input_tensor.shape[0..-2]` (unchanged) | OK |
| out[0] | dim[-1] | `shape[-1] / 2` | `last_dim // 2` | OK |
| out[0] | dtype | `input.options()` (inherited) | `input_tensor.dtype` | OK |

**Step 3: Sentinel check** - C++: no sentinel guards. Meta: no sentinel guards. N/A.

**Step 4: Constraint check** - C++: no TORCH_CHECK. Meta: no torch._check. N/A.

**Verdict**: All dimensions match. Proceed to C-build.

---

### Generic Verification Procedure

For any operator, follow these 4 steps:

**1. Extract C++ shape/dtype logic**

```bash
# Find the C++ implementation file
ls op_plugin/ops/opapi/*{OpName}*.cpp op_plugin/utils/KernelNpuOutputSize.cpp

# Extract shape logic
grep -nE "out_size|apply_tensor_without_format|SmallVector|\.size\(" op_plugin/ops/opapi/{OpName}KernelNpuOpApi.cpp

# Extract dtype logic
grep -nE "ScalarType|options\(\)\.dtype|convert_to_scalar_type|TensorWrapper" op_plugin/ops/opapi/{OpName}KernelNpuOpApi.cpp
```

**2. Extract meta logic**
```bash
grep -A 30 "def npu_{op_name}_meta" op_plugin/python/meta/_meta_registrations.py
```

**3. Fill the comparison table** - every output tensor gets a row for each computed dimension and dtype:

| Output # | Dim / Property | C++ Value | Meta Value | Match? |
|----------|---------------|-----------|------------|--------|
| out[0] | dim[0] | | | |
| out[0] | dim[-1] | | | |
| out[0] | dtype | | | |
| out[1] | dim[0] | | | |
| out[1] | dtype | | | |

**4. Sentinel value cross-check**

For each sentinel value, verify both sides have identical handling:
```bash
grep -nE "\-1|<= 0|nullptr|== 0" op_plugin/ops/opapi/{OpName}KernelNpuOpApi.cpp
grep -nE "\-1|<= 0|is None|== 0" op_plugin/python/meta/_meta_registrations.py | grep -A 2 -B 2 "{op_name}"
```

### Pass/Fail Criteria

| Criterion | Pass | Fail Action |
|-----------|------|-------------|
| All dim computations match | Proceed | Fix meta to match C++ formula, re-verify |
| All dtype specifications match | Proceed | Fix meta dtype, re-verify |
| All sentinel guards match | Proceed | Add missing guard to meta (or fix C++ guard), re-verify |
| Every C++ TORCH_CHECK(shape/dtype) has meta torch._check | Proceed | Add torch._check to meta |
| Output order matches YAML -> (...) | Proceed | Reorder meta return tuple |

**Gate rule**: Do NOT proceed to C-build until all criteria pass. One failure = potential graph mode silent error.

### Verification Script Skeleton

For operators with complex shape logic (>3 outputs or >5 dimensions):
```bash
#!/bin/bash
# verify_meta_{op_name}.sh - run from op-plugin root
OP_NAME="{op_name}"
CPP_FILE="op_plugin/ops/opapi/${OP_NAME}KernelNpuOpApi.cpp"
META_FILE="op_plugin/python/meta/_meta_registrations.py"

echo "=== C++ shape logic ==="
grep -nE "out_size|apply_tensor_without_format|SmallVector" "$CPP_FILE"

echo ""
echo "=== C++ dtype logic ==="
grep -nE "ScalarType|options\(\)\.dtype|TensorWrapper" "$CPP_FILE"

echo ""
echo "=== Meta registration ==="
grep -A 30 "def npu_${OP_NAME}_meta" "$META_FILE"

echo ""
echo "=== C++ sentinel guards ==="
grep -nE "\-1|<= 0|nullptr" "$CPP_FILE"

echo ""
echo "=== Meta sentinel guards ==="
grep -nE "\-1|<= 0|is None" "$META_FILE" | grep -A 2 -B 2 "${OP_NAME}"

echo ""
echo "=== C++ TORCH_CHECK ==="
grep -nE "TORCH_CHECK" "$CPP_FILE"

echo ""
echo "=== Meta torch._check ==="
grep -A 30 "def npu_${OP_NAME}_meta" "$META_FILE" | grep "torch._check"

echo ""
echo "Fill the comparison table and confirm all Match? before C-build."
```

## ACLGraph Structural Verification（条件触发：C-confirm Q2=YES）

When the operator has the ACLGraph manual-update pair (`.out` / `_get_max_workspace`), add these checks. Rules: `references/guides/yaml-guide.md` §Attention ACLGraph Interfaces.

1. **Signature structure check**：run `op_adapt.py verify --op-name <name>` — the verifier includes ACLGraph structural checks: `.out` 的 workspace 前参数段 == functional 参数段（`SymInt[]?`→`int[]?` 同位置降级报 warning 不报 BLOCK）、尾部 workspace + alias 输出、返回 alias 匹配；`_get_max_workspace` 参数段 == functional 且返回 `Tensor`；辅助接口 meta 缺失为预期。
   - **分工**：脚本只负责报 warning；由 LLM 在此处检查 CHECKLIST 是否记录了该参数的 ⚠️ codegen-type 用户确认。无确认记录的降级 → BLOCK 修正。
2. **三条 Host API 参数表核对（人工步，始终执行）**：在目标 C++ 文件中定位 functional 的 `EXEC_NPU*`、`.out` 的 `EXEC_UPDATE_NPU*`、get-max 的 `EXEC_GET_MAX_WORKSPACE_CMD`，逐项编号比对参数顺序——functional 与 `.out` 必须同 aclnn 版本、两路径参数序一致；get-max 与执行分支对应（跨版本覆盖需契约说明）。观察重点：可选 Tensor、host list、layout 字符串、输出 Tensor 的位置。
3. **预处理审查核对**：确认 `.out`/get-max 入口到 Host API 之间的限制（layout 限定、动态参数非空、dtype 转换前提等）均已转为显式 `TORCH_CHECK`，而非仅注释。
4. **Meta 豁免**：`.out` / `_get_max_workspace`（及仓内已有 `_infer_output`）无 meta 属**预期**，不报 BLOCK。

VERDICT: PASS = 结构检查通过 + 三参数表一致 + TORCH_CHECK 覆盖；任一不符 → BLOCK，修复后再进 C-build。

## Troubleshooting

(Accumulate from cases.)
