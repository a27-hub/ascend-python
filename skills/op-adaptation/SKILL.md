---
name: op-adaptation
description: Integrate CANN (Ascend) operators into torch_npu backend following the standardized workflow. Use when the user asks to integrate, adapt, or add CANN/aclnn operators to torch_npu, implement NPU operators, or work with operator definition files (.md documentation and _def.cpp files). Handles the complete workflow from parsing operator specifications to building, installing, testing, and generating documentation.
---

# CANN Operator Integration

Integrate CANN (Ascend) operators into the torch_npu backend following a standardized workflow.

## Workflow

Given operator documentation (.md) and definition file (_def.cpp):

1. **Parse inputs** - Extract parameters, types, constraints from both files, then **immediately generate CHECKLIST.md** (see `references/checklist-template.md`)
2. **Ask user for operator-specific information** - Confirm version variants, SoC dispatch, .List overloads, etc.
3. **Generate YAML** - Add entry to `op_plugin/config/op_plugin_functions.yaml` (see `references/yaml-configuration-guide.md`), **update CHECKLIST.md**
4. **Generate C++** - Create `op_plugin/ops/opapi/{Name}KernelNpuOpApi.cpp` (see `references/cpp-implementation-guide.md`), **update CHECKLIST.md**
5. **Add Meta Registration** - Register in `op_plugin/python/meta/_meta_registrations.py` (see `references/meta-registration-guide.md`), **update CHECKLIST.md**
6. **Build** - Run `bash ci/build.sh --python=3.10 --pytorch=v2.7.1`, **update CHECKLIST.md**
7. **Test** - Use `op-test` skill to create and run tests, **collect test report**, **update CHECKLIST.md**
8. **Generate API docs** - Use `op-apidoc` skill to create documentation, **update CHECKLIST.md**
9. **Generate summary** - Create `notes/summary_{YYYY-MM-DD}.md` using `references/summary-template.md`, include test report from step 7

**CRITICAL**:
- After each step (2-9), mark the corresponding checkbox in CHECKLIST.md as completed
- Always generate the integration summary at the end
- Collect and include the test report from op-test skill in the summary

## Step 1: Parse Inputs

**From .md file:**
- aclnn function signature (GetWorkspaceSize parameters)
- Parameter types, input/output markers
- Dtype/shape constraints

**From _def.cpp file:**
- REQUIRED/OPTIONAL markers
- Default values from Attr(...)
- SymInt types

**Conflict resolution:**
- Optionality: Use _def.cpp ParamType
- Defaults: Use _def.cpp Attr
- Naming: Use _def.cpp snake_case
- Constraints: Use .md documentation

### Information Source Classification

**Derivable from input files:**
- Parameter names, types, optionality, defaults
- Format support, workspace management
- Shape/dtype constraints

**NOT derivable - ask user explicitly:**
- CANN version dispatch (V2/V3/V5 variants)
- SoC-specific dispatch (Ascend950 vs others)
- .List overloads
- Additional torch_npu function variants
- internal_format_opapi field
- Special data format support (mxfp4, etc.)

### Step 1 Augmentation: Completeness Self-Check

**CRITICAL**: After parameter extraction, perform the following checks to avoid missing parameters:

1. **Input Parameter Completeness Check**
   - Traverse all `this->Input("xxx")` calls in _def.cpp
   - Ensure each parameter is recorded in CHECKLIST.md
   - Report any unrecognized parameters as warnings

2. **Disabled Parameter Detection**
   - Check if parameters have "not enabled" comments
   - Mark these as OPTIONAL in the checklist

3. **Default Value Conflict Resolution**
   - When different sources have conflicting defaults:
     1. Prioritize _def.cpp Attr default values
     2. If .md conflicts with _def.cpp, record in CHECKLIST.md and prompt user for confirmation
     3. Keep default values consistent in YAML and C++

4. **Constraint Logic Extraction**
   - Extract dtype/shape derivation rules from .md constraints section:
     - Does output dtype depend on certain parameter presence? (e.g., quant_scale2 exists → output INT8)
     - Are there parameter mutual-exclusion/coexistence rules?
     - Are there scenario branching rules?
   - Record these in CHECKLIST.md "Constraint Logic" section

**Validation Report Template**:
```
Parameter Completeness Validation Report:
===================
Input Parameters:
  - x: ✅ Processed
  - weight: ✅ Processed
  - bias: ✅ Processed
  - per_token_scale: ⚠️ Defined in _def.cpp but not in YAML

Attr Parameters:
  - split_item: ✅ Processed
  - group_type: ⚠️ Default conflict (-1 vs 0)

Recommended Actions:
  1. Add per_token_scale parameter to YAML
  2. Confirm group_type default value
```

**After parsing and validation, immediately generate CHECKLIST.md** using `references/checklist-template.md`

## Step 2-5: Implementation

Refer to detailed guides in `references/`:
- **YAML**: `yaml-configuration-guide.md`
- **C++**: `cpp-implementation-guide.md`
- **Meta**: `meta-registration-guide.md`

## Step 6: Build

```bash
rm -rf build  # If the build folder already exists, delete it first
bash ci/build.sh --python=3.10 --pytorch=v2.10.0 2>&1 | tee build.log
pip install --force-reinstall build/pytorch/dist/torch_npu-*.whl
```

**If build fails:**
1. Read `build.log` to identify root cause

## Step 7: Test

Use `op-test` skill to generate and execute tests. **Collect the test report** for the integration summary.

## Step 8: Generate API Documentation

Use `op-apidoc` skill to create `docs/zh/custom_APIs/torch_npu/torch_npu-npu_{operator_name}.md`

## Step 9: Generate Integration Summary

Create `notes/summary_{YYYY-MM-DD}.md` using the template in `references/summary-template.md`.

**Requirements:**
- Use Chinese language
- Include test report from step 7
- Append to existing file if same date
- Mark CHECKLIST.md as completed

## Common Issues

**Build error: undefined reference to op_api::npu_xxx**
- YAML function signature must match C++ implementation
- Check generated signature in `build/pytorch/third_party/op-plugin/op_plugin/OpApiInterface.h`
- Common mismatches:
  - `int[]?` → `c10::OptionalIntArrayRef` (NOT `c10::optional<at::IntArrayRef>`)
  - `str` → `c10::string_view` (NOT `std::string`)

**Test fails with shape mismatch**
- INT32 output (INT4 packed): last_dim = input_last_dim / 8
- Quantization scale shape varies by quant_mode

## Best Practices

- **Generate and maintain CHECKLIST.md** throughout the workflow
- **Find similar operators** before implementing (search by category or parameter patterns)
  - ⚠️ **CRITICAL**: Only use similar operators to learn **implementation patterns** (TensorWrapper, EXEC_NPU_CMD usage)
  - ❌ **DO NOT** copy features that are not in your input files
  - ❌ **DO NOT** add parameters that the user did not request
  - ✅ **ONLY** implement what is explicitly defined in the input .md and _def.cpp files
- **Check generated signature** in OpApiInterface.h after adding YAML
- **Update CHECKLIST.md** after each step
- **Always generate integration summary** at the end

## ⚠️ CRITICAL: Scope Control

**ONLY implement what is explicitly provided in:**
1. Input files (.md documentation and _def.cpp)
2. User's explicit requests during Step 2

**DO NOT add:**
- Extra parameters not in input files
- Extra function variants not requested by user
- Features from reference implementations unless they exist in input files

**Example of correct scoping:**
- Input file has `dst_type` parameter → ✅ Implement it
- Input file has no `quant_mode` parameter → ❌ Do NOT add it
- Reference implementation has `quant_mode` → ❌ Still do NOT add it (unless in input files)
