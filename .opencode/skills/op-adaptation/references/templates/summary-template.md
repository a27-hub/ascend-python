# {OperatorName} Operator Integration Summary

**Date**: {YYYY-MM-DD}
**Operator Name**: aclnn{OperatorName}
**torch_npu Function Name**: npu_{operator_name}

## 1. Operator Functionality

{Brief description of operator functionality and calculation formula}

## 2. Integration Steps

### C-parse: Parse Input Files
- Input file paths
- Parse results (parameters, constraints)

### C-confirm: Confirm with User
- Operator-specific information confirmed

### C-yaml: Generate YAML
- File path
- Added configuration content

### C-cpp: Implement C++
- File path
- Implementation highlights

### C-meta: Add Meta
- File path
- Implementation highlights

### C-verify: Signature Verification
- Verification result (Passed/Failed)
- Mismatches (if any)

### C-build: Build
- Build command
- Build results
- Issues encountered and solutions

### C-test: Tests
**{Test report collected from op-test skill}**
- NPU availability check results
- Test execution results (Passed/Failed/Skipped)
- If failed, list error messages

### C-docs: API Docs
- File path
- Documentation content summary
- Navigation files updated (`menu_Pytorch_API.md`, `torch_npu_list.md`)
- Python docstring added to `codegen/templates/_op_plugin_docs.py`

### C-baseline: Baseline
- `torch_npu_OpApi_schema_all.json` entries added
- `allowlist_for_publicAPI.json` entry added

### C-delimit: ATK Case Validation
**{One of two states}**
- Executed: ATK case dir path; case run results; delimitation conclusion
- Skipped: 无 ATK 输入，未执行 ATK 用例验证

## 3. Key Technical Points

{List key technical points in implementation, such as TensorWrapper usage, INT4 packing, optional parameter handling, etc.}

## 4. File List

**New Files**:
- {List all new files}

**Modified Files**:
- {List all modified files}

## 5. Verification Results

- ✅/❌ Build successful
- ✅/❌ Installation successful
- ✅/❌ Tests passed
- ✅/❌ API documentation generated
- ✅/❌ Compatibility registration completed
- ✅/⚠️ ATK case validation (⚠️ = skipped, no ATK input)

## 6. Notes

{List special notes, known issues, usage limitations, etc.}

## 7. Case Saving

- Case file: `references/examples/{create-mode|modify-mode}/{operator_name}-{scenario}.md` (or N/A if not a reusable reference)
- Case Index row added to the matching mode's sub-index (`references/examples/{create-mode|modify-mode|m-adapt}/README.md`): ✅/❌
- Format reference: `references/templates/case-template.md`
