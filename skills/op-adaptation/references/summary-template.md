# {OperatorName} Operator Integration Summary

**Date**: {YYYY-MM-DD}
**Operator Name**: aclnn{OperatorName}
**torch_npu Function Name**: npu_{operator_name}

## 1. Operator Functionality

{Brief description of operator functionality and calculation formula}

## 2. Integration Steps

### Step 1: Parse Input Files
- Input file paths
- Parse results (parameters, constraints)

### Step 2: Generate YAML Configuration
- File path
- Added configuration content

### Step 3: Implement C++ Code
- File path
- Implementation highlights

### Step 4: Add Meta Registration
- File path
- Implementation highlights

### Step 5: Build Verification
- Build command
- Build results
- Issues encountered and solutions

### Step 6: Generate and Execute Tests
**{Test report collected from op-test skill}**
- NPU availability check results
- Test execution results (Passed/Failed/Skipped)
- If failed, list error messages

### Step 7: Generate API Documentation
- File path
- Documentation content summary

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

## 6. Notes

{List special notes, known issues, usage limitations, etc.}
