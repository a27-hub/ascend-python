---
name: op-test
description: Generate and execute UT test cases for torch_npu operators. Use when the user asks to write, create, or generate unit tests for a torch_npu operator (e.g. "write tests for npu_xxx", "generate UT for this operator"). Automatically determines the correct output directory, generates test file with comprehensive test coverage, and executes the tests to verify correctness. Skips execution if NPU hardware is unavailable.
---

# torch_npu Operator UT Test Generation

Generate unit tests for torch_npu operators, selecting the correct test directory and test pattern based on operator type.

## Test Directory Guide

| Directory | Scope |
|-----------|-------|
| `test/test_custom_ops/` | New custom operators (`npu_xxx`, operators integrated via op-adaptation) |
| `test/test_base_ops/` | NPU implementations of PyTorch native operators (`torch.xxx`), and a few legacy `npu_xxx` operators |
| `test/core_tests/` | Framework-level tests (FakeTensor, compatibility, etc.) — not typically added manually |

**Decision rule**: New operators integrated via op-adaptation → `test/test_custom_ops/`; NPU implementations of PyTorch native operators → `test/test_base_ops/`.

## Workflow (Within op-adaptation Step 7)

**Note**: This skill is invoked during **op-adaptation Step 7: Test**. The steps below are sub-steps of that phase.

1. **Determine target directory**: Based on operator type, select the correct test directory
2. **Select test pattern**: Choose the appropriate pattern (see Pattern Selection table below), read detailed guide from `references/`
3. **Parameter Coverage Analysis** (CRITICAL - before generating tests):
   - Extract all OPTIONAL Input parameters from _def.cpp
   - Extract constraint scenarios from .md (quantization, PageAttention, etc.)
   - Generate coverage matrix to ensure each parameter is covered in at least one test case
   - Update CHECKLIST.md "Parameter Coverage Checklist"
4. **Generate test file**: Create comprehensive test file at `test/{target_dir}/test_npu_{operator_name}.py` following `references/test-template.md` and `references/test-coverage-guide.md`
5. **Check NPU availability**: Verify if NPU hardware is available in the current environment
6. **Execute tests**: Run the generated test file to verify correctness:
   - If NPU is available: Execute `python test/{target_dir}/test_npu_{operator_name}.py`
   - If NPU is unavailable: Skip execution and inform the user
7. **Report results**:
   - On success: Report the number of tests passed
   - On failure: Show error messages and suggest potential fixes
   - On skip: Inform user that test file was generated but execution was skipped
8. **Generate test report** (for integration with op-adaptation Step 9):
   - Create a structured test report summarizing execution results
   - Include: NPU availability status, number of tests passed/failed/skipped, error messages if any

### Parameter Coverage Analysis

Before generating tests, perform coverage analysis:

1. **Extract OPTIONAL Inputs** from _def.cpp:
   ```python
   optional_inputs = ['bias', 'scale', 'offset', 'group_index', 'per_token_scale', ...]
   ```

2. **Extract Scenarios** from .md:
   - Quantization scenarios
   - PageAttention scenarios
   - MoE scenarios
   - etc.

3. **Generate Coverage Matrix** (for analysis only - temporary):

   | Parameter/Scenario | Basic | Quantization | PageAttention | MoE |
   |-------------------|-------|--------------|---------------|-----|
   | bias              | ✅    | ✅           | ✅            | -   |
   | scale             | -     | ✅           | -             | -   |
   | group_index       | -     | -            | -             | ✅  |
   | per_token_scale   | -     | ✅           | -             | -   |

4. **Ensure Coverage**: Each parameter must appear in at least one test case

5. **Update CHECKLIST.md**: Transfer results to "Parameter Coverage Checklist" (permanent record):

   | # | Parameter Name | Scenario | Test Case Name | Status |
   |---|----------------|----------|----------------|--------|
   | 1 | bias           | Basic    | test_basic     | ✅     |
   | 2 | scale          | Quant    | test_quant     | ✅     |

**Note**: The Coverage Matrix (step 3) is for analysis only. The Parameter Coverage Checklist (step 5) is the permanent record in CHECKLIST.md.

## Pattern Selection

| Operator Type | Pattern | Target Directory | Reference Guide |
|---------------|---------|------------------|-----------------|
| New custom NPU operator (`npu_xxx`) with clear mathematical definition | supported/custom dual-path comparison | `test_custom_ops/` | `references/pattern-1-custom-ops.md` |
| NPU implementation of PyTorch native operator | cpu/npu dual-path comparison | `test_base_ops/` | `references/pattern-2-native-ops.md` |

## Test Report Format

When invoked by op-adaptation skill, generate a structured test report:

```markdown
**NPU Availability**: {Available/Unavailable}
**Test Execution Status**: {Executed/Skipped}
**Test Results**:
- Passed: {N}
- Failed: {N}
- Skipped: {N}

{If failed, list error messages}
```

This report will be included in the op-adaptation integration summary document.
