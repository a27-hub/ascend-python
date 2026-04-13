# {OperatorName} Integration Checklist

## Parse Results
- aclnn function name: aclnn{OperatorName}
- torch_npu function name: npu_{operator_name}
- Input parameters: [list all input parameters and types]
- Output parameters: [list all output parameters and types]
- Key constraints: [list important constraints]

## Information Required from User (not derivable from input files)
- [ ] Are there aclnn version variants (V2/V3/V5)? If so, what are the trigger conditions for each version?
- [ ] Is SoC version dispatch needed (e.g., Ascend950 vs others)?
- [ ] Is a .List overload needed (a Tensor parameter that also supports int[] type)?
- [ ] Are there additional torch_npu function variants (e.g., _asymmetric, _v0)?
- [ ] Is the internal_format_opapi field needed?
- [ ] Is there special data format support (mxfp4, etc.)?

## Constraint Logic Checklist (Cross-Step Verification)

Extracted from .md constraints section (must be synchronized across all implementation steps):

| # | Constraint Rule | .md Source | C++ (Step 4) | Meta (Step 5) | Test (Step 7) | Doc (Step 8) |
|---|-----------------|------------|--------------|---------------|---------------|--------------|
| 1 | {e.g., quant_scale2 exists → output INT8} | {line num} | ☐ | ☐ | ☐ | ☐ |
| 2 | {e.g., query INT8 without quant_scale2 → query_rope dtype} | {line num} | ☐ | ☐ | ☐ | ☐ |

**Instructions**: Mark each checkbox after completing the corresponding step. Ensure constraint rules are consistently implemented across all layers.

## Parameter Coverage Checklist

All OPTIONAL Input parameters from _def.cpp must be covered in at least one test case:

| # | Parameter Name | Scenario | Test Case Name | Status |
|---|----------------|----------|----------------|--------|
| 1 | {param name} | {scenario} | {test_xxx} | ☐ |
| 2 | {param name} | {scenario} | {test_xxx} | ☐ |

## Table Migration Checklist

Constraint information in table format from .md must be fully migrated to API documentation:

| # | Table Content | .md Line | Doc Location | Status |
|---|---------------|----------|--------------|--------|
| 1 | {e.g., sparse_mode table} | {line num} | Constraints section | ☐ |
| 2 | {e.g., quant_mode table} | {line num} | Constraints section | ☐ |

## Progress
- [ ] Step 1: Parse input files
- [ ] Step 2: Generate YAML configuration
- [ ] Step 3: Implement C++ code
- [ ] Step 4: Add Meta registration
- [ ] Step 5: Build verification
- [ ] Step 6: Generate and execute tests
- [ ] Step 7: Generate API documentation
- [ ] Step 8: Generate integration summary
