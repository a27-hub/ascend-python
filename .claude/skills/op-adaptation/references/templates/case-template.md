# {Case Title}

> Language: free choice (Chinese or English). Distinguish facts (this file) from evolving experience (that goes into guides).
>
> **File naming**: `{operator_name}-{scenario}.md` and place under `references/examples/{create-mode|modify-mode}/`. Use the primary operator name as the prefix so filenames are unique and grep-friendly — e.g., `npu_moe_token_permute_grad_v2-refactor.md`, not `interface-refactor.md`.

**type**: `workflow-review` | `domain-reference`
- `workflow-review` — a step-by-step record of one integration/change (Background / Solution / File Changes / Pitfalls). The default case type.
- `domain-reference` — a curated library of domain facts for an operator family (structure, shape/dtype rules, version dispatch), not a single-change replay. Use when the operator is already integrated and the value is the reusable knowledge, not the process. The Solution/File Changes sections are repurposed to "operator structure" and "domain facts".

> The create-mode vs modify-mode split is still expressed by the **directory**, not by `type`. A `domain-reference` case can live in either directory.

## Background

{What problem was being solved? Why was this change needed?}

## Solution

{How was it solved? Include key design decisions and rationale.}

## File Changes

{Detailed per-file changes. Include code snippets, diffs, or structured descriptions as the scenario warrants.}

| File | Action | Details |
|------|--------|---------|
| {file} | {add / modify / append} | {what changed} |

For each file, include **concrete before/after diffs or code blocks**. Interface changes need side-by-side old/new signatures. Logic changes need code excerpts showing what was modified. Table summaries alone are not sufficient — the case is a reference for future similar work, and code examples are the most actionable part.

## Pitfalls (optional — AI-generated summaries)

- {Specific mistake + how it was found + how it was avoided or fixed}

## A5 Platform Specific Info (M-adapt only)

> Required only for M-adapt (A5 platform adaptation) cases. Create / plain Modify cases may omit this section.

### A5 adaptation basics

| Field | Value |
|-------|-------|
| Target SoC | Ascend950 (Atlas A5 training series) |
| aclnn operator | `aclnn{OperatorName}` |
| Routing | `GetSocVersion()` (not `DO_COMPATIBILITY`) |
| Related PR | {gitcode PR link} |

### A5-specific handling

| Aspect | Description | Applies? |
|--------|-------------|----------|
| dtype cast (fp16→fp32→fp16) | needed when aclnn doesn't support the original dtype | {yes/no} |
| output transpose | aclnn output shape differs from the user-facing API | {yes/no} |
| backward unsupported | A5 aclnn has no backward | {yes/no, hard-error} |
| multi-output internal API | `_npu_xxx` returns a tuple | {yes/no} |
| aclnn new params | aclnn has params acl_op lacks (e.g. eps, aligned) | {yes/no} |
| A5-specific constraints | {e.g. boxes 2nd dim must be multiple of 1024} | {if any, describe} |

### A5 test design notes

- Platform isolation: `@SupportedDevices(['Ascend950'])` vs `@SupportedDevices(['Ascend910A', 'Ascend910B'])`
- Input size constraints: {e.g. boxes 2nd dim must be multiple of 1024}
- Precision assertion strategy: ACL path (forced fp16) vs ACLNN path (native fp32) differences
- Error cases: verify illegal inputs raise RuntimeError on A5

