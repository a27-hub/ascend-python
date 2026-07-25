---
name: op-apidoc
description: Generate Chinese API reference documentation for torch_npu custom operators, following the standard structure used in docs/zh/custom_APIs/torch_npu/. Use when the user asks to write or generate API docs for a torch_npu operator. Covers product support tables, function signatures, parameter descriptions, return values, constraints, and call examples (single-op and graph mode).
---

# torch_npu API Documentation Generation

## Workflow

1. Read `references/template.md` for the complete document structure template
2. Collect operator information:
   - **Via op-adaptation**: Read CHECKLIST.md Parameter Information Table
   - **Standalone**: Ask user for operator name, parameter list, constraint descriptions, and YAML `func:` signature
3. Fill in each section based on the collected information. See `references/writing-standards.md` for parameter formatting, constraint extraction, and call example rules.
4. Output to `docs/zh/custom_APIs/torch_npu/torch_npu-npu_{operator_name}.md`
5. Update navigation files: `menu_Pytorch_API.md` and `torch_npu_list.md`
6. Update Python docstrings: Append `_add_torch_npu_docstr("npu_{operator_name}", """...""")` to `codegen/templates/_op_plugin_docs.py`. See `references/docstring-template.md`.

## Document Structure

The generated documentation follows this structure. See `references/template.md` for the complete template.

## Documentation Quality Checklist

After generating documentation, verify:
- [ ] Function prototype matches actual implementation (CHECKLIST.md YAML func line)
- [ ] All parameters have descriptions (including _def.cpp-only parameters, confirmed by user)
- [ ] All sections filled (no empty sections or template placeholder text)
- [ ] Parameter keys match between `## Function Prototype` and `## Parameter Description`
- [ ] **Parameter completeness**: All parameters from CHECKLIST.md Parameter Information Table appear in the documentation
- [ ] **Docstring-Markdown sync**: Parameter list in `_op_plugin_docs.py` docstring matches the Markdown document (names, types, defaults). Inconsistencies must be resolved.
- [ ] Constraint tables fully migrated from .md input (not degraded to text)
- [ ] Product support table matches `op_api:` version range
- [ ] Call examples: single-op uses `>>>` format, graph mode uses `torch.compile`
- [ ] Report verification results to CHECKLIST.md
