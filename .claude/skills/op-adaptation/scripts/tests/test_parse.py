"""Golden tests for op_adapt.py parse command.

Uses test fixtures that match real CANN file formats:
  - _def.cpp: OpDef class with chained method calls
      this->Input("smooth_scales").ParamType(OPTIONAL).DataType({...})
      this->Attr("activate_left").AttrType(OPTIONAL).Bool(false)
      .Version(SWI_GLU_QUANT_VERSION_TWO)
  - .md: camelCase params with suffixes, HTML tables, GetWorkspaceSize signature

Run: pytest tests/test_parse.py -v
"""

import json
import os
import sys
from pathlib import Path

# Ensure scripts/ is importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.parser import run_parse, normalize_name, parse_def_cpp, parse_md, detect_pitfalls


# ── Test fixtures matching real CANN format ──────────────────────────

SWI_GLU_QUANT_MD = """\
# aclnnSwiGluQuant

## 功能介绍

SwiGlu+量化融合算子。

## 接口定义

aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *smoothScalesOptional,
    const aclTensor *offsetsOptional,
    const aclTensor *groupIndexOptional,
    bool activateLeft,
    char *quantMode,
    aclTensor *yOut,
    aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)

## 约束与限制

<table>
<tr><th>参数</th><th>约束</th></tr>
<tr><td>x</td><td>只支持2D-3D输入，dtype为FLOAT或FLOAT16</td></tr>
<tr><td>quantMode</td><td>必须为"dynamic"或"static"</td></tr>
<tr><td>smoothScales</td><td>必须与x的shape一致</td></tr>
</table>

x的最后一维必须能被2整除。
"""

SWI_GLU_QUANT_DEF = """\
#include "ge_opereg.h"

class SwiGluQuantOp : public OpDef {
    SwiGluQuantOp(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16});
        this->Input("smooth_scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT});
        this->Input("offsets")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT});
        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8});
        this->Output("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT});
        this->Attr("activate_left")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("quant_mode")
            .AttrType(OPTIONAL)
            .String("dynamic");
        this->Attr("dst_type")
            .AttrType(OPTIONAL)
            .DataType(DT_INT8);
        .Version(SWI_GLU_QUANT_VERSION_TWO);
    }

    void GetKirinCoreConfig() {
        this->Input("x").ParamType(REQUIRED);
        this->Output("y").ParamType(REQUIRED);
    }
};
"""

DYNAMIC_QUANT_MD = """\
# aclnnDynamicQuant

## 接口定义

aclnnStatus aclnnDynamicQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *smoothScalesOptional,
    const aclTensor *groupIndexOptional,
    int64_t dstType,
    aclTensor *yOut,
    aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)

## 约束与限制

| 参数 | 约束 |
|------|------|
| x | 只支持2D-3D输入 |
| dstType | 必须为DT_INT8或DT_INT4x2 |
"""

DYNAMIC_QUANT_DEF = """\
class DynamicQuantOp : public OpDef {
    DynamicQuantOp(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT});
        this->Input("smooth_scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT});
        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32});
        this->Output("y")
            .DataType({ge::DT_INT8});
        this->Output("scale")
            .DataType({ge::DT_FLOAT});
        this->Attr("dst_type")
            .AttrType(OPTIONAL)
            .Int(2);
        this->Attr("quant_mode")
            .AttrType(OPTIONAL)
            .String("pertoken");
        .Version(V2);
    }
};
"""


# ── Name normalization tests ────────────────────────────────────────

def test_normalize_camelcase_to_snake():
    assert normalize_name("smoothScales") == "smooth_scales"
    assert normalize_name("groupIndex") == "group_index"
    assert normalize_name("activateLeft") == "activate_left"


def test_normalize_strip_suffix():
    assert normalize_name("smoothScalesOptional") == "smooth_scales"
    assert normalize_name("yOut") == "y"
    assert normalize_name("scaleOut") == "scale"
    assert normalize_name("groupIndexOptional") == "group_index"


def test_normalize_already_snake():
    assert normalize_name("smooth_scales") == "smooth_scales"
    assert normalize_name("x") == "x"


# ── _def.cpp parser tests (OpDef chained format) ───────────────────

def test_parse_def_inputs_opdef_format():
    """Verify OpDef chained Input("name").ParamType() is parsed."""
    result = parse_def_cpp(SWI_GLU_QUANT_DEF)
    input_names = [i["name"] for i in result["inputs"]]
    assert "x" in input_names
    assert "smooth_scales" in input_names
    assert "offsets" in input_names
    assert "group_index" in input_names


def test_parse_def_input_optionality():
    """Verify .ParamType(OPTIONAL) is extracted for inputs."""
    result = parse_def_cpp(SWI_GLU_QUANT_DEF)
    by_name = {i["name"]: i for i in result["inputs"]}
    assert by_name["x"]["optionality"] == "REQUIRED"
    assert by_name["smooth_scales"]["optionality"] == "OPTIONAL"
    assert by_name["offsets"]["optionality"] == "OPTIONAL"
    assert by_name["group_index"]["optionality"] == "OPTIONAL"


def test_parse_def_outputs_no_duplicates():
    """Verify Kirin config re-declaration doesn't cause duplicates."""
    result = parse_def_cpp(SWI_GLU_QUANT_DEF)
    output_names = [o["name"] for o in result["outputs"]]
    assert output_names.count("y") == 1
    assert output_names.count("scale") == 1


def test_parse_def_attrs_chained_defaults():
    """Verify .Bool(false), .String("dynamic"), .Int(2) extract defaults."""
    result = parse_def_cpp(SWI_GLU_QUANT_DEF)
    by_name = {a["name"]: a for a in result["attrs"]}
    assert by_name["activate_left"]["default_raw"] == "false"
    assert by_name["activate_left"]["type_hint"] == "Bool"
    assert by_name["quant_mode"]["default_raw"] == "dynamic"
    assert by_name["quant_mode"]["type_hint"] == "String"
    assert by_name["dst_type"]["default_raw"] == "DT_INT8"
    assert by_name["dst_type"]["type_hint"] == "DataType"


def test_parse_def_version_named_constant():
    """Verify .Version(SWI_GLU_QUANT_VERSION_TWO) is detected."""
    result = parse_def_cpp(SWI_GLU_QUANT_DEF)
    assert len(result["versions"]) > 0
    assert "SWI_GLU_QUANT_VERSION_TWO" in result["versions"]


def test_parse_def_version_literal():
    """Verify .Version(V2) literal is also detected."""
    result = parse_def_cpp(DYNAMIC_QUANT_DEF)
    assert "V2" in result["versions"]


def test_parse_def_kirin_scoped():
    """Verify GetKirinCoreConfig body is excluded from input/output parsing."""
    result = parse_def_cpp(SWI_GLU_QUANT_DEF)
    # Should have exactly 4 inputs (not 8 from Kirin re-declaration)
    assert len(result["inputs"]) == 4
    assert len(result["outputs"]) == 2


# ── .md parser tests ────────────────────────────────────────────────

def test_parse_md_params_camelcase():
    """Verify camelCase params are extracted and normalized."""
    result = parse_md(SWI_GLU_QUANT_MD)
    param_names = {p["name"] for p in result["params"]}
    assert "smoothScalesOptional" in param_names
    assert "groupIndexOptional" in param_names
    assert "yOut" in param_names


def test_parse_md_params_normalized():
    """Verify normalized names match _def.cpp snake_case."""
    result = parse_md(SWI_GLU_QUANT_MD)
    by_normalized = {p["normalized_name"]: p for p in result["params"] if p.get("normalized_name")}
    assert "smooth_scales" in by_normalized
    assert "group_index" in by_normalized
    assert "y" in by_normalized
    assert "scale" in by_normalized


def test_parse_md_workspace_filtered():
    """Verify workspaceSize and executor are filtered out."""
    result = parse_md(SWI_GLU_QUANT_MD)
    param_names = [p["name"] for p in result["params"] if p.get("name")]
    assert "workspaceSize" not in param_names
    assert "executor" not in param_names


def test_parse_md_html_table_constraints():
    """Verify HTML <table> constraints are extracted."""
    result = parse_md(SWI_GLU_QUANT_MD)
    html_constraints = [c for c in result["constraints"] if c["source"] == "html_table"]
    assert len(html_constraints) >= 3
    rules = " ".join(c["rule"] for c in html_constraints)
    assert "FLOAT" in rules or "dynamic" in rules.lower()


def test_parse_md_markdown_table_constraints():
    """Verify Markdown pipe table constraints are extracted."""
    result = parse_md(DYNAMIC_QUANT_MD)
    md_constraints = [c for c in result["constraints"] if c["source"] == "markdown_table"]
    assert len(md_constraints) >= 1


# ── Reconciliation tests ────────────────────────────────────────────

def test_reconcile_name_normalization():
    """Verify camelCase .md params match snake_case _def.cpp params."""
    result = run_parse_from_strings(SWI_GLU_QUANT_MD, SWI_GLU_QUANT_DEF)
    by_name = {p["name"]: p for p in result["parameters"]}
    # These should be "both" — matched via normalization
    assert by_name["smooth_scales"]["source"] == "both"
    assert by_name["group_index"]["source"] == "both"
    assert by_name["activate_left"]["source"] == "both"
    assert by_name["quant_mode"]["source"] == "both"


def test_reconcile_no_false_conflicts():
    """Verify normalization eliminates false conflicts."""
    result = run_parse_from_strings(SWI_GLU_QUANT_MD, SWI_GLU_QUANT_DEF)
    # Should NOT have conflicts for smooth_scales, group_index, etc.
    conflict_params = {c["param"] for c in result["conflicts"]}
    assert "smooth_scales" not in conflict_params
    assert "group_index" not in conflict_params
    assert "activate_left" not in conflict_params


def test_reconcile_input_optionality():
    """Verify input optionality from .ParamType() is preserved."""
    result = run_parse_from_strings(SWI_GLU_QUANT_MD, SWI_GLU_QUANT_DEF)
    by_name = {p["name"]: p for p in result["parameters"]}
    assert by_name["x"]["optionality"] == "REQUIRED"
    assert by_name["smooth_scales"]["optionality"] == "OPTIONAL"


def test_reconcile_attr_type_from_hint():
    """Verify attr type is inferred from .Bool()/.String()/.Int() hints."""
    result = run_parse_from_strings(SWI_GLU_QUANT_MD, SWI_GLU_QUANT_DEF)
    by_name = {p["name"]: p for p in result["parameters"]}
    assert by_name["activate_left"]["type"] == "bool"
    assert by_name["quant_mode"]["type"] == "str"
    assert by_name["dst_type"]["type"] == "int"


# ── End-to-end tests ────────────────────────────────────────────────

def test_parse_basic_name_and_type():
    """Verify aclnn name derivation and operator type detection."""
    result = run_parse_from_strings(SWI_GLU_QUANT_MD, SWI_GLU_QUANT_DEF)
    assert result["aclnn_name"] == "aclnnSwiGluQuant"
    assert result["npu_name"] == "npu_swi_glu_quant"
    assert result["op_type"] == "Forward"


def test_parse_backward_op_type():
    """Verify _grad suffix detection."""
    result = run_parse_from_strings(SWI_GLU_QUANT_MD, SWI_GLU_QUANT_DEF, "npu_fast_gelu_grad")
    assert result["op_type"] == "Backward"


def test_parse_constraints_confidence_breakdown():
    """Verify constraints have proper confidence levels."""
    result = run_parse_from_strings(SWI_GLU_QUANT_MD, SWI_GLU_QUANT_DEF)
    breakdown = result["parse_meta"]["constraint_confidence_breakdown"]
    assert breakdown["medium"] >= 3  # HTML table rows
    assert breakdown["low"] >= 1     # freetext


def test_parse_dtypes():
    """Verify DataType constants are extracted."""
    result = run_parse_from_strings(SWI_GLU_QUANT_MD, SWI_GLU_QUANT_DEF)
    dtypes = result["def_dtypes_seen"]
    assert "DT_FLOAT" in dtypes
    assert "DT_INT8" in dtypes


# ── Helper ───────────────────────────────────────────────────────────

def run_parse_from_strings(md_str: str, def_str: str, op_name: str | None = None) -> dict:
    """Run parse on string inputs by writing to temp files."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f_md:
        f_md.write(md_str)
        md_path = f_md.name
    with tempfile.NamedTemporaryFile(suffix="_def.cpp", mode="w", delete=False) as f_def:
        f_def.write(def_str)
        def_path = f_def.name
    try:
        return run_parse(md_path, def_path, op_name)
    finally:
        os.unlink(md_path)
        os.unlink(def_path)


# ── Pitfall detector ─────────────────────────────────────────────────

def test_detect_pitfalls_conditional_output_fires():
    # bool attr + >=2 outputs → conditional_output_allocation alert
    params = [
        {"name": "x", "category": "input", "type": "Tensor"},
        {"name": "out_flag", "category": "attr", "type": "bool"},
    ]
    outputs = [{"name": "y"}, {"name": "expert_idx"}, {"name": "out"}]
    alerts = detect_pitfalls(params, outputs)
    ids = [a["id"] for a in alerts]
    assert "conditional_output_allocation" in ids
    a = next(a for a in alerts if a["id"] == "conditional_output_allocation")
    assert "out_flag" in a["trigger"]
    assert "C-cpp" in a["applies_to_steps"] and "C-meta" in a["applies_to_steps"]
    assert a["must_read_case"].endswith("npu_moe_gating_top_k-create.md")


def test_detect_pitfalls_no_bool_attr_silent():
    params = [
        {"name": "x", "category": "input", "type": "Tensor"},
        {"name": "k", "category": "attr", "type": "int"},
    ]
    outputs = [{"name": "y"}, {"name": "idx"}]
    assert detect_pitfalls(params, outputs) == []


def test_detect_pitfalls_single_output_silent():
    # bool attr but only 1 output → no conditional-output trap
    params = [{"name": "flag", "category": "attr", "type": "bool"}]
    outputs = [{"name": "y"}]
    assert detect_pitfalls(params, outputs) == []


# ── H6: nested parens in chained methods ──────────────────────────

def test_parse_def_nested_parens_in_method_args():
    """Verify _scan_chained_methods (extract_balanced) handles nested parens
    that [^)]* regex would truncate on. H6 fix for parser.py:147,186.

    The key: a method with nested parens (like .Bool(some_func(a, b))) would
    truncate the old [^)]* chain capture, losing SUBSEQUENT methods
    (like .ParamType(REQUIRED) appearing after it). _scan_chained_methods
    captures the full chain via extract_balanced."""
    DEF_WITH_NESTED = """\
class TestOp : public OpDef {
    TestOp(const char *name) : OpDef(name) {
        this->Input("x")
            .Bool(some_func(a, b))
            .ParamType(REQUIRED);
        this->Attr("mode")
            .AttrType(OPTIONAL)
            .Int(42);
    }
};
"""
    result = parse_def_cpp(DEF_WITH_NESTED)
    # Input "x": .Bool(some_func(a,b)) has nested parens — old [^)]* would
    # truncate the chain at the first ')', losing .ParamType(REQUIRED)
    by_name = {i["name"]: i for i in result["inputs"]}
    assert by_name["x"]["optionality"] == "REQUIRED", \
        f"Chain truncated by nested parens, lost .ParamType, got: {by_name['x']}"

    # Attr "mode": simple .Int(42) — default and type should be extracted
    by_attr = {a["name"]: a for a in result["attrs"]}
    assert by_attr["mode"]["type_hint"] == "Int", \
        f"Attr type_hint wrong: {by_attr['mode'].get('type_hint')}"
    assert by_attr["mode"]["default_raw"] == "42", \
        f"Attr default wrong: {by_attr['mode'].get('default_raw')}"


# ── H7: definition after call site ────────────────────────────────

def test_parse_cpp_definition_after_call_site():
    """Verify _find_definition_match picks the definition (')' followed by '{')
    over a call site when the call site appears AFTER the definition. H7 fix.

    The regex matches 'ReturnType op_name(' — both the definition and a call
    like 'return npu_foo(' match. Old matches[-1] picked the LAST = call site
    (wrong). _find_definition_match picks the one followed by '{' (correct)."""
    from lib.signature import _find_definition_match
    import re as _re

    content = """at::Tensor npu_foo(const at::Tensor& input) {
    return input;
}

at::Tensor caller(const at::Tensor& x) {
    return npu_foo(x);
}
"""
    pattern = _re.compile(
        r'(\w[\w\s:<>,*&]*?)\s+' + _re.escape("npu_foo") + r'\s*\(',
        _re.DOTALL
    )
    matches = list(pattern.finditer(content))
    assert len(matches) >= 2, f"Expected >=2 matches (definition + call site), got {len(matches)}"

    m = _find_definition_match(matches, content)
    # Definition is in the FIRST function (at::Tensor npu_foo(...) {)
    # Call site is in the SECOND function (return npu_foo(x);)
    assert m.start() < content.index("caller"), \
        f"Should pick definition (before 'caller'), got match at {m.start()}"


# ── P1 negative tests ─────────────────────────────────────────────

def test_parse_truncated_def_cpp():
    """Parser should not crash on truncated/malformed _def.cpp."""
    truncated = 'class Foo : public OpDef {\n    Foo() : OpDef("foo") {\n        this->Input("x")\n            .ParamType(REQUIRED'  # no closing
    result = parse_def_cpp(truncated)
    # Should still extract the input name, just without full chain
    assert any(i["name"] == "x" for i in result["inputs"])


def test_parse_empty_def_cpp():
    """Parser should handle empty/minimal _def.cpp gracefully."""
    result = parse_def_cpp("")
    assert result["inputs"] == []
    assert result["outputs"] == []
    assert result["attrs"] == []


def test_summary_template_blocks_exist():
    """Template-drift guard: summary.py's _PARSE_BLOCK/_VERIFY_BLOCK must still
    exist in summary-template.md (H4 from design doc)."""
    from lib.summary import _PARSE_BLOCK, _VERIFY_BLOCK, TEMPLATE_PATH
    tpl = TEMPLATE_PATH.read_text()
    assert _PARSE_BLOCK in tpl, "_PARSE_BLOCK not in summary-template.md (template drifted)"
    assert _VERIFY_BLOCK in tpl, "_VERIFY_BLOCK not in summary-template.md (template drifted)"

