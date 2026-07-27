"""C-parse: parse .md (aclnn signature + constraints) and _def.cpp (inputs/attrs/versions).

Supports real CANN file formats:
  - _def.cpp: OpDef class with chained method calls
      this->Input("smooth_scales").ParamType(OPTIONAL).DataType({...})
      this->Attr("activate_left").AttrType(OPTIONAL).Bool(false)
      .Version(SWI_GLU_QUANT_VERSION_TWO)
  - .md: camelCase params with suffixes (smoothScalesOptional, yOut)
         HTML <table> and Markdown pipe tables
         GetWorkspaceSize signature with workspaceSize/executor params

Confidence levels:
  high   — structured C++ macros / fixed-format signature lines
  medium — .md constraint tables (format varies across CANN repos)
  low    — .md free-text constraint logic
"""

import re
import json
import os
from pathlib import Path
from typing import Any

from .common import find_repo_root, repo_paths, extract_balanced


def _scan_chained_methods(content: str, start: int) -> str:
    """Scan chained .method(args) calls from position start.

    Uses extract_balanced for each method's args, handling nested parens
    that [^)]* regex cannot (H6 fix). Returns concatenated chain string."""
    result = []
    i = start
    n = len(content)
    while i < n:
        while i < n and content[i] in ' \t\n\r':
            i += 1
        if i >= n or content[i] != '.':
            break
        j = i + 1
        while j < n and (content[j].isalnum() or content[j] == '_'):
            j += 1
        if j >= n or content[j] != '(':
            break
        inner = extract_balanced(content, j)
        if inner is None:
            break
        # extract_balanced returns content BETWEEN parens (exclusive of both)
        # position of ')' = j + len(inner) + 1; position after ')' = +2
        close_end = j + len(inner) + 2
        result.append(content[i:close_end])
        i = close_end
    return ''.join(result)


# ── Name normalization ───────────────────────────────────────────────

# Suffixes in .md param names that encode optionality/output direction
_MD_SUFFIXES = ["Optional", "Out", "Input", "Output", "Workspace"]

def normalize_name(name: str) -> str:
    """Normalize a parameter name for cross-source matching.

    .md files use camelCase with suffixes: smoothScalesOptional, yOut, scaleOut
    _def.cpp uses snake_case: smooth_scales, y, scale
    YAML uses snake_case: smooth_scales, y, scale

    This function:
      1. Strips known suffixes (Optional, Out, Input, Output, Workspace)
      2. Converts camelCase → snake_case
      3. Returns the normalized name
    """
    if not name:
        return name

    result = name
    # Strip suffixes (try longest first)
    changed = True
    while changed:
        changed = False
        for suffix in sorted(_MD_SUFFIXES, key=len, reverse=True):
            if result.endswith(suffix) and len(result) > len(suffix):
                result = result[:-len(suffix)]
                changed = True
                break

    # camelCase → snake_case
    # Insert _ before uppercase letters, then lowercase
    result = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', result)
    result = result.lower()

    return result


# ── _def.cpp parser ──────────────────────────────────────────────────
# CANN _def.cpp uses OpDef class with chained method calls:
#   this->Input("smooth_scales").ParamType(OPTIONAL).DataType({...})
#   this->Attr("activate_left").AttrType(OPTIONAL).Bool(false)
#   .Version(SWI_GLU_QUANT_VERSION_TWO)
#
# Also supports older REG_OP macro format:
#   .INPUT(x, ge::TensorType(DT_FLOAT))
#   .ATTR(dst_type, AttrType(OPTIONAL), ge::DataType DT_INT8)

def _extract_constructor_body(content: str) -> str:
    """Extract the main constructor body, excluding per-chip config blocks.

    Real _def.cpp files have a constructor with this->Input/Output/Attr calls
    (the main definition), plus OpAICoreConfig blocks that re-declare everything
    per chip (config950.Input(...), config_310p_910.Input(...), GetKirinCoreConfig()).
    We only want the main constructor body.
    """
    lines = content.split("\n")
    result_lines = []
    skip_mode = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect start of per-chip config blocks:
        #   OpAICoreConfig config950;
        #   OpAICoreConfig config_310p_910;
        #   OpAICoreConfig config_kirin;
        # Also detect GetKirinCoreConfig method
        if re.search(r'(OpAICoreConfig\s+config\w+|GetKirin\w+\s*\()', stripped):
            skip_mode = True

        if skip_mode:
            # End skip when we hit AddConfig (closes the config block usage)
            # or when we hit a closing brace that ends a method
            if 'AddConfig' in stripped or re.match(r'^\s*\}\s*$', stripped):
                skip_mode = False
            continue

        # Also skip lines that start with a config variable name (config950.Input, config_310p_910.Output)
        if re.match(r'^\s*config\w+\s*\.', stripped):
            continue

        result_lines.append(line)

    return "\n".join(result_lines)


def parse_def_cpp(content: str) -> dict:
    """Parse a _def.cpp file content.

    Returns:
        {
            "inputs": [{"name": str, "optionality": "REQUIRED"|"OPTIONAL"|None}],
            "outputs": [{"name": str}],
            "attrs": [{"name": str, "default_raw": str|None, "type_hint": str|None}],
            "versions": [str, ...],
            "dtypes_seen": [str, ...],
        }
    """
    # Scope to constructor body only (skip GetKirinCoreConfig etc.)
    scoped = _extract_constructor_body(content)

    result: dict[str, Any] = {
        "inputs": [],
        "outputs": [],
        "attrs": [],
        "versions": [],
        "dtypes_seen": [],
    }

    # ── Parse Input/Output/Attr with chained method calls ──
    # Pattern: this->Input("name").ParamType(OPTIONAL).DataType(...)
    #          this->Output("name")...
    #          this->Attr("name").AttrType(OPTIONAL).Bool(false)

    # Find all this->Input("name") ... chains
    # Allow whitespace (including newlines) between chained method calls
    input_pattern = re.compile(
        r'(?:this->)?\.?\s*Input\s*\(\s*"([^"]+)"\s*\)',
        re.IGNORECASE
    )
    for m in input_pattern.finditer(scoped):
        name = m.group(1)
        chain = _scan_chained_methods(scoped, m.end())
        optionality = _extract_param_type_from_chain(chain)
        result["inputs"].append({"name": name, "optionality": optionality})

    # Also support old .INPUT(name, ...) format (no quotes)
    old_input_pattern = re.compile(
        r'\.?\s*INPUT\s*\(\s*([^,\s"]+)\s*[,)]',
    )
    for m in old_input_pattern.finditer(scoped):
        name = m.group(1)
        # Don't add if already found via quoted pattern
        if not any(i["name"] == name for i in result["inputs"]):
            result["inputs"].append({"name": name, "optionality": None})

    # Outputs
    output_pattern = re.compile(
        r'(?:this->)?\.?\s*Output\s*\(\s*"([^"]+)"\s*\)',
        re.IGNORECASE
    )
    for m in output_pattern.finditer(scoped):
        name = m.group(1)
        result["outputs"].append({"name": name})

    old_output_pattern = re.compile(
        r'\.?\s*OUTPUT\s*\(\s*([^,\s"]+)\s*[,)]',
    )
    for m in old_output_pattern.finditer(scoped):
        name = m.group(1)
        if not any(o["name"] == name for o in result["outputs"]):
            result["outputs"].append({"name": name})

    # Attrs with chained methods
    attr_pattern = re.compile(
        r'(?:this->)?\.?\s*Attr\s*\(\s*"([^"]+)"\s*\)',
        re.IGNORECASE
    )
    for m in attr_pattern.finditer(scoped):
        name = m.group(1)
        chain = _scan_chained_methods(scoped, m.end())
        attr_type = _extract_attr_type_from_chain(chain)
        default_raw, type_hint = _extract_attr_default_from_chain(chain)
        result["attrs"].append({
            "name": name,
            "default_raw": default_raw,
            "type_hint": type_hint,
            "attr_type": attr_type,
        })

    # Old .ATTR(name, AttrType(...), default) format
    old_attr_pattern = re.compile(
        r'\.?\s*ATTR\s*\(\s*([^,\s"]+)\s*,\s*AttrType\s*\(\s*(\w+)\s*\)\s*(?:,\s*([^)]+))?\s*[,)]',
        re.IGNORECASE
    )
    for m in old_attr_pattern.finditer(scoped):
        name = m.group(1)
        attr_type = m.group(2)
        default_raw = m.group(3).strip() if m.group(3) else None
        if not any(a["name"] == name for a in result["attrs"]):
            result["attrs"].append({
                "name": name,
                "default_raw": default_raw,
                "type_hint": None,
                "attr_type": attr_type,
            })

    # Version markers — match any .Version(...) content (not just V\d+)
    version_pattern = re.compile(
        r'\.Version\s*\(\s*([^)]+)\s*\)',
        re.IGNORECASE
    )
    for m in version_pattern.finditer(scoped):
        v = m.group(1).strip()
        if v not in result["versions"]:
            result["versions"].append(v)

    # DataType constants
    for m in re.finditer(r'DT_\w+', content):
        if m.group(0) not in result["dtypes_seen"]:
            result["dtypes_seen"].append(m.group(0))

    return result


def _extract_param_type_from_chain(chain: str) -> str | None:
    """Extract ParamType(REQUIRED|OPTIONAL) from chained method calls."""
    m = re.search(r'\.ParamType\s*\(\s*(REQUIRED|OPTIONAL)\s*\)', chain, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _extract_attr_type_from_chain(chain: str) -> str | None:
    """Extract AttrType(REQUIRED|OPTIONAL) from chained method calls."""
    m = re.search(r'\.AttrType\s*\(\s*(REQUIRED|OPTIONAL)\s*\)', chain, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _extract_attr_default_from_chain(chain: str) -> tuple[str | None, str | None]:
    """Extract default value and type hint from chained method calls.

    Real format: .Bool(false), .String("dynamic"), .Int(0), .Float(1.0),
                 .DataType(DT_INT8), .ScalarType(...)
    """
    # Try each type method in order
    type_methods = [
        ("Bool", r'\.Bool\s*\(\s*([^)]*)\s*\)'),
        ("String", r'\.String\s*\(\s*"([^"]*)"\s*\)'),
        ("Int", r'\.Int\s*\(\s*([^)]*)\s*\)'),
        ("Float", r'\.Float\s*\(\s*([^)]*)\s*\)'),
        ("DataType", r'\.DataType\s*\(\s*(DT_\w+)\s*\)'),
        ("ScalarType", r'\.ScalarType\s*\(\s*([^)]*)\s*\)'),
    ]

    for type_name, pattern in type_methods:
        m = re.search(pattern, chain, re.IGNORECASE)
        if m:
            return m.group(1).strip(), type_name

    return None, None


# ── .md parser ───────────────────────────────────────────────────────

def _extract_aclnn_name(content: str) -> str | None:
    """Extract the aclnn function name (e.g. aclnnDynamicQuant)."""
    m = re.search(r'\baclnn([A-Z]\w+)', content)
    if m:
        return "aclnn" + m.group(1)
    return None


def _extract_md_params(content: str, aclnn_name: str | None) -> list[dict]:
    """Extract parameters from the .md GetWorkspaceSize signature.

    .md params use camelCase with suffixes: smoothScalesOptional, yOut, scaleOut
    These are normalized to snake_case for matching with _def.cpp.
    """
    params = []
    if not aclnn_name:
        return params

    # Try GetWorkspaceSize signature first
    gws_name = aclnn_name + "GetWorkspaceSize"
    sig_re = re.compile(re.escape(gws_name) + r'\s*\(([^)]*)\)', re.DOTALL)
    m = sig_re.search(content)

    if not m:
        sig_re = re.compile(re.escape(aclnn_name) + r'\s*\(([^)]*)\)', re.DOTALL)
        m = sig_re.search(content)

    if not m:
        return params

    raw_params = m.group(1).strip()
    if not raw_params:
        return params

    parts = _split_params(raw_params)
    for part in parts:
        part = part.strip()
        if not part or part == "void":
            continue
        # Skip workspace and executor params (case-insensitive)
        part_lower = part.lower()
        if "workspace" in part_lower or "executor" in part_lower:
            continue
        # Skip pointer-to-pointer outputs (aclTensor** yOut)
        if "**" in part:
            continue

        # Extract name: last identifier in the declaration
        pm = re.search(r'(\b\w+)\s*$', part)
        if pm:
            pname = pm.group(1)
            ptype = part[:pm.start()].strip()
            # Normalize the name for matching
            normalized = normalize_name(pname)
            params.append({
                "name": pname,
                "normalized_name": normalized,
                "type_raw": ptype,
            })
        else:
            params.append({"name": None, "normalized_name": None, "type_raw": part})

    return params


def _split_params(s: str) -> list[str]:
    """Split parameter string by commas, respecting nested parens/brackets."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


# ── Constraint classification ────────────────────────────────────────
# High-precision indicators that a row states an implementable constraint
# (numeric/shape/modal requirements), not a pure parameter description.
_STRONG_CONSTRAINT_INDICATORS = [
    "不支持", "仅支持", "仅能", "必须", "不得", "的倍数", "不超过",
    "不大于", "不小于", "大于", "小于", "取值", "保持一致", "一致",
    "支持范围", "范围", "为空", "非空", "空tensor",
    "must", "cannot", "only support", "require",
]


def _detect_table_type(headers: list[str]) -> str:
    """Classify a .md table by its header row.

    Returns one of: 'constraint', 'error_code', 'param_desc', 'unknown'.
    CANN .md tables: param-description tables use '参数|输入|输出|说明',
    constraint tables use '约束|限制', error tables use '错误码|错误码描述'.
    """
    h = " ".join(headers).lower()
    if "错误码" in h or "error code" in h or "error_code" in h:
        return "error_code"
    if "约束" in h or "constraint" in h or "限制" in h:
        return "constraint"
    if "参数" in h or "输入" in h or "输出" in h:
        return "param_desc"
    return "unknown"


def _is_likely_constraint(rule: str, table_type: str | None) -> bool:
    """Signal whether a row likely states an implementable constraint.

    Boundary note: this is a heuristic SIGNAL for the LLM, not a decision.
    Rows are never dropped — the LLM makes the final call.
    """
    # Error-code table rows describe errors, not constraints to implement
    if table_type == "error_code":
        return False
    # Explicit constraint table → trust the table
    if table_type == "constraint":
        return True
    # Param-desc / unknown / freetext → flag by strong indicator content
    low = rule.lower()
    return any(k in low for k in _STRONG_CONSTRAINT_INDICATORS)


def _extract_md_constraints(content: str) -> list[dict]:
    """Extract constraint rules from .md.

    Handles both Markdown pipe tables and HTML <table> tags.
    Confidence: medium for table rows, low for freetext.
    Each constraint carries an `is_likely_constraint` signal (heuristic) and,
    for table-sourced rows, a `table_type` so the LLM can filter noise without
    dropping embedded constraints.
    """
    constraints = []
    lines = content.split("\n")

    # ── Strategy 1: Markdown pipe tables ──
    in_table = False
    table_headers: list[str] = []
    md_table_type = "unknown"
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r'^\|[\s\-:|]+\|$', next_line):
                in_table = True
                table_headers = [h.strip() for h in stripped.split("|") if h.strip()]
                md_table_type = _detect_table_type(table_headers)
                continue
        if in_table:
            if stripped.startswith("|"):
                cells = [c.strip() for c in stripped.split("|") if c.strip()]
                if all(re.match(r'^[-:]+$', c) for c in cells):
                    continue
                header_lower = [h.lower() for h in table_headers]
                # Capture rows from constraint/param-desc tables (the latter
                # often embed real constraints in the '说明' column)
                if any("constraint" in h or "限制" in h or "约束" in h or "说明" in h for h in header_lower):
                    rule = " | ".join(cells)
                    constraints.append({
                        "rule": rule,
                        "md_line": i + 1,
                        "confidence": "medium",
                        "source": "markdown_table",
                        "table_type": md_table_type,
                        "is_likely_constraint": _is_likely_constraint(rule, md_table_type),
                    })
            else:
                in_table = False

    # ── Strategy 2: HTML tables ──
    constraints.extend(_extract_html_table_constraints(content))

    # ── Strategy 3: Free text with constraint keywords ──
    constraint_keywords = [
        "must be", "必须", "不支持", "仅支持", "only support",
        "限制", "constraint", "要求", "require", "不能", "cannot",
        "范围", "range", "取值",
    ]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("|") or stripped.startswith("#"):
            continue
        if stripped.startswith("<"):
            continue
        for kw in constraint_keywords:
            if kw.lower() in stripped.lower():
                constraints.append({
                    "rule": stripped,
                    "md_line": i + 1,
                    "confidence": "low",
                    "source": "freetext",
                    "table_type": None,
                    "is_likely_constraint": True,
                })
                break

    return constraints


def _extract_html_table_constraints(content: str) -> list[dict]:
    """Extract constraints from HTML <table> tags.

    Captures rows from ALL tables (param-description tables often embed real
    constraints in a '说明' column); each row carries a `table_type` and an
    `is_likely_constraint` signal so the LLM can filter noise without dropping
    embedded constraints.
    """
    constraints = []
    # Find all <table>...</table> blocks
    table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
    for table_match in table_pattern.finditer(content):
        table_content = table_match.group(1)
        # Extract rows
        row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
        rows = row_pattern.findall(table_content)

        if not rows:
            continue

        # First row is usually headers
        headers = _extract_html_row_cells(rows[0])
        table_type = _detect_table_type(headers)
        # Constraint/param-desc tables: medium confidence; others: low
        header_lower = [h.lower() for h in headers]
        is_constraintish = any(
            "constraint" in h or "限制" in h or "约束" in h or "说明" in h
            for h in header_lower
        )
        confidence = "medium" if is_constraintish else "low"
        base_line = content[:table_match.start()].count("\n") + 1

        for i, row in enumerate(rows[1:], 1):  # skip header row
            cells = _extract_html_row_cells(row)
            if not cells:
                continue
            rule = " | ".join(cells)
            constraints.append({
                "rule": rule,
                "md_line": base_line,
                "confidence": confidence,
                "source": "html_table",
                "table_type": table_type,
                "is_likely_constraint": _is_likely_constraint(rule, table_type),
            })

    return constraints


def _extract_html_row_cells(row_html: str) -> list[str]:
    """Extract cell text from <tr> content."""
    cells = []
    cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
    for m in cell_pattern.finditer(row_html):
        # Strip HTML tags from cell content
        text = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        cells.append(text)
    return cells


# ── Parameter reconciliation ──────────────────────────────────────────

def parse_md(content: str) -> dict:
    """Parse a .md file content.

    Returns:
        {
            "aclnn_name": str|None,
            "params": [{"name": str, "normalized_name": str, "type_raw": str}],
            "constraints": [{"rule": str, "md_line": int, "confidence": str, "source": str}],
        }
    """
    aclnn_name = _extract_aclnn_name(content)
    params = _extract_md_params(content, aclnn_name)
    constraints = _extract_md_constraints(content)
    return {
        "aclnn_name": aclnn_name,
        "params": params,
        "constraints": constraints,
    }


def reconcile_parameters(def_data: dict, md_data: dict) -> list[dict]:
    """Reconcile parameters from _def.cpp and .md, applying source priority.

    Uses name normalization to match camelCase .md params with snake_case _def.cpp params.
    Priority: User > _def.cpp > .md
    """
    # Build lookup by normalized name
    def_inputs = {}
    for inp in def_data["inputs"]:
        n = normalize_name(inp["name"])
        def_inputs[n] = inp

    def_attrs = {}
    for attr in def_data["attrs"]:
        n = normalize_name(attr["name"])
        def_attrs[n] = attr

    # Also build output lookup — .md output params (yOut, scaleOut) should match
    # _def.cpp outputs (y, scale), not be flagged as conflicts
    def_outputs = {}
    for out in def_data.get("outputs", []):
        n = normalize_name(out["name"])
        def_outputs[n] = out

    md_params = {}
    for p in md_data["params"]:
        n = p.get("normalized_name") or normalize_name(p["name"]) if p.get("name") else None
        if n:
            md_params[n] = p

    all_names: list[str] = []
    seen = set()
    for name in list(def_inputs.keys()) + list(def_attrs.keys()) + list(md_params.keys()):
        if name not in seen:
            all_names.append(name)
            seen.add(name)

    reconciled = []
    for name in all_names:
        in_def = name in def_inputs
        in_attr = name in def_attrs
        in_md = name in md_params
        in_def_output = name in def_outputs

        entry: dict[str, Any] = {"name": name}

        # Source determination — output params that match def_outputs are not conflicts
        if (in_def or in_attr) and in_md:
            entry["source"] = "both"
        elif in_def or in_attr:
            entry["source"] = "_def.cpp only"
            entry["conflict"] = "missing_in_md"
        elif in_md and in_def_output:
            # .md param matches a _def.cpp output — this is normal (output tensor)
            entry["source"] = "both"
            entry["category"] = "output"
        elif in_md:
            entry["source"] = ".md only"
            entry["conflict"] = "missing_in_def"

        # Type
        if in_def:
            entry["type"] = "Tensor"
            entry["category"] = "input"
        elif in_attr:
            type_hint = def_attrs[name].get("type_hint")
            entry["type"] = _infer_type_from_hint(type_hint)
            entry["category"] = "attr"
            entry["default_raw"] = def_attrs[name].get("default_raw")
            entry["type_hint"] = type_hint
        elif in_def_output:
            entry["type"] = "Tensor"
            entry["category"] = "output"
        elif in_md:
            entry["type"] = "unknown"
            entry["category"] = "md_only"
            entry["type_raw"] = md_params[name].get("type_raw", "")

        # Optionality
        if in_attr:
            attr_type = def_attrs[name].get("attr_type")
            entry["attr_type"] = attr_type
            if attr_type == "OPTIONAL":
                entry["optionality"] = "OPTIONAL"
            elif attr_type == "REQUIRED":
                entry["optionality"] = "REQUIRED"
            else:
                entry["optionality"] = None
                entry["conflict"] = entry.get("conflict", "") + " attr_type_unknown" if not entry.get("conflict") else entry["conflict"]
        elif in_def:
            # Check ParamType from _def.cpp
            opt = def_inputs[name].get("optionality")
            entry["attr_type"] = None
            if opt:
                entry["optionality"] = opt
            else:
                entry["optionality"] = "REQUIRED"  # default for inputs without explicit ParamType

        # Confidence
        entry["confidence"] = "high"

        reconciled.append(entry)

    return reconciled


def _infer_type_from_hint(type_hint: str | None) -> str:
    """Infer YAML-like type from _def.cpp attr type hint."""
    if not type_hint:
        return "unknown"
    hint_lower = type_hint.lower()
    if "bool" in hint_lower:
        return "bool"
    if "string" in hint_lower or "str" in hint_lower:
        return "str"
    if "int" in hint_lower:
        return "int"
    if "float" in hint_lower or "double" in hint_lower:
        return "float"
    if "datatype" in hint_lower:
        return "int"  # DataType attrs map to int in YAML
    if "scalartype" in hint_lower:
        return "ScalarType?"
    return "unknown"


# ── Operator type detection ──────────────────────────────────────────

def detect_op_type(npu_name: str) -> str:
    """Auto-detect operator type from name."""
    if "_grad" in npu_name:
        return "Backward"
    return "Forward"


# ── Derivatives binding check ─────────────────────────────────────────

def check_existing_derivative(op_name: str, repo_root: Path) -> dict | None:
    """Check if the operator has an existing derivatives.yaml binding."""
    paths = repo_paths(repo_root)
    deriv_path = paths["derivatives"]
    if not deriv_path.exists():
        return None
    content = deriv_path.read_text()

    # Match exact op name followed by ( to avoid substring matches
    pattern = re.compile(
        r'^\s*-\s*name:\s*(' + re.escape(op_name) + r'\(.*)$',
        re.MULTILINE
    )
    m = pattern.search(content)
    if not m:
        return None

    name_line = m.group(1).strip()
    line_num = content[:m.start()].count("\n") + 1

    # Read gradient formulas
    after = content[m.end():]
    formulas = {}
    for line in after.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- name:"):
            break
        if stripped.startswith("version:"):
            break
        if stripped.startswith("output_differentiability:"):
            continue
        if stripped.startswith("target:"):
            continue
        fm = re.match(r'^(\w[\w,]*)\s*:\s*(.*)$', stripped)
        if fm:
            param_key = fm.group(1).strip()
            formula = fm.group(2).strip()
            formulas[param_key] = formula

    return {
        "name_line": name_line,
        "line_number": line_num,
        "formulas": formulas,
    }


# ── Main parse function ───────────────────────────────────────────────

def detect_pitfalls(parameters: list[dict], def_outputs: list[dict]) -> list[dict]:
    """Deterministically flag known integration pitfalls that a case file covers.

    These alerts are NOT semantic guesses — they fire on structural signals
    (a bool attr + multiple outputs) so the relevant case is surfaced even when
    the operator's *type/scenario* does not obviously match the case's title.
    Each alert names the exact case to read BEFORE writing C++/meta.
    """
    alerts = []

    # Pitfall: flag-controlled conditional output (out_flag / output_softmax / return_*).
    # A bool attr very often gates whether an output tensor carries valid data. The trap
    # (real bug seen in npu_moe_gating_top_k): allocating that output as {0}/zero-sized
    # when flag=False → kernel tiling CheckOutShape rejects rank mismatch (err 561002).
    bool_attrs = [
        p["name"] for p in parameters
        if p.get("category") == "attr" and p.get("type") == "bool"
    ]
    output_count = len(def_outputs)
    if bool_attrs and output_count >= 2:
        alerts.append({
            "id": "conditional_output_allocation",
            "severity": "high",
            "trigger": f"bool attr(s) {bool_attrs} + {output_count} outputs",
            "risk": (
                "A bool flag may gate a conditional output. Allocating that output as "
                "{0}/zero-sized on flag=False can trip kernel tiling CheckOutShape "
                "(rank mismatch, err 561002). The flag usually controls WHETHER valid "
                "data is written, NOT whether/what-shape the tensor is allocated."
            ),
            "must_read_case": "references/examples/create-mode/npu_moe_gating_top_k-create.md",
            "applies_to_steps": ["C-cpp", "C-meta"],
            "action": (
                "BEFORE writing C++ and meta for any flag-gated output: read the case. "
                "Decide allocation shape from THIS operator's aclnn header "
                "(required/optional) + .md param-table dims — never copy a sibling's {0}. "
                "C++ and meta must use the identical shape rule."
            ),
        })

    return alerts


def run_parse(md_path: str, def_path: str, op_name: str | None = None) -> dict:
    """Parse .md + _def.cpp and produce structured JSON output."""
    md_content = Path(md_path).read_text()
    def_content = Path(def_path).read_text()

    def_data = parse_def_cpp(def_content)
    md_data = parse_md(md_content)

    # Derive operator names
    aclnn_name = md_data["aclnn_name"]
    if not op_name:
        if aclnn_name:
            camel = aclnn_name.replace("aclnn", "")
            op_name = "npu_" + re.sub(r'([A-Z])', r'_\1', camel).lower().lstrip("_")
        else:
            op_name = "npu_unknown"

    op_type = detect_op_type(op_name)

    # Reconcile parameters
    parameters = reconcile_parameters(def_data, md_data)

    # Detect conflicts
    conflicts = []
    for p in parameters:
        if p.get("conflict"):
            if "missing_in_md" in p["conflict"]:
                conflicts.append({
                    "param": p["name"],
                    "type": "missing_in_md",
                    "desc": f"_def.cpp has parameter '{p['name']}' but .md does not mention it",
                    "resolution": "awaiting_user_confirmation",
                })
            elif "missing_in_def" in p["conflict"]:
                conflicts.append({
                    "param": p["name"],
                    "type": "missing_in_def",
                    "desc": f".md has parameter '{p['name']}' but _def.cpp does not define it",
                    "resolution": "exclude_def_cpp_is_authoritative",
                })
            elif "attr_type_unknown" in p.get("conflict", ""):
                conflicts.append({
                    "param": p["name"],
                    "type": "attr_type_unknown",
                    "desc": f"Attr '{p['name']}' has no explicit AttrType(REQUIRED/OPTIONAL) marker",
                    "resolution": "awaiting_user_confirmation",
                })

    # Check existing derivatives
    repo_root = None
    has_derivative = None
    try:
        repo_root = find_repo_root()
        has_derivative = check_existing_derivative(op_name, repo_root)
    except FileNotFoundError:
        pass

    # Version markers
    version_markers = []
    for v in def_data["versions"]:
        version_markers.append({
            "version": v,
            "note": f"Version marker '{v}' detected in _def.cpp. Review which attrs are version-specific in C-confirm.",
        })

    result = {
        "aclnn_name": aclnn_name,
        "npu_name": op_name,
        "op_type": op_type,
        "has_existing_derivative": has_derivative,
        "parameters": parameters,
        "conflicts": conflicts,
        "constraints_logic": md_data["constraints"],
        "version_markers": version_markers,
        "def_outputs": def_data["outputs"],
        "def_dtypes_seen": def_data["dtypes_seen"],
        "pitfall_alerts": detect_pitfalls(parameters, def_data["outputs"]),
        "parse_meta": {
            "md_path": str(md_path),
            "def_path": str(def_path),
            "def_input_count": len(def_data["inputs"]),
            "def_attr_count": len(def_data["attrs"]),
            "md_param_count": len(md_data["params"]),
            "constraint_count": len(md_data["constraints"]),
            "constraint_confidence_breakdown": {
                "high": sum(1 for c in md_data["constraints"] if c["confidence"] == "high"),
                "medium": sum(1 for c in md_data["constraints"] if c["confidence"] == "medium"),
                "low": sum(1 for c in md_data["constraints"] if c["confidence"] == "low"),
            },
            "constraint_source_breakdown": {
                "markdown_table": sum(1 for c in md_data["constraints"] if c["source"] == "markdown_table"),
                "html_table": sum(1 for c in md_data["constraints"] if c["source"] == "html_table"),
                "freetext": sum(1 for c in md_data["constraints"] if c["source"] == "freetext"),
            },
            "constraint_likely_count": sum(1 for c in md_data["constraints"] if c.get("is_likely_constraint")),
        },
    }

    return result
