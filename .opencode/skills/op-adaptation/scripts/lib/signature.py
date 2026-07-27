"""C-verify: cross-check signatures across YAML, C++, Meta, and Derivatives.

Extracts parameter lists from each source and compares them.
Reports mismatches in parameter names, types, and order.
"""

import re
import json
from pathlib import Path
from typing import Any

from .common import find_repo_root, repo_paths, YAML_TO_CPP, extract_balanced


def _load_known_inconsistent() -> set:
    """Load the known_inconsistent allowlist (op names with accepted historical mismatches).

    Same file the test suite consumes (tests/known_inconsistent.json) — wired into
    the runtime so verify_op and tests share one semantics.
    """
    p = Path(__file__).resolve().parent.parent / "tests" / "known_inconsistent.json"
    try:
        data = json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return set()
    ops = data.get("operators", []) if isinstance(data, dict) else []
    return set(ops)


# ── YAML func: line parser ────────────────────────────────────────────

def parse_yaml_func_line(func_line: str) -> dict:
    """Parse a YAML ``func:`` value like::

        npu_op(Tensor x, int dim, *, Tensor? mask=None) -> Tensor
        npu_op(Tensor input, *, Tensor? a=None) -> (Tensor, Tensor)

    Returns:
        {
            "op_name": "npu_op",
            "return_type": "Tensor",
            "params": [
                {"name": "x", "type": "Tensor", "required": True, "default": None},
                {"name": "dim", "type": "int", "required": True, "default": None},
                {"name": "mask", "type": "Tensor?", "required": False, "default": "None"},
            ],
        }
    """
    # Extract op name (first identifier before the first "(")
    m = re.match(r'^\s*(\w+)\s*\(', func_line)
    if not m:
        return {"op_name": None, "return_type": None, "params": [], "error": f"Cannot parse func line: {func_line}"}

    op_name = m.group(1)
    paren_start = m.end() - 1  # position of "("

    # Find the matching ")" by tracking depth
    depth = 0
    paren_end = -1
    for i in range(paren_start, len(func_line)):
        if func_line[i] == "(":
            depth += 1
        elif func_line[i] == ")":
            depth -= 1
            if depth == 0:
                paren_end = i
                break

    if paren_end == -1:
        return {"op_name": op_name, "return_type": None, "params": [], "error": f"Unbalanced parens: {func_line}"}

    args_str = func_line[paren_start + 1:paren_end].strip()

    # Everything after the closing ")" is the return type (after "->")
    after_parens = func_line[paren_end + 1:].strip()
    return_type = None
    if after_parens.startswith("->"):
        return_type = after_parens[2:].strip()

    params = _parse_param_list(args_str)
    return {"op_name": op_name, "return_type": return_type, "params": params}


def _parse_param_list(args_str: str) -> list[dict]:
    """Parse a comma-separated parameter list, handling ``*`` separator."""
    params = []
    if not args_str:
        return params

    # Split by comma respecting nested brackets
    parts = _split_respecting_brackets(args_str)

    past_star = False
    for part in parts:
        part = part.strip()
        if part == "*":
            past_star = True
            continue
        if not part:
            continue

        # Handle .List overload: npu_op.List(...)
        if part.startswith("."):
            continue

        # Parse "Type name" or "Type? name=default" or "Type name=default"
        default = None
        if "=" in part:
            part, default = part.rsplit("=", 1)
            part = part.strip()
            default = default.strip()

        # Last token is the name, rest is the type
        tokens = part.split()
        if len(tokens) >= 2:
            ptype = " ".join(tokens[:-1])
            pname = tokens[-1]
        elif len(tokens) == 1:
            ptype = ""
            pname = tokens[0]
        else:
            continue

        required = not past_star and not default

        params.append({
            "name": pname,
            "type": ptype,
            "required": required,
            "default": default,
        })

    return params


def _split_respecting_brackets(s: str) -> list[str]:
    """Split by comma, respecting nested () and []."""
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


def find_yaml_func_raw(op_name: str, yaml_path: Path) -> str | None:
    """Return the verbatim func: signature for op_name (multi-line joined).

    Preserves the exact original text incl. '*' separators and default
    formatting — use this (not a reconstruction) when the signature string
    must match the YAML exactly, e.g. for schema JSON `func:` keys.
    """
    content = yaml_path.read_text()
    lines = content.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("- func:") or stripped.startswith("func:"):
            func_val = stripped.replace("- func:", "").replace("func:", "").strip()
            if func_val.startswith(op_name + "("):
                # The func line may span multiple lines if it has many params.
                # Continuation lines are indented (start with spaces, not "-").
                full = func_val
                j = i
                while full.count("(") > full.count(")"):
                    j += 1
                    if j >= len(lines):
                        break
                    next_stripped = lines[j].strip()
                    if next_stripped.startswith("- ") or next_stripped.startswith("func:") or next_stripped.startswith("op_api:") or next_stripped.startswith("acl_op:") or next_stripped.startswith("exposed:"):
                        break
                    full += " " + next_stripped
                return full
    return None


def find_yaml_entry(op_name: str, yaml_path: Path) -> dict | None:
    """Find a specific operator's func: line in op_plugin_functions.yaml."""
    full = find_yaml_func_raw(op_name, yaml_path)
    if full is None:
        return None
    return parse_yaml_func_line(full)


# ── C++ function signature parser ────────────────────────────────────

def find_cpp_file(op_name: str, opapi_dir: Path) -> Path | None:
    """Find the C++ file containing the operator implementation.

    Searches all .cpp files in the opapi directory for the function definition.
    File naming varies: *KernelNpuOpApi.cpp, *NpuOpApi.cpp, *OpApi.cpp, etc.
    Skips files that cannot be read (encoding/permission errors) instead of
    aborting the entire verify (H16 fix).
    """
    if not opapi_dir.exists():
        return None
    pattern = re.compile(r'\b' + re.escape(op_name) + r'\s*\(')
    for cpp_file in sorted(opapi_dir.glob("*.cpp")):
        try:
            content = cpp_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            continue
        if pattern.search(content):
            return cpp_file
    return None


def _find_definition_match(matches, content: str):
    """Prefer the match whose ')' is followed by '{' (definition, not call site).

    Uses extract_balanced to find the matching ')' for each match's '(',
    then checks if '{' follows. Falls back to first match if none qualify.
    """
    for m in matches:
        open_pos = m.end() - 1  # position of '('
        inner = extract_balanced(content, open_pos)
        if inner is not None:
            # extract_balanced returns content BETWEEN parens (exclusive of both)
            # position of ')' = open_pos + len(inner) + 1
            # position AFTER ')' = open_pos + len(inner) + 2
            close_end = open_pos + len(inner) + 2
            after = content[close_end:close_end + 50]
            if re.match(r'\s*\{', after):
                return m
    return matches[0]


def parse_cpp_signature(content: str, op_name: str) -> dict | None:
    """Extract function signature from C++ file.

    Finds ``ReturnType op_name(...)`` and parses parameters.
    """
    # Find the function definition
    # Pattern: ReturnType op_name(
    # ReturnType may be complex: at::Tensor, std::tuple<...>, at::Tensor&
    # We look for op_name preceded by return type and followed by (
    pattern = re.compile(
        r'(\w[\w\s:<>,*&]*?)\s+' + re.escape(op_name) + r'\s*\(',
        re.DOTALL
    )
    matches = list(pattern.finditer(content))
    if not matches:
        return None

    # Prefer the definition (')' followed by '{') over call sites
    m = _find_definition_match(matches, content)

    # Extract everything from op_name( to the matching )
    start = m.end() - 1  # position of (
    depth = 0
    end = start
    for i in range(start, len(content)):
        if content[i] == "(":
            depth += 1
        elif content[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break

    args_str = content[start + 1:end].strip()
    return_type = m.group(1).strip()

    params = _parse_cpp_params(args_str)

    # H18: assess parse confidence (A1 regex DRAFT demotion)
    confidence = "high"
    if "<" in return_type or ">" in return_type:
        confidence = "low"  # templates like std::tuple<...> — regex unreliable
    if any(p.get("name") is None for p in params if p):
        confidence = "low"  # couldn't extract param name — parse failure

    return {
        "op_name": op_name,
        "return_type": return_type,
        "params": params,
        "parse_confidence": confidence,
    }


def _parse_cpp_params(args_str: str) -> list[dict]:
    """Parse C++ parameter list into name/type pairs."""
    if not args_str or args_str == "void":
        return []

    params = []
    parts = _split_respecting_brackets(args_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Remove default values
        default = None
        if "=" in part:
            part, default = part.rsplit("=", 1)
            part = part.strip()
            default = default.strip()

        # The parameter name is the last identifier
        # Type is everything before it
        # Handle: "const at::Tensor &input", "c10::optional<at::ScalarType> dst_type"
        # Handle reference/pointer: "at::Tensor& result", "const at::Tensor& self"
        pm = re.match(r'^(.*?)(\b\w+)\s*$', part, re.DOTALL)
        if pm:
            ptype = pm.group(1).strip()
            pname = pm.group(2).strip()
            if ptype:
                params.append({"name": pname, "type": ptype, "default": default})
            else:
                # Only one token — might be just a type without name
                params.append({"name": None, "type": pname, "default": default})
        else:
            params.append({"name": None, "type": part, "default": default})

    return params


# ── Meta registration parser ─────────────────────────────────────────

def parse_meta_signature(content: str, op_name: str) -> dict | None:
    """Extract Meta registration from _meta_registrations.py.

    Finds ``@impl(m, "op_name")`` followed by ``def xxx_meta(...):``
    """
    # Find @impl(m, "op_name")
    pattern = re.compile(
        r'@impl\s*\(\s*\w+\s*,\s*"' + re.escape(op_name) + r'"'
    )
    m = pattern.search(content)
    if not m:
        return None

    # Find the def line after @impl
    after = content[m.end():m.end() + 500]
    def_match = re.search(r'def\s+(\w+)\s*\(([^)]*)\)', after, re.DOTALL)
    if not def_match:
        return None

    func_name = def_match.group(1)
    args_str = def_match.group(2).strip()

    # Remove 'self' if present (it's the dispatcher context, not a real param)
    params = _parse_python_params(args_str)

    return {
        "op_name": op_name,
        "meta_func_name": func_name,
        "params": params,
    }


def _parse_python_params(args_str: str) -> list[dict]:
    """Parse Python function parameter list."""
    if not args_str:
        return []

    params = []
    parts = _split_respecting_brackets(args_str)
    past_star = False

    for part in parts:
        part = part.strip()
        if part == "*":
            past_star = True
            continue
        if not part:
            continue

        default = None
        if "=" in part:
            part, default = part.rsplit("=", 1)
            part = part.strip()
            default = default.strip()

        # Python params don't have explicit types
        tokens = part.split()
        pname = tokens[-1] if tokens else part

        params.append({
            "name": pname,
            "type": None,  # Python has no explicit types
            "required": not past_star and not default,
            "default": default,
        })

    return params


# ── Derivatives parser ───────────────────────────────────────────────

def parse_derivatives_entry(op_name: str, deriv_path: Path) -> dict | None:
    """Find the derivatives.yaml entry for an operator.

    Returns the name line and gradient formulas.
    """
    if not deriv_path.exists():
        return None
    content = deriv_path.read_text()

    # Find "- name: op_name(" — match the exact op name followed by ( to avoid
    # substring matches (e.g. npu_moe_token_permute vs npu_moe_token_permute_with_routing_map)
    pattern = re.compile(
        r'^\s*-\s*name:\s*(' + re.escape(op_name) + r'\(.*)$',
        re.MULTILINE
    )
    m = pattern.search(content)
    if not m:
        return None

    name_line = m.group(1).strip()
    line_num = content[:m.start()].count("\n") + 1

    # Parse the name line as a func signature
    sig = parse_yaml_func_line(name_line)

    # Read gradient formulas (lines after name until next - name: or version:)
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
        # Formula: "param_name: backward_op(grad, ...)"
        fm = re.match(r'^(\w[\w,]*)\s*:\s*(.*)$', stripped)
        if fm:
            param_key = fm.group(1).strip()
            formula = fm.group(2).strip()
            formulas[param_key] = formula

    return {
        "name_line": name_line,
        "line_number": line_num,
        "signature": sig,
        "formulas": formulas,
    }


# ── Cross-verification ───────────────────────────────────────────────

def verify_op(op_name: str, repo_root: Path | None = None, cpp_file: str | None = None) -> dict:
    """Cross-verify signatures for an operator across all sources.

    Returns a report dict with:
      - yaml: parsed YAML func entry
      - cpp: parsed C++ signature
      - meta: parsed Meta registration
      - derivatives: parsed derivatives entry
      - mismatches: list of mismatch descriptions
      - passed: bool
    """
    if repo_root is None:
        repo_root = find_repo_root()
    paths = repo_paths(repo_root)

    report: dict[str, Any] = {
        "op_name": op_name,
        "yaml": None,
        "cpp": None,
        "meta": None,
        "derivatives": None,
        "mismatches": [],
        "passed": True,
    }

    # 1. YAML
    yaml_entry = find_yaml_entry(op_name, paths["yaml"])
    report["yaml"] = yaml_entry
    if not yaml_entry:
        report["mismatches"].append({"check": "yaml", "error": f"Operator '{op_name}' not found in op_plugin_functions.yaml"})

    # 2. C++
    cpp_path = None
    cpp_sig = None
    if cpp_file:
        cpp_path = Path(cpp_file)
    else:
        cpp_path = find_cpp_file(op_name, paths["opapi_dir"])
    if cpp_path:
        cpp_content = cpp_path.read_text()
        cpp_sig = parse_cpp_signature(cpp_content, op_name)
        report["cpp"] = cpp_sig
        report["cpp_file"] = str(cpp_path)
        if not cpp_sig:
            report["mismatches"].append({"check": "cpp", "error": f"Cannot parse C++ signature for '{op_name}' in {cpp_path}"})
    else:
        # No hand-written C++ — may be a gen_opapi operator. Warn, don't hard-fail.
        report["mismatches"].append({
            "check": "cpp",
            "error": f"C++ file not found for '{op_name}' in {paths['opapi_dir']} (may be a gen_opapi operator)",
            "severity": "warning",
        })

    # 3. Meta
    meta_content = paths["meta"].read_text()
    meta_sig = parse_meta_signature(meta_content, op_name)
    report["meta"] = meta_sig
    if not meta_sig:
        if _is_aclgraph_aux_op(op_name, paths["yaml"]):
            # ACLGraph 辅助接口（带 workspace 的 .out / _get_max_workspace / _infer_output）
            # 不注册 meta 是预期形态（handler 直调，不走 fake tensor meta），
            # 降级为 warning。普通 .out 变体（无 workspace）不在豁免范围，缺 meta 仍 BLOCK。
            report["mismatches"].append({
                "check": "meta",
                "error": f"Meta registration not found for '{op_name}' (expected: ACLGraph auxiliary interfaces have no meta)",
                "severity": "warning",
            })
        else:
            report["mismatches"].append({"check": "meta", "error": f"Meta registration not found for '{op_name}' in _meta_registrations.py"})

    # 4. Derivatives (optional — not all operators have derivatives)
    deriv_entry = parse_derivatives_entry(op_name, paths["derivatives"])
    report["derivatives"] = deriv_entry

    # 5. ACLGraph structural checks — only fires when the op has .out /
    #    _get_max_workspace YAML entries; skipped for auxiliary interfaces
    #    themselves (".out" in name / leading "_"). Does not affect normal ops.
    if yaml_entry and "." not in op_name and not op_name.startswith("_"):
        out_exists = find_yaml_func_raw(op_name + ".out", paths["yaml"]) is not None
        gmw_exists = find_yaml_func_raw("_" + op_name + "_get_max_workspace", paths["yaml"]) is not None
        if out_exists or gmw_exists:
            report["aclgraph"] = {
                "out_entry": out_exists,
                "get_max_workspace_entry": gmw_exists,
            }
            report["mismatches"].extend(check_aclgraph_structure(op_name, paths["yaml"]))

    # ── Cross-checks ──
    if yaml_entry and cpp_sig:
        _check_yaml_vs_cpp(yaml_entry, cpp_sig, report)

    if yaml_entry and meta_sig:
        _check_yaml_vs_meta(yaml_entry, meta_sig, report)

    if yaml_entry and deriv_entry:
        _check_derivatives(yaml_entry, deriv_entry, report, paths["yaml"])

    # ── Apply known_inconsistent allowlist (shared with test suite) ──
    known = _load_known_inconsistent()
    if op_name in known:
        report["known_inconsistent"] = True
        for m in report["mismatches"]:
            m["severity"] = "known"
    else:
        report["known_inconsistent"] = False

    # passed: only hard mismatches fail. Warnings (param order, intentional
    # derivatives name differences) and known-inconsistent entries do NOT fail —
    # otherwise ops with legitimate warnings could never reach passed:true.
    hard = [m for m in report["mismatches"] if m.get("severity") not in ("warning", "known")]
    report["passed"] = len(hard) == 0
    report["warning_count"] = sum(1 for m in report["mismatches"] if m.get("severity") == "warning")
    report["known_count"] = sum(1 for m in report["mismatches"] if m.get("severity") == "known")

    return report


def _check_yaml_vs_cpp(yaml_entry: dict, cpp_sig: dict, report: dict):
    """Compare YAML func params with C++ function params."""
    yaml_params = yaml_entry.get("params", [])
    cpp_params = cpp_sig.get("params", [])

    yaml_names = [p["name"] for p in yaml_params if p.get("name")]
    cpp_names = [p["name"] for p in cpp_params if p.get("name")]

    # Check param name set
    yaml_set = set(yaml_names)
    cpp_set = set(cpp_names)

    only_yaml = yaml_set - cpp_set
    only_cpp = cpp_set - yaml_set

    if only_yaml:
        report["mismatches"].append({
            "check": "yaml_vs_cpp",
            "error": f"Params in YAML but not in C++: {sorted(only_yaml)}",
        })
    if only_cpp:
        report["mismatches"].append({
            "check": "yaml_vs_cpp",
            "error": f"Params in C++ but not in YAML: {sorted(only_cpp)}",
        })

    # Check type compatibility for common params
    yaml_by_name = {p["name"]: p for p in yaml_params if p.get("name")}
    cpp_by_name = {p["name"]: p for p in cpp_params if p.get("name")}
    for name in sorted(yaml_set & cpp_set):
        yaml_type = yaml_by_name[name]["type"]
        cpp_type = cpp_by_name[name]["type"]
        if not _types_compatible(yaml_type, cpp_type):
            report["mismatches"].append({
                "check": "yaml_vs_cpp",
                "error": f"Param '{name}': YAML type '{yaml_type}' vs C++ type '{cpp_type}' — incompatible (heuristic; may be a false alarm like Tensor(b!) vs at::Tensor&)",
                "severity": "warning",
            })

    # Check order
    common_order_yaml = [n for n in yaml_names if n in cpp_set]
    common_order_cpp = [n for n in cpp_names if n in yaml_set]
    if common_order_yaml != common_order_cpp:
        report["mismatches"].append({
            "check": "yaml_vs_cpp",
            "error": f"Parameter order differs. YAML: {common_order_yaml}, C++: {common_order_cpp}",
            "severity": "warning",
        })


def _check_yaml_vs_meta(yaml_entry: dict, meta_sig: dict, report: dict):
    """Compare YAML func params with Meta Python params."""
    yaml_params = yaml_entry.get("params", [])
    meta_params = meta_sig.get("params", [])

    yaml_names = [p["name"] for p in yaml_params if p.get("name")]
    meta_names = [p["name"] for p in meta_params if p.get("name")]

    # Normalize: Meta registrations sometimes rename params with _dummy suffix
    # (e.g. "input" → "input_dummy"). Strip the suffix for comparison.
    def _normalize_meta_name(name: str) -> str:
        if name.endswith("_dummy"):
            return name[:-len("_dummy")]
        return name

    meta_names_normalized = [_normalize_meta_name(n) for n in meta_names]

    yaml_set = set(yaml_names)
    meta_set = set(meta_names_normalized)

    only_yaml = yaml_set - meta_set
    only_meta = meta_set - yaml_set

    if only_yaml:
        report["mismatches"].append({
            "check": "yaml_vs_meta",
            "error": f"Params in YAML but not in Meta: {sorted(only_yaml)}",
        })
    if only_meta:
        # Check if the raw (un-normalized) names differ — may be _dummy renames
        raw_only = set(meta_names) - set(n for n in meta_names if _normalize_meta_name(n) in yaml_set)
        if raw_only:
            report["mismatches"].append({
                "check": "yaml_vs_meta",
                "error": f"Params in Meta but not in YAML: {sorted(raw_only)}",
                "severity": "warning",
            })


def _check_derivatives(yaml_entry: dict, deriv_entry: dict, report: dict, yaml_path: Path | None = None):
    """Verify derivatives name: matches YAML func: and backward op is registered.

    Three checks per verify-guide.md:
      1. name: field matches forward func: line (warn on mismatch, don't fail —
         fast_gelu vs npu_fast_gelu is legitimate)
      2. Every parameter referenced in backward formula body exists in forward signature
      3. Backward operator name has a registered func: entry in op_plugin_functions.yaml
    """
    deriv_sig = deriv_entry.get("signature", {})
    yaml_name = yaml_entry.get("op_name")
    deriv_name = deriv_sig.get("op_name")

    # Check 1: name: vs func: line — report mismatch as warning (not failure)
    if deriv_name and yaml_name and deriv_name != yaml_name:
        report["mismatches"].append({
            "check": "derivatives",
            "error": f"Derivatives name '{deriv_name}' != YAML func name '{yaml_name}' "
                     f"(may be intentional for native ops like fast_gelu vs npu_fast_gelu)",
            "severity": "warning",
        })

    # Check 2: formula keys and formula body params must reference forward signature params
    yaml_param_names = {p["name"] for p in yaml_entry.get("params", []) if p.get("name")}

    # Known placeholders that are valid in formula expressions
    placeholders = {"grad", "result0", "result1", "result2", "result3",
                    "grads[0]", "grads[1]", "grads[2]", "grads[3]",
                    "non_differentiable", "auto_linear", "target"}

    for param_key, formula in deriv_entry.get("formulas", {}).items():
        # Check formula key is a forward param or known placeholder
        if param_key not in yaml_param_names and param_key not in placeholders:
            report["mismatches"].append({
                "check": "derivatives",
                "error": f"Derivatives formula key '{param_key}' not in YAML params: {sorted(yaml_param_names)}",
                "severity": "warning",
            })

        # Check formula body: extract referenced identifiers and verify they're
        # either forward params, known placeholders, or backward op params
        # (backward op params we can't fully check without parsing the backward
        # signature, so we check for obvious typos against forward params)
        formula_identifiers = set(re.findall(r'\b(\w+)\b', formula))
        # Remove function name and known keywords
        formula_identifiers -= {"npu", "grad", "result", "result0", "result1", "result2",
                                 "grads", "non_differentiable", "auto_linear",
                                 "sym_size", "sym_sizes", "scalar_type", "dim", "expect_int",
                                 "size", "dtype", "true", "false", "None"}
        # Also remove the backward op name (first identifier before "(")
        formula_call_match = re.match(r'(\w+)\s*\(', formula.strip())
        if formula_call_match:
            formula_identifiers -= {formula_call_match.group(1)}

        # Any remaining identifier that looks like a param name but isn't in YAML
        suspicious = formula_identifiers - yaml_param_names - placeholders
        # Filter out things that are clearly not param names (method calls, types, etc.)
        suspicious = {s for s in suspicious if s.isidentifier() and not s[0].isupper()}
        if suspicious:
            report["mismatches"].append({
                "check": "derivatives",
                "error": f"Formula for '{param_key}' references unknown identifiers: {sorted(suspicious)}. "
                         f"YAML params: {sorted(yaml_param_names)}",
                "severity": "warning",
            })

    # Check 3: backward operator name must have a registered func: entry
    for param_key, formula in deriv_entry.get("formulas", {}).items():
        formula = formula.strip()
        if formula in ("non_differentiable", "auto_linear"):
            continue
        # Extract backward op name: first identifier before "("
        m = re.match(r'(\w+)\s*\(', formula)
        if m:
            backward_op = m.group(1)
            # Check if this op is registered in YAML
            if yaml_path and yaml_path.exists():
                backward_entry = find_yaml_entry(backward_op, yaml_path)
                if not backward_entry:
                    report["mismatches"].append({
                        "check": "derivatives",
                        "error": f"Backward operator '{backward_op}' referenced in derivatives formula "
                                 f"has no func: entry in op_plugin_functions.yaml",
                    })


# ── ACLGraph structural checks (rules: yaml-guide.md §Attention ACLGraph Interfaces) ──────────────────────
#
# 分工约定：脚本只做确定性结构检查；`SymInt[]?` → `int[]?` 同位置降级只报
# warning（"存疑形态"），是否已有用户确认由 LLM 在 C-verify 核对 CHECKLIST。
# 这些检查仅在对应 YAML 条目存在时触发，不影响普通算子的 verify 行为。

_ACLGRAPH_CHECK = "aclgraph"


def _is_aclgraph_aux_op(op_name: str, yaml_path: Path | None = None) -> bool:
    """ACLGraph 辅助接口判据：

    - ``_xxx_get_max_workspace`` / ``_xxx_infer_output``：命名独占，按名判定；
    - ``.out``：仅当签名含 ``workspace`` 参数才是 ACLGraph 手工 update 形态——
      普通 mutation out 变体（``add.out``/``abs.out`` 等，无 workspace）不在此列，
      仍需 meta，缺 meta 仍报 BLOCK。
    """
    if re.match(r'^_.+_(get_max_workspace|infer_output)$', op_name):
        return True
    if ".out" in op_name:
        if yaml_path is None:
            return False  # 无 YAML 可查时保守不豁免（保留原 BLOCK 行为）
        raw = find_yaml_func_raw(op_name, yaml_path)
        return raw is not None and "workspace" in raw
    return False


def _split_func_raw(func_raw: str):
    """Split a raw func: string into (op_name, args_str, return_type), verbatim.

    Unlike parse_yaml_func_line, the args string is preserved byte-for-byte —
    needed for 逐字 comparisons and for gen-aclgraph's copy-and-insert strategy.
    """
    m = re.match(r'^\s*([\w.]+)\s*\(', func_raw)
    if not m:
        return None, None, None
    start = m.end() - 1  # position of "("
    depth = 0
    end = -1
    for i in range(start, len(func_raw)):
        if func_raw[i] == "(":
            depth += 1
        elif func_raw[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return m.group(1), None, None
    args_str = func_raw[start + 1:end].strip()
    after = func_raw[end + 1:].strip()
    return_type = after[2:].strip() if after.startswith("->") else None
    return m.group(1), args_str, return_type


def _return_arity(return_type: str | None) -> int:
    """Number of return values: '(Tensor, Tensor)' -> 2, 'Tensor' -> 1."""
    if not return_type:
        return 0
    rt = return_type.strip()
    if rt.startswith("(") and rt.endswith(")"):
        inner = rt[1:-1]
        return len([p for p in _split_respecting_brackets(inner) if p.strip()])
    return 1


def _alias_letters(n: int) -> list[str]:
    """Alias letters for n outputs: a, b, c, ... (按输出数量继续 (a!)(b!)(c!))。"""
    return [chr(ord('a') + i) for i in range(n)]


def _is_symint_downgrade(func_part: str, other_part: str) -> bool:
    """True iff other_part is func_part with SymInt -> int at the same position.

    仓内实例：FA v3 的 prefix（SymInt[]? -> int[]?）。属未验证孤例 —— 报 warning，
    不报 BLOCK；CHECKLIST 确认记录由 LLM 在 C-verify 核对。
    """
    return "SymInt" in func_part and func_part.replace("SymInt", "int") == other_part


def _downgrade_warning(label: str, func_part: str, other_part: str) -> dict:
    return {
        "check": _ACLGRAPH_CHECK,
        "severity": "warning",
        "error": f"{label}: 参数 '{func_part}' 降级为 '{other_part}'（SymInt→int 同位置降级）"
                 f"——存疑形态，默认应保留 SymInt 写法（IFA 形态）；需 CHECKLIST 有用户确认记录"
                 f"（分工：脚本只报 warning，确认检查由 LLM 在 C-verify 做）",
    }


def check_aclgraph_structure(op_name: str, yaml_path: Path) -> list[dict]:
    """Structural checks for op_name's ACLGraph entries (.out / _get_max_workspace).

    Only fires when the corresponding entries exist. Returns mismatch dicts;
    hard mismatches carry no "severity" key, downgrades are severity="warning".
    """
    mismatches: list[dict] = []
    func_raw = find_yaml_func_raw(op_name, yaml_path)
    if not func_raw:
        return mismatches
    _, func_args, func_ret = _split_func_raw(func_raw)
    if func_args is None:
        return mismatches

    out_raw = find_yaml_func_raw(op_name + ".out", yaml_path)
    if out_raw:
        mismatches += _check_out_entry(op_name, func_args, func_ret, out_raw)

    gmw_raw = find_yaml_func_raw("_" + op_name + "_get_max_workspace", yaml_path)
    if gmw_raw:
        mismatches += _check_get_max_workspace_entry(op_name, func_args, gmw_raw)

    return mismatches


def _check_out_entry(op_name: str, func_args: str, func_ret: str | None, out_raw: str) -> list[dict]:
    """`.out` 条目：workspace 前参数段逐字 == functional；尾部 workspace + alias 输出；
    返回 alias 形式匹配。"""
    errs: list[dict] = []
    label = f"{op_name}.out"
    _, out_args, out_ret = _split_func_raw(out_raw)
    if out_args is None:
        errs.append({"check": _ACLGRAPH_CHECK, "error": f"{label}: 无法解析 func 行: {out_raw}"})
        return errs

    fparts = [p.strip() for p in _split_respecting_brackets(func_args) if p.strip()]
    oparts = [p.strip() for p in _split_respecting_brackets(out_args) if p.strip()]

    if len(oparts) < len(fparts):
        errs.append({
            "check": _ACLGRAPH_CHECK,
            "error": f"{label}: 参数段（{len(oparts)} 项）比 functional（{len(fparts)} 项）短，"
                     f"workspace 前参数段必须与 functional 逐字一致",
        })
        return errs

    # 1. workspace 前参数段逐字一致（SymInt→int 同位置降级报 warning，不报 BLOCK）
    for i, (fp, op) in enumerate(zip(fparts, oparts)):
        if fp == op:
            continue
        if _is_symint_downgrade(fp, op):
            errs.append(_downgrade_warning(label, fp, op))
        else:
            errs.append({
                "check": _ACLGRAPH_CHECK,
                "error": f"{label}: workspace 前第 {i + 1} 个参数与 functional 不一致："
                         f"'{op}' != '{fp}'（名称/顺序/类型/默认值须一致）",
            })

    # 2. 尾部：Tensor? workspace=None + alias 输出（全位置参数 functional 允许插入 *）
    tail = oparts[len(fparts):]
    ws_idx = next(
        (i for i, p in enumerate(tail) if re.match(r'^Tensor\?\s+workspace\s*=\s*None$', p)),
        None,
    )
    if ws_idx is None:
        errs.append({
            "check": _ACLGRAPH_CHECK,
            "error": f"{label}: 尾部缺少 'Tensor? workspace=None'",
        })
        tail_outs = [p for p in tail if p != "*"]
    else:
        tail_outs = [p for p in tail[ws_idx + 1:] if p != "*"]

    # 3. alias 输出参数：Tensor(a!) xxx 形式，数量 == functional 返回 arity，字母 a,b,c... 连续
    arity = _return_arity(func_ret)
    letters = []
    for p in tail_outs:
        m = re.match(r'^Tensor\(([a-z]+)!\)\s+\w+$', p)
        if m:
            letters.append(m.group(1))
        else:
            errs.append({
                "check": _ACLGRAPH_CHECK,
                "error": f"{label}: 输出参数 '{p}' 不是 mutation alias 形式 Tensor(x!)",
            })
    if len(letters) != arity:
        errs.append({
            "check": _ACLGRAPH_CHECK,
            "error": f"{label}: alias 输出数量 {len(letters)} != functional 返回数量 {arity}",
        })
    expected_letters = _alias_letters(arity)
    if letters and letters != expected_letters:
        errs.append({
            "check": _ACLGRAPH_CHECK,
            "error": f"{label}: alias 字母 {letters} 应为连续序列 {expected_letters}",
        })

    # 4. 返回必须使用相同 alias
    if arity > 1:
        expected_ret = "(" + ", ".join(f"Tensor({l}!)" for l in expected_letters) + ")"
    elif arity == 1:
        expected_ret = "Tensor(a!)"
    else:
        expected_ret = None
    if expected_ret is not None and (out_ret or "") != expected_ret:
        errs.append({
            "check": _ACLGRAPH_CHECK,
            "error": f"{label}: 返回应为 alias 形式 '{expected_ret}'，实际为 '{out_ret}'",
        })

    return errs


def _check_get_max_workspace_entry(op_name: str, func_args: str, gmw_raw: str) -> list[dict]:
    """`_get_max_workspace` 条目：参数段逐字 == functional（同款降级例外）；返回 Tensor。"""
    errs: list[dict] = []
    label = f"_{op_name}_get_max_workspace"
    _, gmw_args, gmw_ret = _split_func_raw(gmw_raw)
    if gmw_args is None:
        errs.append({"check": _ACLGRAPH_CHECK, "error": f"{label}: 无法解析 func 行: {gmw_raw}"})
        return errs

    if gmw_args != func_args:
        fparts = [p.strip() for p in _split_respecting_brackets(func_args) if p.strip()]
        gparts = [p.strip() for p in _split_respecting_brackets(gmw_args) if p.strip()]
        if len(fparts) == len(gparts) and all(
            fp == gp or _is_symint_downgrade(fp, gp) for fp, gp in zip(fparts, gparts)
        ):
            for fp, gp in zip(fparts, gparts):
                if fp != gp:
                    errs.append(_downgrade_warning(label, fp, gp))
        else:
            errs.append({
                "check": _ACLGRAPH_CHECK,
                "error": f"{label}: 参数段与 functional 不逐字一致",
            })

    if (gmw_ret or "") != "Tensor":
        errs.append({
            "check": _ACLGRAPH_CHECK,
            "error": f"{label}: 返回必须是 '-> Tensor'，实际为 '-> {gmw_ret}'",
        })

    return errs


def _strip_alias(yaml_type: str) -> str:
    """Strip alias annotations like (a!)/(b!)/(c!) from YAML type strings."""
    return re.sub(r'\([a-zA-Z0-9_]+!\)', '', yaml_type)


def _types_compatible(yaml_type: str, cpp_type: str) -> bool:
    """Check if a YAML type is compatible with a C++ type."""
    if not yaml_type or not cpp_type:
        return True  # Can't verify, assume OK

    # Strip alias annotations before matching
    yaml_type = _strip_alias(yaml_type)

    # Direct match
    if yaml_type in YAML_TO_CPP:
        compatible_cpp = YAML_TO_CPP[yaml_type]
        for cpp_pattern in compatible_cpp:
            if cpp_pattern in cpp_type:
                return True

    # Fallback: check if the yaml type string appears in the cpp type
    # (handles edge cases not in the mapping table)
    if yaml_type.replace("?", "") in cpp_type:
        return True

    return False
