"""M-discover: grep search for an operator across the codebase.

Wraps the 8 grep commands from modify-guide.md into a single script.
Outputs a structured status table + derivatives binding direction analysis.
"""

import re
import subprocess
from pathlib import Path
from typing import Any

from .common import find_repo_root, repo_paths
from .signature import parse_derivatives_entry


def run_discover(op_name: str, repo_root: Path | None = None) -> dict:
    """Search for an operator across all file categories in the repo.

    Args:
        op_name: Operator name without version suffix (e.g. npu_dynamic_quant)
        repo_root: Repository root path (auto-detected if None)

    Returns:
        {
            "op_name": str,
            "files": [{"file": str, "found": bool, "lines": [...], "summary": str}],
            "derivatives_binding": dict|None,
        }
    """
    if repo_root is None:
        repo_root = find_repo_root()
    paths = repo_paths(repo_root)

    # Strip version suffix for search (npu_xxx_v2 → npu_xxx)
    search_name = re.sub(r'_v\d+$', '', op_name)

    # Define search targets: (label, file_path, grep_pattern)
    targets = [
        ("op_plugin_functions.yaml", paths["yaml"], search_name),
        ("derivatives.yaml", paths["derivatives"], search_name),
        ("KernelNpuOutputSize.cpp", paths["output_size_cpp"], search_name),
        ("KernelNpuOutputSize.h", paths["output_size_h"], search_name),
        ("_meta_registrations.py", paths["meta"], search_name),
        ("test_fake_tensor.py", paths["fake_tensor_test"], search_name),
        ("torch_npu_OpApi_schema_all.json", paths["schema_json"], search_name),
        ("allowlist_for_publicAPI.json", paths["allowlist_json"], search_name),
    ]

    results = []

    # Also search opapi directory for C++ files
    opapi_results = _grep_directory(paths["opapi_dir"], search_name, "*.cpp")
    if opapi_results:
        for path_str, lines in opapi_results.items():
            rel = str(Path(path_str).relative_to(repo_root))
            results.append({
                "file": rel,
                "found": True,
                "lines": lines[:5],
                "summary": f"Found at {len(lines)} location(s)",
            })
    else:
        results.append({
            "file": "op_plugin/ops/opapi/*.cpp",
            "found": False,
            "lines": [],
            "summary": "Not found in opapi directory",
        })

    # Search each target file
    for label, file_path, pattern in targets:
        if not file_path.exists():
            results.append({
                "file": label,
                "found": False,
                "lines": [],
                "summary": f"File not found: {file_path}",
            })
            continue

        hits = _grep_file(file_path, pattern)
        if hits:
            rel = str(file_path.relative_to(repo_root))
            results.append({
                "file": rel,
                "found": True,
                "lines": hits[:5],
                "summary": _summarize_hits(hits, label),
            })
        else:
            rel = str(file_path.relative_to(repo_root))
            results.append({
                "file": rel,
                "found": False,
                "lines": [],
                "summary": "Not present",
            })

    # Derivatives binding analysis
    deriv_binding = _analyze_derivatives_binding(search_name, paths["derivatives"])

    # H17: M-adapt Step 0 state classification (A/B/C/D/E)
    m_adapt_state = _classify_m_adapt_state(search_name, paths)

    return {
        "op_name": op_name,
        "search_name": search_name,
        "files": results,
        "derivatives_binding": deriv_binding,
        "m_adapt_state": m_adapt_state,
    }


def _extract_yaml_entry_block(yaml_content: str, op_name: str) -> str | None:
    """Extract the YAML list-entry block for op_name from op_plugin_functions.yaml.

    The YAML is a top-level list where each entry starts with ``- func: op_name(...)``.
    We return the text between that ``- func:`` line and the next ``- func:`` (or EOF),
    so flags like op_api:/acl_op:/gen_opapi: are checked only within this operator's
    entry (H17 fix: whole-file regex gave false positives when any other op had flags).
    """
    lines = yaml_content.split("\n")
    n = len(lines)
    start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not (stripped.startswith("- func:") or stripped.startswith("func:")):
            continue
        func_val = stripped.replace("- func:", "").replace("func:", "").strip()
        if func_val.startswith(op_name + "("):
            # Resolve multi-line func signature (balanced parens may span lines)
            j = i
            while func_val.count("(") > func_val.count(")") and j + 1 < n:
                j += 1
                nxt = lines[j].strip()
                if nxt.startswith("- ") or nxt.startswith("func:"):
                    break
                func_val += " " + nxt
            start = i
            break
    if start is None:
        return None
    # Find end: next line whose stripped content starts with "- func:" or "- name:" (top-level list item)
    end = n
    for k in range(start + 1, n):
        s = lines[k].lstrip()
        if s.startswith("- func:") or s.startswith("- name:") or s.startswith("- backward:"):
            end = k
            break
    return "\n".join(lines[start:end])


def _classify_m_adapt_state(op_name: str, paths: dict) -> str | None:
    """Classify operator's M-adapt Step 0 state (A/B/C/D/E).

    Per m-adapt-guide.md §2:
    A = hand-written OpApi already (op_api: + opapi/*.cpp exists)
    B = gen_opapi structured adapter (gen_opapi: sub-block)
    C = genuine candidate (acl_op: only, no op_api:, aclops/*.cpp exists)
    D = nothing to modify (absent from YAML or no aclops cpp)
    E = A5-only (op_api: only, no acl_op:, no aclops cpp)
    """
    if not paths["yaml"].exists():
        return None
    try:
        yaml_content = paths["yaml"].read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError, OSError):
        return None

    entry = _extract_yaml_entry_block(yaml_content, op_name)
    if entry is None:
        # Op not present in YAML at all — state D (nothing to do)
        return "D"

    has_op_api = bool(re.search(r'^\s*op_api:', entry, re.MULTILINE) or
                      re.search(r'^\s*internal_format_opapi:', entry, re.MULTILINE))
    has_acl_op = bool(re.search(r'^\s*acl_op:', entry, re.MULTILINE))
    has_gen_opapi = bool(re.search(r'^\s*gen_opapi:', entry, re.MULTILINE))

    def _dir_contains_op(dir_path: Path) -> bool:
        if not dir_path.exists():
            return False
        for f in dir_path.glob("*.cpp"):
            try:
                txt = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            if op_name in txt:
                return True
        return False

    has_opapi_cpp = _dir_contains_op(paths["opapi_dir"])

    # Check for aclops cpp
    aclops_dir = paths["root"] / "op_plugin" / "ops" / "aclops"
    has_aclops_cpp = _dir_contains_op(aclops_dir)

    if has_gen_opapi:
        return "B"
    if has_op_api and has_opapi_cpp:
        return "A"
    if has_acl_op and not has_op_api and has_aclops_cpp:
        return "C"
    if not has_acl_op and has_op_api and not has_aclops_cpp:
        return "E"
    if has_acl_op and not has_aclops_cpp:
        return "D"
    return None


def _grep_file(file_path: Path, pattern: str) -> list[dict]:
    """Grep for a pattern in a single file, return matching lines.

    Uses word-boundary matching to avoid substring matches
    (e.g. npu_dynamic_quant should NOT match npu_dynamic_quant_asymmetric).
    Matches the pattern followed by a non-identifier char (not letter/digit/_)
    or end of line — this way _v2 variants ARE found but _asymmetric is NOT.
    """
    try:
        content = file_path.read_text()
    except Exception:
        return []

    # Build regex: pattern followed by non-identifier char or _v<digit> (version suffix) or EOL
    # This matches "npu_op(" and "npu_op_v2(" but NOT "npu_op_other("
    regex = re.compile(
        re.escape(pattern) + r'(?:_backward|_grad)?' + r'(?=$|[^a-zA-Z0-9_]|_v\d)'
    )

    hits = []
    for i, line in enumerate(content.split("\n"), 1):
        if regex.search(line):
            hits.append({"line": i, "text": line.strip()[:120]})
    return hits


def _grep_directory(dir_path: Path, pattern: str, glob: str) -> dict:
    """Grep for a pattern across all files in a directory matching glob."""
    results = {}
    if not dir_path.exists():
        return results

    regex = re.compile(
        re.escape(pattern) + r'(?:_backward|_grad)?' + r'(?=$|[^a-zA-Z0-9_]|_v\d)'
    )

    for f in sorted(dir_path.glob(glob)):
        try:
            content = f.read_text()
        except Exception:
            continue
        lines = []
        for i, line in enumerate(content.split("\n"), 1):
            if regex.search(line):
                lines.append({"line": i, "text": line.strip()[:120]})
        if lines:
            results[str(f)] = lines
    return results


def _summarize_hits(hits: list[dict], label: str) -> str:
    """Create a one-line summary of grep hits."""
    if not hits:
        return "Not present"
    first = hits[0]
    return f"L{first['line']}: {first['text'][:80]}"


def _analyze_derivatives_binding(op_name: str, deriv_path: Path) -> dict | None:
    """Analyze derivatives.yaml binding direction for the operator.

    Reuses parse_derivatives_entry (signature.py) for the forward case so the
    `formulas` field is populated consistently (single source of truth — avoids
    the duplicate formula-reading loop that previously dropped formulas on blank
    lines). The backward-referenced case is unique to discover.
    """
    # Forward case: op is a `- name:` entry with backward formulas.
    entry = parse_derivatives_entry(op_name, deriv_path)
    if entry:
        return {
            "direction": "forward_has_backward",
            "name_line": entry.get("name_line"),
            "line_number": entry.get("line_number"),
            "formulas": entry.get("formulas", {}),
        }

    # Backward case: op is referenced as a backward op in some forward's formula.
    if not deriv_path.exists():
        return None
    content = deriv_path.read_text()
    backward_pattern = re.compile(r'\b' + re.escape(op_name) + r'\b')
    for line in content.split("\n"):
        if backward_pattern.search(line) and ':' in line:
            stripped = line.strip()
            if stripped and not stripped.startswith("- name:") and not stripped.startswith("version:"):
                return {
                    "direction": "backward_referenced_by_forward",
                    "reference_line": stripped[:120],
                }

    return None
