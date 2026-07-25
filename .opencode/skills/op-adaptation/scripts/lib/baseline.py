"""C-baseline: generate schema JSON + allowlist entries.

Default --dry-run outputs diff. --write modifies files atomically via temp-file + os.replace.
"""

import json
import re
import os
import tempfile
from pathlib import Path
from typing import Any

from .common import find_repo_root, repo_paths
from .signature import find_yaml_func_raw


def run_gen_baseline(op_name: str, repo_root: Path | None = None, write: bool = False,
                     allowlist_list: str | None = None) -> dict:
    """Generate schema JSON + allowlist entries for an operator.

    Args:
        op_name: Operator name (e.g. npu_dynamic_quant)
        repo_root: Repository root (auto-detected if None)
        write: If True, atomically modify target files. If False, dry-run output only.
        allowlist_list: Target allowlist list name (e.g. "all_version", "v2.1.0").
            When set, forces the allowlist write into that list even for
            version-specific ops — the LLM decided placement, the script does the
            safe formatting-preserving insert. NEVER hand-edit the JSON.

    Returns:
        {
            "op_name": str,
            "yaml_func_line": str,
            "schema_entries": [...],
            "allowlist_entry": str,
            "dry_run": bool,
            "written": bool,
        }
    """
    if repo_root is None:
        repo_root = find_repo_root()
    paths = repo_paths(repo_root)

    # 1. Read YAML func: line verbatim (preserves '*' separators and defaults)
    func_line = find_yaml_func_raw(op_name, paths["yaml"])
    if not func_line:
        return {"op_name": op_name, "error": f"Operator '{op_name}' not found in op_plugin_functions.yaml"}

    # Determine version from YAML op_api field
    version = _read_yaml_version(op_name, paths["yaml"])

    # Allowlist placement: all_version ops go to the all_version list; version-
    # specific ops (any vX.Y tag) are NOT auto-added to all_version — the real
    # allowlist keeps them in version-specific lists, so the LLM must choose.
    allowlist_placement = _allowlist_placement(version)

    # 2. Generate schema JSON entries
    schema_entries = [
        {
            "key": f"func: {func_line}",
            "version": version,
        },
        {
            "key": f"op_api: torch_npu.{op_name}(*args, **kwargs)",
            "version": version,
        },
    ]

    # 3. Generate allowlist entry
    allowlist_entry = op_name

    result = {
        "op_name": op_name,
        "yaml_func_line": func_line,
        "version": version,
        "schema_entries": schema_entries,
        "allowlist_entry": allowlist_entry,
        "allowlist_placement": allowlist_placement,
        "dry_run": not write,
        "written": False,
        "allowlist_written": False,
    }

    if write:
        result["schema_entries_written"] = _write_schema_json(schema_entries, paths["schema_json"])
        if allowlist_list:
            # LLM explicitly chose the target list — do the safe formatting-preserving
            # insert regardless of version-specific gating.
            result["allowlist_written"] = _write_allowlist(
                allowlist_entry, paths["allowlist_json"], allowlist_list)
            result["allowlist_target_list"] = allowlist_list
        elif allowlist_placement == "all_version":
            result["allowlist_written"] = _write_allowlist(allowlist_entry, paths["allowlist_json"])
        # version-specific w/o --allowlist-list: skip allowlist write — LLM must
        # re-run with --allowlist-list <name>, NEVER hand-edit the JSON.
        result["written"] = True
    else:
        # Dry-run: show what would be added
        result["diff"] = _generate_diff(schema_entries, allowlist_entry, paths, allowlist_placement)

    return result


def _allowlist_placement(version: list) -> str:
    """Decide allowlist placement from the schema version range.

    Returns 'all_version' (auto-write to all_version list) or
    'version_specific_review_needed' (any vX.Y tag — the real allowlist keeps
    version-specific ops out of all_version, so the LLM must choose the list).
    """
    if any(re.search(r'^v\d', v) for v in version):
        return "version_specific_review_needed"
    return "all_version"


def _read_yaml_version(op_name: str, yaml_path: Path) -> list:
    """Read the op_api version range for an operator from YAML."""
    content = yaml_path.read_text()
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if f"func: {op_name}(" in line:
            # Look for op_api in next few lines
            for j in range(i, min(i + 5, len(lines))):
                stripped = lines[j].strip()
                if stripped.startswith("op_api:"):
                    val = stripped.replace("op_api:", "").strip()
                    if val == "all_version":
                        return ["all_version"]
                    # Parse [v2.1, newest] format
                    val = val.strip("[]")
                    return [v.strip().strip("'\"") for v in val.split(",")]
            break
    return ["newest"]  # default


def _insert_schema_entry_text(raw: str, key: str, version: list) -> tuple[str, bool]:
    """Insert one schema entry via targeted text edit (preserves formatting).

    Returns (new_text, inserted). No-op if the key already exists. The schema
    JSON is a flat object `{\n  "key": {\n    "version": [...]\n  },...\n}`;
    we append before the final top-level '}' instead of round-tripping, which
    would reformat every inline array into multi-line (huge git noise).
    """
    needle = json.dumps(key, ensure_ascii=False)  # the key as a JSON string literal
    # Check if key exists as a top-level key (not as a value substring).
    # Top-level keys appear as `  "key": {` at the start of a line (H12 fix).
    if re.search(r'^\s*' + re.escape(needle) + r'\s*:', raw, re.MULTILINE):
        return raw, False  # already present as a top-level key
    ver_inline = '[' + ', '.join(f'"{v}"' for v in version) + ']'
    entry = f'  {needle}: {{\n    "version": {ver_inline}\n  }}'
    stripped = raw.rstrip()
    if not stripped.endswith('}'):
        return raw, False  # unexpected format, bail rather than corrupt
    inner = stripped[:-1].rstrip()  # drop the top-level '}'
    if inner.endswith('{'):
        return '{\n' + entry + '\n}', True  # was empty object
    return inner + ',\n' + entry + '\n}', True


def _insert_allowlist_op_text(raw: str, op_name: str, target_list: str = "all_version") -> tuple[str, bool]:
    """Append op_name to torch_npu.<target_list> via targeted line edit.

    Returns (new_text, inserted). No-op if already present or no such list.
    Preserves the file's existing one-op-per-line array formatting.
    """
    lines = raw.split('\n')
    av_start = None
    for i, l in enumerate(lines):
        if f'"{target_list}"' in l and '[' in l:
            av_start = i
            break
    if av_start is None:
        return raw, False
    # find closing ']' (first line whose stripped content is ']' or '],')
    close_idx = None
    for j in range(av_start + 1, len(lines)):
        s = lines[j].strip()
        if s in (']', '],'):
            close_idx = j
            break
    if close_idx is None:
        return raw, False
    needle = f'"{op_name}"'
    needle_comma = f'"{op_name}",'
    for j in range(av_start + 1, close_idx):
        stripped = lines[j].strip()
        # Exact-match guard (H12 fix): prevent false positive when op_name is a
        # prefix of an existing entry (e.g. inserting "npu_add" should not be
        # blocked by existing "npu_add_v2"). Match only when the stripped line
        # is exactly the quoted op name, with optional trailing comma.
        if stripped == needle or stripped == needle_comma:
            return raw, False  # already present
    # element indent from the first element line (default 6-space)
    elem_indent = '      '
    has_elements = False
    for j in range(av_start + 1, close_idx):
        if lines[j].strip().startswith('"'):
            has_elements = True
            elem_indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
            break
    new_lines = list(lines)
    if has_elements:
        last_elem = None
        for j in range(close_idx - 1, av_start, -1):
            if lines[j].strip().startswith('"'):
                last_elem = j
                break
        if not new_lines[last_elem].rstrip().endswith(','):
            new_lines[last_elem] = new_lines[last_elem].rstrip() + ','
        new_lines.insert(close_idx, f'{elem_indent}"{op_name}"')
    else:
        new_lines.insert(close_idx, f'{elem_indent}"{op_name}"')
    return '\n'.join(new_lines), True


def _atomic_write(path: Path, content: str):
    """Write content atomically: temp file + os.replace (H13 fix).

    Avoids .bak overwrite on multi-op sequential writes. On POSIX, os.replace
    is atomic — the file is either old or new, never partially written."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except:
        os.unlink(tmp_path)
        raise


def _write_schema_json(entries: list[dict], schema_path: Path) -> int:
    """Write entries to torch_npu_OpApi_schema_all.json (targeted, preserves formatting).

    Returns the number of entries actually inserted (skips already-present keys).
    """
    raw = schema_path.read_text()
    new_raw = raw
    inserted = 0
    for entry in entries:
        new_raw, did = _insert_schema_entry_text(new_raw, entry["key"], entry["version"])
        if did:
            inserted += 1
    if inserted:
        _atomic_write(schema_path, new_raw)
    return inserted


def _write_allowlist(op_name: str, allowlist_path: Path, target_list: str = "all_version") -> bool:
    """Append op_name to allowlist torch_npu.<target_list> (targeted, preserves formatting).

    Returns True if inserted, False if already present / no such list.
    """
    raw = allowlist_path.read_text()
    new_raw, inserted = _insert_allowlist_op_text(raw, op_name, target_list)
    if inserted:
        _atomic_write(allowlist_path, new_raw)
    return inserted


def _load_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _generate_diff(entries: list[dict], allowlist_entry: str, paths: dict, allowlist_placement: str = "all_version") -> str:
    """Generate a human-readable diff for dry-run mode.

    Checks existing entries so already-present keys are marked (the --write
    path skips them via `if key not in data`); this keeps the dry-run honest.
    """
    lines = ["=== Dry-run: would add the following entries ===\n"]

    schema_data = _load_json_safe(paths["schema_json"])
    lines.append(f"Schema JSON ({paths['schema_json'].name}):")
    for entry in entries:
        ver = entry["version"]
        if isinstance(ver, list):
            ver_json = "[" + ", ".join(f'"{v}"' for v in ver) + "]"
        else:
            ver_json = f'["{ver}"]'
        status = "(already present, will skip)" if entry["key"] in schema_data else ""
        lines.append(f'  + "{entry["key"]}": {{"version": {ver_json}}} {status}'.rstrip())

    allow_data = _load_json_safe(paths["allowlist_json"])
    all_ver = allow_data.get("torch_npu", {}).get("all_version", [])
    lines.append(f"\nAllowlist ({paths['allowlist_json'].name}):")
    if allowlist_placement == "all_version":
        al_status = "(already present, will skip)" if allowlist_entry in all_ver else ""
        lines.append(f'  + "{allowlist_entry}" → torch_npu.all_version {al_status}'.rstrip())
    else:
        # List the available lists so the LLM can pick one for --allowlist-list.
        available = [k for k in allow_data.get("torch_npu", {}).keys()]
        lines.append(f'  ! "{allowlist_entry}" is version-specific — NOT auto-added.')
        lines.append(f'    → Re-run with --allowlist-list <name> to insert safely (preserves formatting).')
        lines.append(f'    → Available lists: {", ".join(available) if available else "(none found)"}')
        lines.append(f'    ⚠ NEVER hand-edit the allowlist JSON (json.dump reformats the whole file).')

    return "\n".join(lines)
