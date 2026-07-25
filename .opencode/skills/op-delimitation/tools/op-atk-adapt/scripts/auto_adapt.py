#!/usr/bin/env python3
"""Auto-adapt CANN ATK -> PTA ATK: copy to PTA dir, extract opName + interface prototype, patch node.yaml.

Automates:
  1  Get opName from CHECKLIST.md (torch_npu function name:)
  2  Create {project_root}/{opName}_pta_atk/ and copy the deliverable files there
     (4-file mode: atk.yaml/generate.py/executor.py/node.yaml;
      3-file mode: pre-generated case *.json + executor.py + node.yaml)
  3  Find interface prototype from codegen/templates/_op_plugin_docs.py
  4  Patch node*.yaml in the PTA dir: backend pyaclnn -> npu
  5  Locate the executor class to adapt (the one inheriting BaseApi) + api_type

Code generation (editing the class's __call__ method) is left to the LLM.

Usage:
    python auto_adapt.py <cann_dir> [--project-root <root>] [--dry-run]
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def find_project_root(start):
    """Walk upward until we find codegen/templates/_op_plugin_docs.py."""
    d = os.path.abspath(start)
    while d and d != os.sep:
        if os.path.isfile(os.path.join(d, "codegen", "templates", "_op_plugin_docs.py")):
            return d
        d = os.path.dirname(d)
    return None


def resolve_torch_path(override=None):
    """Locate the torch install dir (contains _torch_docs.py / _tensor_docs.py).

    Order: --torch-path override -> in-process import -> subprocess python -c.
    Returns dir path or None.
    """
    if override:
        return override if os.path.isdir(override) else None

    try:
        import torch  # noqa: PLC0415
        return torch.__path__[0]
    except Exception:
        pass

    # Run from a neutral cwd so a local ./torch dir can't shadow the package.
    neutral = os.sep if os.path.isdir(os.sep) else None
    for py in ("python", "python3"):
        try:
            proc = subprocess.run(
                [py, "-c", "import torch; print(torch.__path__[0])"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=30, cwd=neutral,
            )
        except Exception:
            continue
        out = proc.stdout.decode("utf-8", "ignore").strip()
        if proc.returncode == 0 and out and os.path.isdir(out):
            return out
        # import failed — but the traceback still leaks the real torch path
        # (broken/mixed conda env). Parse it out of the combined output.
        combined = out + "\n" + proc.stderr.decode("utf-8", "ignore")
        path = _torch_dir_from_traceback(combined)
        if path:
            return path
    return None


def _torch_dir_from_traceback(text):
    """Extract the torch package dir from an import-failure traceback.

    Prefer the frame `File ".../torch/__init__.py"` (the module we tried to
    import); fall back to any `.../torch/<file>.py` frame. Returns dir or None.
    """
    m = re.search(r'File "([^"]*[/\\]torch)[/\\]__init__\.py"', text)
    if m and os.path.isdir(m.group(1)):
        return m.group(1)
    for fm in re.finditer(r'File "([^"]*[/\\]torch)[/\\][^"]*\.py"', text):
        if os.path.isdir(fm.group(1)):
            return fm.group(1)
    return None


# ---------------------------------------------------------------------------
# Step 2.1: Extract opName from CHECKLIST.md
# ---------------------------------------------------------------------------

def extract_opname(project_root):
    """Return (opname, error_message)."""
    checklist = os.path.join(project_root, "CHECKLIST.md")
    if not os.path.isfile(checklist):
        return None, f"CHECKLIST.md not found at {checklist}"

    text = read(checklist)

    # Look for "torch_npu function name:" (with flexible whitespace / colon variants)
    m = re.search(r"torch_npu\s*function\s*name\s*[:：]\s*(\S+)", text)
    if m:
        # Strip markdown formatting (backticks, bold asterisks) that may wrap the name in CHECKLIST.md
        opname = m.group(1).strip("`*")
        return opname, None

    return None, "'torch_npu function name:' not found in CHECKLIST.md"


# ---------------------------------------------------------------------------
# Input mode detection: 4-file (atk.yaml + generate.py + ...) vs 3-file
# (pre-generated case *.json + executor*.py + node*.yaml, no atk.yaml)
# ---------------------------------------------------------------------------

def _is_atk_case_yaml(fname):
    """True when fname is an ATK case yaml (bare or with an `atk` token).

    Matches `atk.yaml`/`atk.yml`, or any `.yaml`/`.yml` whose stem contains
    `atk` as an underscore-delimited token (suffix `_atk`, prefix `atk_`,
    infix `_atk_`). Excludes node*/executor*/generate* — those are not the
    case yaml.
    """
    lower = fname.lower()
    if not lower.endswith((".yaml", ".yml")):
        return False
    if lower.startswith(("node", "executor", "generate")):
        return False
    stem = lower.rsplit(".", 1)[0]
    tokens = stem.split("_")
    return stem == "atk" or "atk" in tokens


def detect_mode(cann_dir):
    """Return (mode, json_files) for the input directory.

    mode: "4file" — atk.yaml present (cases still need `atk case` generation)
          "3file" — top-level *.json + executor*.py + node*.yaml (cases pre-generated)
    Exits with an error when neither file set is complete (never silently
    falls back to one of the modes).
    """
    top_files = [
        item for item in sorted(os.listdir(cann_dir))
        if os.path.isfile(os.path.join(cann_dir, item))
    ]

    # 4-file mode: a case yaml is present. Accept the bare `atk.yaml` and any
    # real-world deliverable whose name carries `atk` as an underscore-delimited
    # token — `<op>_atk.yaml` (suffix), `atk_<x>.yaml` (prefix), `<a>_atk_<b>.yaml`
    # (infix). node*/executor*/generate* yamls are excluded — they are not the
    # case yaml.
    if any(_is_atk_case_yaml(f) for f in top_files):
        return "4file", []

    json_files = [f for f in top_files if f.endswith(".json")]
    has_executor = any(f.startswith("executor") and f.endswith(".py") for f in top_files)
    has_node = any(f.startswith("node") and f.endswith((".yaml", ".yml")) for f in top_files)
    if json_files and has_executor and has_node:
        return "3file", json_files

    sys.exit(
        f"Cannot detect input mode in {cann_dir}:\n"
        "  4-file mode requires : atk.yaml (+ generate.py, executor*.py, node*.yaml)\n"
        "  3-file mode requires : a case *.json + executor*.py + node*.yaml\n"
        f"  found                : {', '.join(top_files) or '(empty directory)'}"
    )


# ---------------------------------------------------------------------------
# Step 2: Create PTA dir and copy deliverable files
# ---------------------------------------------------------------------------

def prepare_pta_dir(project_root, opname, cann_dir, mode, dry_run=False):
    """Create {project_root}/{opname}_pta_atk and copy deliverable files from cann_dir.

    Copies only .yaml and .py files (the 4-file deliverable); mode "3file"
    additionally copies the pre-generated case .json files.
    Returns (pta_dir, copied_files, error_message).
    """
    pta_dir = os.path.join(project_root, f"{opname}_pta_atk")

    exts = (".yaml", ".yml", ".py")
    if mode == "3file":
        exts += (".json",)

    files_to_copy = [
        item for item in sorted(os.listdir(cann_dir))
        if os.path.isfile(os.path.join(cann_dir, item))
        and item.endswith(exts)
    ]

    if dry_run:
        return pta_dir, files_to_copy, None

    os.makedirs(pta_dir, exist_ok=True)
    copied = []
    for item in files_to_copy:
        shutil.copy2(
            os.path.join(cann_dir, item),
            os.path.join(pta_dir, item),
        )
        copied.append(item)

    return pta_dir, copied, None


# ---------------------------------------------------------------------------
# Step 2.2: Find interface prototype in _op_plugin_docs.py
# ---------------------------------------------------------------------------

def parse_signature(sig_line):
    """Parse a function-signature line into (func_name, [param_dicts]).

    Each param dict:
        name, type, is_tensor, is_optional, default

    Example:
        "torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)"
        -> ("torch_npu.npu_swiglu", [
             {"name":"self","type":"Tensor","is_tensor":True,
              "is_optional":False,"default":None},
             {"name":"dim","type":"int","is_tensor":False,
              "is_optional":False,"default":"-1"},
           ])
    """
    sig = sig_line.strip()

    # Find function name (up to first '(')
    paren_open = sig.find("(")
    if paren_open == -1:
        return None, []
    func_name = sig[:paren_open].strip()

    # Find the MATCHING closing ')' for params (handles nested parens like out=(var, m, v))
    depth = 0
    paren_close = -1
    for i in range(paren_open, len(sig)):
        if sig[i] == "(":
            depth += 1
        elif sig[i] == ")":
            depth -= 1
            if depth == 0:
                paren_close = i
                break
    if paren_close == -1:
        return None, []

    params_str = sig[paren_open + 1 : paren_close].strip()
    if not params_str:
        return func_name, []

    # Split on commas that are NOT inside parentheses
    depth = 0
    parts = []
    cur = ""
    for ch in params_str:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())

    params = []
    for part in parts:
        # Forms: "Tensor self" | "Tensor? offset=None" | "int dim=-1" | "self" | "keepdim=False"
        pm = re.match(r"(\w+)(\??)\s+(\w+)(?:\s*=\s*(.+))?$", part)
        if pm:
            raw_type = pm.group(1)
            is_optional = pm.group(2) == "?"
            params.append({
                "name": pm.group(3),
                "type": raw_type,
                "is_tensor": raw_type == "Tensor",
                "is_optional": is_optional,
                "default": pm.group(4).strip() if pm.group(4) else None,
            })
        else:
            # Fallback: bare param without type prefix: "self" | "keepdim=False"
            pm = re.match(r"(\w+)(?:\s*=\s*(.+))?$", part)
            if pm:
                params.append({
                    "name": pm.group(1),
                    "type": None,          # unknown — resolved via docstring fallback
                    "is_tensor": False,    # tentative; corrected by _merge_docstring_types
                    "is_optional": False,
                    "default": pm.group(2).strip() if pm.group(2) else None,
                })

    return func_name, params


def parse_docstring_param_types(lines, block_start, block_end):
    """Scan the docstring block for a '参数说明' section and return {name: type}.

    Matches lines like:
        self (Tensor) - 输入张量。
        dim (Int) - 待降低维度。
        keepdim (Bool，默认值为False) - 输出张量是否保留dim。
    """
    type_map = {}
    in_params = False
    for i in range(block_start, block_end + 1):
        line = lines[i].strip()
        if "参数说明" in line or "参数" == line[:2]:
            in_params = True
            continue
        if in_params:
            # End of params section: blank line or a new section header
            if line and not re.match(r"^\w+\s*\(", line):
                break
            m = re.match(r"(\w+)\s*\(\s*(\w+)", line)
            if m:
                type_map[m.group(1)] = m.group(2)
    return type_map


def find_prototype(project_root, opname):
    """Return (signature_line, func_name, params, error_message)."""
    docs = os.path.join(project_root, "codegen", "templates", "_op_plugin_docs.py")
    if not os.path.isfile(docs):
        return None, None, None, "_op_plugin_docs.py not found"

    lines = read(docs).splitlines()

    # 1. Find _add_torch_npu_docstr("opname", ...) block start
    block_start = None
    for i, line in enumerate(lines):
        if "_add_torch_npu_docstr" in line:
            chunk = "\n".join(lines[i : i + 4])
            if f'"{opname}"' in chunk or f"'{opname}'" in chunk:
                block_start = i
                break

    if block_start is None:
        return None, None, None, f'opName "{opname}" not found in _op_plugin_docs.py'

    # 2. Find block end (matching closing parenthesis)
    depth = 0
    block_end = block_start
    for i in range(block_start, len(lines)):
        depth += lines[i].count("(") - lines[i].count(")")
        if i > block_start and depth <= 0:
            block_end = i
            break

    # 3. Search for "接口原型" within the block
    sig_line = func_name = params = None
    for j in range(block_start, block_end + 1):
        if "接口原型" in lines[j]:
            # Check if signature is on the same line after 接口原型:xxxxx
            rest = lines[j].split("接口原型", 1)[1]
            rest = rest.lstrip(":： \t").strip()
            if rest and "(" in rest:
                func_name, params = parse_signature(rest)
                if func_name:
                    sig_line = rest
                    break

            # Otherwise scan next non-empty lines for the signature
            for k in range(j + 1, min(j + 5, block_end + 1)):
                candidate = lines[k].strip()
                if candidate and "(" in candidate:
                    func_name, params = parse_signature(candidate)
                    if func_name:
                        sig_line = candidate
                        break
            if sig_line:
                break

    # 4. Fallback: some ops have no "接口原型" header — the signature is the
    #    first non-empty line of the docstring (e.g. npu_apply_adam)
    if sig_line is None:
        for k in range(block_start + 2, block_end + 1):
            candidate = lines[k].strip()
            if candidate and "(" in candidate and "功能" not in candidate:
                func_name, params = parse_signature(candidate)
                if func_name and func_name.endswith(opname):
                    sig_line = candidate
                    break

    if sig_line is None:
        return None, None, None, f"接口原型 not found in block (lines {block_start+1}-{block_end+1})"

    # 5. If params have unknown types (bare signatures like "self, dim, keepdim=False"),
    #    merge types from the docstring "参数说明" section.
    if any(p["type"] is None for p in params):
        type_map = parse_docstring_param_types(lines, block_start, block_end)
        for p in params:
            if p["type"] is None and p["name"] in type_map:
                p["type"] = type_map[p["name"]]
                p["is_tensor"] = type_map[p["name"]] == "Tensor"

    return sig_line, func_name, params, None


# ---------------------------------------------------------------------------
# Native PyTorch ops: parse signature from _torch_docs.py / _tensor_docs.py
# ---------------------------------------------------------------------------

# Conservative tensor-name heuristic for bare (un-annotated) torch params.
_TENSOR_NAME_HINTS = {
    "input", "self", "other", "tensor", "tensors", "mat", "mat1", "mat2",
    "batch1", "batch2", "vec", "vec1", "vec2", "weight", "bias", "target",
    "grad", "grad_output", "indices", "index", "src",
}


def parse_torch_signature(sig_line):
    """Parse a torch docstring signature (type AFTER name) into (func_name, params).

    Param dict matches parse_signature exactly: name/type/is_tensor/is_optional/default.

    Handles:
        "absolute(input: Tensor, *, out: Optional[Tensor]) -> Tensor"
        "addmm(input, mat1, mat2, out_dtype=None, *, beta=1, out=None) -> Tensor"
        line-continuation backslashes.
    """
    sig = re.sub(r"\\\s*\n\s*", " ", sig_line).strip()

    paren_open = sig.find("(")
    if paren_open == -1:
        return None, []
    func_name = sig[:paren_open].strip()

    # Matching closing paren (handles nested brackets in annotations).
    depth = 0
    paren_close = -1
    for i in range(paren_open, len(sig)):
        if sig[i] in "([":
            depth += 1
        elif sig[i] in ")]":
            depth -= 1
            if depth == 0 and sig[i] == ")":
                paren_close = i
                break
    if paren_close == -1:
        return None, []

    params_str = sig[paren_open + 1 : paren_close].strip()
    if not params_str:
        return func_name, []

    # Split top-level commas (ignore commas inside []/()).
    depth = 0
    parts = []
    cur = ""
    for ch in params_str:
        if ch in "([":
            depth += 1
            cur += ch
        elif ch in ")]":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())

    params = []
    for part in parts:
        if part in ("*", "/", ""):  # keyword/positional separators
            continue
        name = anno = default = None
        # name: Type = default  |  name: Type
        m = re.match(r"(\*{0,2}\w+)\s*:\s*([^=]+?)(?:\s*=\s*(.+))?$", part)
        if m:
            name, anno = m.group(1), m.group(2).strip()
            default = m.group(3).strip() if m.group(3) else None
        else:
            # name = default  |  name
            m = re.match(r"(\*{0,2}\w+)(?:\s*=\s*(.+))?$", part)
            if not m:
                continue
            name = m.group(1)
            default = m.group(2).strip() if m.group(2) else None

        name = name.lstrip("*")
        if anno:
            is_tensor = "Tensor" in anno
            is_optional = "Optional" in anno
            ptype = anno
        else:
            # bare param: conservative name heuristic, flagged for LLM review
            is_tensor = name in _TENSOR_NAME_HINTS
            is_optional = default == "None"
            ptype = None
        params.append({
            "name": name,
            "type": ptype,
            "is_tensor": is_tensor,
            "is_optional": is_optional,
            "default": default,
        })

    return func_name, params


def _find_docstr_signature(text, pattern):
    """Given file text and a regex locating the add_docstr entry, return the
    first non-empty docstring line (the signature), or None."""
    m = re.search(pattern, text)
    if not m:
        return None
    # Find the docstring opening triple-quote after the match.
    q = re.search(r'r?"""', text[m.end():])
    if not q:
        return None
    start = m.end() + q.end()
    lines = text[start:].splitlines()
    sig = None
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        # First non-empty line = start of signature. Accumulate continuation
        # lines while it ends with '\' or the parens are still unbalanced.
        sig = line
        j = idx
        while sig.endswith("\\") or (sig.count("(") > sig.count(")")):
            sig = sig.rstrip("\\").rstrip()
            j += 1
            if j >= len(lines):
                break
            sig += " " + lines[j].strip()
        break
    return sig


def find_torch_prototype(opname, torch_path):
    """Return (sig_line, func_name, params, source, error) for a native torch op.

    Scans _torch_docs.py (torch.<op>) first, then _tensor_docs.py ("<op>").
    """
    if not torch_path:
        return None, None, None, None, "torch path not found (import torch failed)"

    candidates = [
        ("_torch_docs.py", r"add_docstr\(\s*torch\.%s\s*," % re.escape(opname)),
        ("_tensor_docs.py", r"add_docstr_all\(\s*[\"']%s[\"']\s*," % re.escape(opname)),
    ]
    for fname, pattern in candidates:
        fpath = os.path.join(torch_path, fname)
        if not os.path.isfile(fpath):
            continue
        sig_line = _find_docstr_signature(read(fpath), pattern)
        if sig_line:
            func_name, params = parse_torch_signature(sig_line)
            if func_name:
                return sig_line, func_name, params, fname, None

    return None, None, None, None, \
        '"%s" not found in _torch_docs.py / _tensor_docs.py' % opname


# ---------------------------------------------------------------------------
# Patch node*.yaml: pyaclnn -> npu
# ---------------------------------------------------------------------------

def patch_node_yaml(cann_dir, dry_run=False):
    """Return (patched, already_npu, no_match, error_message).

    - patched:    files where pyaclnn was found and replaced
    - already_npu: files already containing backend: npu
    - no_match:   files with neither pyaclnn nor npu (reported as WARNING)
    """
    files = sorted(set(
        glob.glob(os.path.join(cann_dir, "node*.yaml"))
        + glob.glob(os.path.join(cann_dir, "nodes*.yaml"))
        + glob.glob(os.path.join(cann_dir, "node*.yml"))
        + glob.glob(os.path.join(cann_dir, "nodes*.yml"))
    ))
    if not files:
        return [], [], [], "No node*.yaml found in " + cann_dir

    patched = []
    already_npu = []
    no_match = []
    for nf in files:
        text = read(nf)
        new_text = re.sub(
            r"backend:\s*[\"']?pyaclnn[\"']?", "backend: npu", text, flags=re.IGNORECASE
        )
        if new_text != text:
            if not dry_run:
                with open(nf, "w", encoding="utf-8") as f:
                    f.write(new_text)
            patched.append(os.path.basename(nf))
        else:
            if re.search(r"backend:\s*npu", text, re.IGNORECASE):
                already_npu.append(os.path.basename(nf))
            else:
                no_match.append(os.path.basename(nf))
    return patched, already_npu, no_match, None


# ---------------------------------------------------------------------------
# Resolve api_type (unified for both modes)
# ---------------------------------------------------------------------------

def parse_yaml_api_type(cann_dir):
    """Return (api_type, source, error_message).

    Single priority chain, mode-agnostic:
      1. case yaml ``api_type:`` field (atk.yaml — present in 4-file mode only)
      2. executor*.py ``@register("<api_type>")``
      3. case *.json content (``"api_type"``/``"op_type"`` field)
    4-file mode hits level 1; 3-file mode has no case yaml and falls through
    to level 2/3. If all fail, tell the LLM to determine it manually
    (warning only, non-blocking).
    """
    yamls = sorted(
        glob.glob(os.path.join(cann_dir, "*.yaml"))
        + glob.glob(os.path.join(cann_dir, "*.yml"))
    )
    # Filter out node*/executor*/generator* files
    yamls = [y for y in yamls if not os.path.basename(y).startswith(("node", "executor", "generate"))]

    for yf in yamls:
        text = read(yf)
        m = re.search(r"^\s*api_type\s*:\s*(\S+)\s*$", text, re.M)
        if m:
            return m.group(1), os.path.basename(yf), None

    for ef in sorted(glob.glob(os.path.join(cann_dir, "executor*.py"))):
        m = re.search(r'@register\(\s*["\']([^"\']+)["\']', read(ef))
        if m:
            return m.group(1), os.path.basename(ef) + " (@register)", None

    for jf in sorted(glob.glob(os.path.join(cann_dir, "*.json"))):
        m = re.search(r'"(?:api_type|op_type)"\s*:\s*"([^"]+)"', read(jf))
        if m:
            return m.group(1), os.path.basename(jf), None

    return None, None, (
        "api_type not found in yaml/executor @register/case json "
        "— LLM must determine api_type manually (for CaseSet naming)"
    )


# ---------------------------------------------------------------------------
# Locate the executor class to adapt (unified for both modes)
# ---------------------------------------------------------------------------

def find_executor_class(cann_dir):
    """Return (class_name, filename, warning).

    The class to adapt is the one inheriting **BaseApi directly**. Subclasses
    of ``AclnnBaseApi`` are CANN aclnn operators — NOT the PTA operator we
    want to verify, so they are excluded (word-boundary match: ``\\bBaseApi\\b``
    does not match ``AclnnBaseApi``). This is the unified locator for both
    input modes: mode "3file" has no atk.yaml, so locating via
    ``@register("<api_type>")`` is unreliable; inheritance is not. The
    ``__call__`` method of this class is where the device branch goes.
    """
    matches = []
    for ef in sorted(glob.glob(os.path.join(cann_dir, "executor*.py"))):
        for m in re.finditer(r"class\s+(\w+)\s*\(([^)]*)\)", read(ef)):
            if re.search(r"\bBaseApi\b", m.group(2)):
                matches.append((m.group(1), os.path.basename(ef)))

    if not matches:
        return None, None, (
            "no class inheriting BaseApi found in executor*.py "
            "— LLM must locate the executor class manually"
        )

    cls, fn = matches[0]
    warn = None
    if len(matches) > 1:
        warn = ("multiple BaseApi subclasses found: "
                + ", ".join(f"{c} ({f})" for c, f in matches)
                + f" — defaulting to {cls}; confirm with the user if unsure")
    return cls, fn, warn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Auto-adapt CANN ATK -> PTA ATK (copy to PTA dir, extract info, patch node.yaml)"
    )
    ap.add_argument("cann_dir", help="CANN ATK case directory")
    ap.add_argument("--project-root", default=None,
                    help="Project root (auto-detected from cann_dir upward)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print info only; do not copy or patch files")
    ap.add_argument("--torch-path", default=None,
                    help="torch install dir (auto-detected via import torch if omitted); "
                         "used to locate _torch_docs.py/_tensor_docs.py for native pytorch ops")
    args = ap.parse_args()

    cann_dir = os.path.abspath(args.cann_dir)
    if not os.path.isdir(cann_dir):
        sys.exit(f"Not a directory: {cann_dir}")

    project_root = args.project_root or find_project_root(cann_dir) or find_project_root(os.getcwd())
    if not project_root:
        sys.exit("Cannot find project root (no codegen/templates/_op_plugin_docs.py)")

    print(f"Project root : {project_root}")
    print(f"CANN dir     : {cann_dir}")
    mode, json_files = detect_mode(cann_dir)
    print(f"Input mode   : {mode}"
          + (" (atk.yaml present — cases generated via `atk case`)"
             if mode == "4file"
             else " (pre-generated case json — `atk case` step is skipped)"))
    print()

    errors = []

    if mode == "3file" and len(json_files) > 1:
        warn = ("multiple top-level *.json found: " + ", ".join(json_files)
                + " — specify which one is the case json (not auto-picking)")
        print(f"  WARNING: {warn}")
        print()
        errors.append(warn)

    # --- Step 1: opName ---
    print("=" * 60)
    print("Step 1: Get opName from CHECKLIST.md")
    print("=" * 60)
    opname, err = extract_opname(project_root)
    if err:
        print(f"  ERROR: {err}")
        errors.append(err)
    else:
        print(f"  opName = {opname}")
    print()

    # --- Step 2: prepare PTA dir (copy files) ---
    print("=" * 60)
    print("Step 2: Create PTA dir and copy files")
    print("=" * 60)
    pta_dir = None
    if opname:
        pta_dir, copied, err = prepare_pta_dir(project_root, opname, cann_dir, mode, dry_run=args.dry_run)
        if err:
            print(f"  ERROR: {err}")
            errors.append(err)
        else:
            action = "Would copy" if args.dry_run else "Copied"
            print(f"  PTA dir  : {pta_dir}")
            for f in copied:
                print(f"  {action}: {f}")
    else:
        print("  SKIPPED (no opName)")
    print()

    # All subsequent steps operate on the PTA dir.
    # In dry-run the PTA dir doesn't exist yet, so fall back to cann_dir for previewing.
    work_dir = pta_dir if (pta_dir and not args.dry_run) else cann_dir

    # --- Step 3: interface prototype ---
    print("=" * 60)
    print("Step 3: Find interface prototype")
    print("=" * 60)
    sig_line = func_name = params = None
    proto_source = None      # where the prototype came from (for SUMMARY/traceability)
    is_native = opname is not None and not opname.startswith("npu_")
    if opname and not is_native:
        # torch_npu custom op -> op-plugin's _op_plugin_docs.py (existing path)
        sig_line, func_name, params, err = find_prototype(project_root, opname)
        proto_source = "_op_plugin_docs.py"
    elif opname:
        # native pytorch op -> torch install dir _torch_docs.py / _tensor_docs.py
        print(f'  "{opname}" is not npu_-prefixed -> treating as native pytorch op')
        torch_path = resolve_torch_path(args.torch_path)
        if torch_path:
            print(f"  torch path : {torch_path}")
        sig_line, func_name, params, proto_source, err = find_torch_prototype(opname, torch_path)
    else:
        err = None

    if not opname:
        print("  SKIPPED (no opName)")
    elif err:
        print(f"  WARNING: {err}")
        print("  -> LLM must find the interface prototype manually")
        errors.append(err)
    else:
        print(f"  Source    : {proto_source}")
        print(f"  Signature : {sig_line}")
        print(f"  Function  : {func_name}")
        if is_native:
            print("  NOTE: native pytorch op — bare (un-annotated) params use a name "
                  "heuristic for is_tensor; LLM must verify.")
        if params:
            print(f"  Parameters ({len(params)}):")
            for p in params:
                tags = []
                if p["is_tensor"]:
                    tags.append("TENSOR -> .to(npu)")
                if p["is_optional"]:
                    tags.append("optional")
                if p["default"] is not None:
                    tags.append(f'default={p["default"]}')
                tag_str = f"  [{', '.join(tags)}]" if tags else ""
                ptype = p["type"] if p["type"] is not None else "?"
                print(f"    {p['name']:<20} {ptype:<12}{tag_str}")
        else:
            print("  No parameters")
    print()

    # --- Step 4: patch node.yaml in PTA dir ---
    print("=" * 60)
    print("Step 4: Patch node.yaml in PTA dir (pyaclnn -> npu)")
    print("=" * 60)
    patched, already_npu, no_match, err = patch_node_yaml(work_dir, dry_run=args.dry_run)
    if err:
        print(f"  WARNING: {err}")
        errors.append(err)
    else:
        action = "Would patch" if args.dry_run else "Patched"
        for f in patched:
            print(f"  {action}: {f}  (backend: pyaclnn -> npu)")
        for f in already_npu:
            print(f"  Already npu: {f}  (no action needed)")
        for f in no_match:
            print(f"  WARNING: {f} has no 'backend: pyaclnn' and no 'backend: npu' — check manually")
            errors.append(f"{f}: no pyaclnn/npu backend found")
    print()

    # --- Step 5: executor class (BaseApi subclass) + api_type ---
    print("=" * 60)
    print("Step 5: executor class & api_type")
    print("=" * 60)
    exec_cls, exec_file, warn = find_executor_class(work_dir)
    if warn and exec_cls is None:
        print(f"  WARNING: {warn}")
        errors.append(warn)
    else:
        print(f"  executor class : {exec_cls}  (in {exec_file}, inherits BaseApi)")
        print(f"  -> Adapt THIS class: add the device branch in its `__call__` method")
        if warn:
            print(f"  WARNING: {warn}")
            errors.append(warn)
    api_type, api_type_src, err = parse_yaml_api_type(work_dir)
    if err:
        print(f"  WARNING: {err}")
    else:
        print(f"  api_type       : {api_type}  (from {api_type_src})")
        print(f"  -> for CaseSet/xlsx naming; sanity-check it matches @register on the class above")
    print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY (for LLM code generation)")
    print("=" * 60)
    print(f"  mode              : {mode}")
    if mode == "3file":
        if len(json_files) == 1:
            print(f"  case_json         : {json_files[0]}  (copied into PTA dir; `atk case` is SKIPPED, use this for atk task -c)")
        else:
            print(f"  case_json         : AMBIGUOUS — candidates: {', '.join(json_files)}; specify one for atk task -c")
    else:
        print("  case_json         : run step 2 `atk case -f atk.yaml -p generate.py` to generate result/<api_type>.json")
    if pta_dir:
        print(f"  PTA dir           : {pta_dir}")
    if opname:
        print(f"  opName            : {opname}")
    if proto_source:
        print(f"  prototype source  : {proto_source}")
    if func_name:
        if is_native:
            base = "<tensor>" if proto_source == "_tensor_docs.py" else "torch"
            print(f"  call form         : {base}.{func_name}(...)  (native pytorch op)")
        else:
            print(f"  call form         : {func_name}(...)")
    if params:
        print(f"  tensor params     : {', '.join(p['name'] for p in params if p['is_tensor'])}")
        print(f"  scalar params     : {', '.join(p['name'] for p in params if not p['is_tensor'])}")
        opt = [p['name'] for p in params if p['is_optional']]
        if opt:
            print(f"  optional params   : {', '.join(opt)}")
    if exec_cls:
        print(f"  executor class    : {exec_cls}  (inherits BaseApi — patch its `__call__`)")
    if api_type:
        print(f"  api_type          : \"{api_type}\"  (from {api_type_src}; CaseSet naming, = @register value)")
    print()

    if errors:
        print(f"Errors/Warnings: {len(errors)}")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All automated steps completed successfully.")


if __name__ == "__main__":
    main()
