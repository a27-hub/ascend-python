"""Golden tests for discover, gen-baseline, gen-summary scripts.

Run: pytest tests/test_other_scripts.py -v
"""

import json
import sys
import os
import tempfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.discover import run_discover
from lib.baseline import run_gen_baseline
from lib.summary import run_gen_summary
from lib.common import find_repo_root


@pytest.fixture(scope="module")
def repo_root():
    return find_repo_root()


# ── discover tests ──────────────────────────────────────────────────

def test_discover_finds_existing_op(repo_root):
    """discover should find npu_dynamic_quant_asymmetric in the repo."""
    result = run_discover("npu_dynamic_quant_asymmetric", repo_root)
    found_files = [f for f in result["files"] if f.get("found")]
    assert len(found_files) >= 3, f"Expected >=3 files, got {len(found_files)}"
    # Must find YAML entry
    assert any("yaml" in f["file"] for f in found_files), "YAML not found"
    # Must find Meta registration
    assert any("meta" in f["file"] for f in found_files), "Meta not found"


def test_discover_deleted_op_not_in_registration(repo_root):
    """discover for a deleted op should NOT find registration files.

    Skips if npu_dynamic_quant is still present in the repo — the test's
    precondition (op deleted) is environment-dependent and shouldn't hard-fail
    when the repo still contains the op.
    """
    yaml_path = repo_root / "op_plugin" / "config" / "op_plugin_functions.yaml"
    if yaml_path.exists() and "npu_dynamic_quant(" in yaml_path.read_text():
        pytest.skip("npu_dynamic_quant still present in repo (deletion precondition not met)")
    result = run_discover("npu_dynamic_quant", repo_root)
    found = [f for f in result["files"] if f.get("found")]
    reg_found = [f for f in found if any(k in f["file"] for k in
        ("op_plugin_functions.yaml", "_meta_registrations", "schema_all", "allowlist", "test_fake_tensor"))]
    assert len(reg_found) == 0, f"Registration files still found: {[f['file'] for f in reg_found]}"


def test_discover_regex_includes_backward_grad_excludes_unrelated():
    """The word-boundary regex must find the op + its _backward/_grad (autograd chain)
    but NOT unrelated suffixes like _asymmetric. Guards against over-narrowing that
    silently drops reverse-impact signal when modifying a forward op."""
    from lib.discover import _grep_file
    import tempfile
    content = "\n".join([
        '@impl(m, "npu_fast_gelu")',
        '@impl(m, "npu_fast_gelu_backward")',
        'def npu_fast_gelu_grad(grad, self):',
        '@impl(m, "npu_fast_gelu_asymmetric")',
        'npu_fast_gelu_v2(',
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        hits = _grep_file(Path(path), "npu_fast_gelu")
    finally:
        Path(path).unlink()
    texts = [h["text"] for h in hits]
    # forward, backward, grad, v2 must all be found
    assert any('npu_fast_gelu"' in t for t in texts), f"forward missing: {texts}"
    assert any('npu_fast_gelu_backward' in t for t in texts), f"backward missing: {texts}"
    assert any('npu_fast_gelu_grad' in t for t in texts), f"grad missing: {texts}"
    assert any('npu_fast_gelu_v2' in t for t in texts), f"v2 missing: {texts}"
    # asymmetric must NOT be found
    assert not any('asymmetric' in t for t in texts), f"unrelated _asymmetric should be excluded: {texts}"


def test_discover_strips_version_suffix(repo_root):
    """discover must strip _v<N> suffix from the op name before searching,
    so searching npu_xxx_v2 finds the base npu_xxx. Guards multi-digit (_v10)."""
    r2 = run_discover("npu_fast_gelu_v2", repo_root)
    assert r2["search_name"] == "npu_fast_gelu", f"_v2 not stripped: {r2['search_name']}"
    # multi-digit suffix must also strip (\d+ not \d)
    r10 = run_discover("npu_fast_gelu_v10", repo_root)
    assert r10["search_name"] == "npu_fast_gelu", f"_v10 not stripped: {r10['search_name']}"
    # base name (no suffix) is unchanged
    r0 = run_discover("npu_fast_gelu", repo_root)
    assert r0["search_name"] == "npu_fast_gelu"


def test_discover_backward_referenced_direction(repo_root):
    """An op referenced only as a backward in a forward's formula (not a - name:
    entry itself) must report direction=backward_referenced_by_forward.
    Covers the second binding direction (forward_has_backward is covered by
    npu_fast_gelu in other tests)."""
    result = run_discover("npu_fast_gelu_backward", repo_root)
    binding = result.get("derivatives_binding")
    assert binding is not None, "expected a derivatives_binding for a backward-referenced op"
    assert binding["direction"] == "backward_referenced_by_forward", (
        f"expected backward_referenced_by_forward, got {binding['direction']}")


def test_discover_word_boundary_no_substring(repo_root):
    """discover for npu_fast_gelu should NOT match npu_fast_gelu_backward as same op."""
    result = run_discover("npu_fast_gelu", repo_root)
    # Should find npu_fast_gelu in YAML
    yaml_hit = [f for f in result["files"] if f.get("found") and "yaml" in f["file"]]
    assert len(yaml_hit) >= 1, "npu_fast_gelu not found in YAML"
    # The YAML line should contain "func: npu_fast_gelu(" not "npu_fast_gelu_backward"
    yaml_lines = yaml_hit[0]["lines"]
    line_texts = [l["text"] for l in yaml_lines]
    assert any("npu_fast_gelu(" in t for t in line_texts), f"Expected func: npu_fast_gelu(, got: {line_texts}"


def test_discover_derivatives_binding(repo_root):
    """discover should detect derivatives binding for npu_fast_gelu."""
    result = run_discover("npu_fast_gelu", repo_root)
    deriv = result.get("derivatives_binding")
    assert deriv is not None, "Expected derivatives binding for npu_fast_gelu"
    assert deriv["direction"] == "forward_has_backward"


# ── gen-baseline tests ──────────────────────────────────────────────

def test_gen_baseline_dry_run(repo_root):
    """gen-baseline default should be dry-run (no file modification)."""
    result = run_gen_baseline("npu_dynamic_quant_asymmetric", repo_root, write=False)
    assert result.get("dry_run") is True
    assert result.get("written") is False
    assert len(result.get("schema_entries", [])) >= 2
    assert result.get("allowlist_entry") is not None


def test_gen_baseline_func_line_from_yaml(repo_root):
    """gen-baseline should read func: line from YAML."""
    result = run_gen_baseline("npu_dynamic_quant_asymmetric", repo_root, write=False)
    func_line = result.get("yaml_func_line", "")
    assert "npu_dynamic_quant_asymmetric" in func_line
    assert "Tensor" in func_line


def test_gen_baseline_schema_entries_format(repo_root):
    """Schema entries should have 'func:' and 'op_api:' keys."""
    result = run_gen_baseline("npu_dynamic_quant_asymmetric", repo_root, write=False)
    entries = result.get("schema_entries", [])
    assert len(entries) >= 2
    keys = [e["key"] for e in entries]
    assert any(k.startswith("func:") for k in keys), f"Missing func: entry in {keys}"
    assert any(k.startswith("op_api:") for k in keys), f"Missing op_api: entry in {keys}"


# ── --write path tests (targeted insertion, format-preserving) ──────

def test_insert_schema_entry_preserves_formatting():
    """--write must NOT reformat the whole file (no inline→multi-line array expansion)."""
    from lib.baseline import _insert_schema_entry_text
    raw = '{\n  "func: npu_existing(Tensor x) -> Tensor": {\n    "version": ["v2.1", "v2.5", "newest"]\n  }\n}\n'
    orig_lines = raw.count("\n") + 1
    orig_inline = raw.count('"version": ["v2.1", "v2.5", "newest"]')  # inline array present
    new_raw, inserted = _insert_schema_entry_text(raw, "func: npu_new_op(Tensor y) -> Tensor", ["newest"])
    assert inserted is True
    assert new_raw.count("\n") + 1 - orig_lines <= 3  # only the new entry, no reformatting
    assert new_raw.count('"version": ["v2.1", "v2.5", "newest"]') == orig_inline  # existing inline array unchanged
    d = json.loads(new_raw)  # valid JSON
    assert "func: npu_new_op(Tensor y) -> Tensor" in d
    assert d["func: npu_existing(Tensor x) -> Tensor"]["version"] == ["v2.1", "v2.5", "newest"]


def test_insert_schema_entry_idempotent():
    """Re-inserting an existing key is a no-op."""
    from lib.baseline import _insert_schema_entry_text
    raw = '{\n  "func: npu_x(Tensor x) -> Tensor": {\n    "version": ["newest"]\n  }\n}\n'
    _, first = _insert_schema_entry_text(raw, "func: npu_x(Tensor x) -> Tensor", ["newest"])
    assert first is False  # already present → no-op


def test_insert_allowlist_op_appends_and_preserves_format():
    """Allowlist --write appends to all_version without reformatting."""
    from lib.baseline import _insert_allowlist_op_text
    raw = '{\n  "torch_npu": {\n    "all_version": [\n      "npu_a",\n      "npu_b"\n    ]\n  }\n}\n'
    orig_lines = raw.count("\n") + 1
    new_raw, inserted = _insert_allowlist_op_text(raw, "npu_new")
    assert inserted is True
    assert new_raw.count("\n") + 1 - orig_lines == 1  # exactly one new line
    d = json.loads(new_raw)
    assert "npu_new" in d["torch_npu"]["all_version"]
    assert d["torch_npu"]["all_version"][-1] == "npu_new"


def test_insert_allowlist_op_idempotent():
    """Re-appending an existing op is a no-op."""
    from lib.baseline import _insert_allowlist_op_text
    raw = '{\n  "torch_npu": {\n    "all_version": [\n      "npu_a"\n    ]\n  }\n}\n'
    _, inserted = _insert_allowlist_op_text(raw, "npu_a")
    assert inserted is False


def test_insert_allowlist_op_prefix_not_false_positive():
    """H12: prefix of existing entry must NOT be treated as already present.

    E.g. inserting "npu_add" when "npu_add_v2" already exists should INSERT,
    not silently skip (the old substring check `if needle in lines[j]`
    false-positived on this case).
    """
    from lib.baseline import _insert_allowlist_op_text
    raw = '{\n  "torch_npu": {\n    "all_version": [\n      "npu_add_v2",\n      "npu_b"\n    ]\n  }\n}\n'
    new_raw, inserted = _insert_allowlist_op_text(raw, "npu_add")
    assert inserted is True, "npu_add should be inserted even though npu_add_v2 exists"
    d = json.loads(new_raw)
    assert "npu_add" in d["torch_npu"]["all_version"]
    assert "npu_add_v2" in d["torch_npu"]["all_version"]
    # Re-inserting the truly-existing op is idempotent
    _, inserted2 = _insert_allowlist_op_text(new_raw, "npu_add")
    assert inserted2 is False


def test_gen_baseline_write_uses_targeted_insertion(tmp_path):
    """End-to-end: --write on temp files inserts entries without reformatting."""
    from lib.baseline import _write_schema_json, _write_allowlist
    schema = tmp_path / "schema.json"
    schema.write_text('{\n  "func: npu_old(Tensor x) -> Tensor": {\n    "version": ["v2.1", "newest"]\n  }\n}\n')
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text('{\n  "torch_npu": {\n    "all_version": [\n      "npu_old"\n    ]\n  }\n}\n')
    n_inserted = _write_schema_json([{"key": "func: npu_new(Tensor x) -> Tensor", "version": ["newest"]}], schema)
    assert n_inserted == 1
    assert _write_allowlist("npu_new", allowlist) is True
    # H13: atomic write (temp+rename), no .bak file expected
    # inline array preserved (not expanded)
    new_raw = schema.read_text()
    assert '"version": ["v2.1", "newest"]' in new_raw  # existing inline array intact
    d = json.loads(new_raw)
    assert "func: npu_new(Tensor x) -> Tensor" in d
    # second call: idempotent (no new insert, no .bak overwrite needed)
    n2 = _write_schema_json([{"key": "func: npu_new(Tensor x) -> Tensor", "version": ["newest"]}], schema)
    assert n2 == 0


# ── gen-summary tests ───────────────────────────────────────────────

def test_gen_summary_dry_run():
    """gen-summary with dry_run=True should not write file."""
    result = run_gen_summary("npu_test_op", dry_run=True)
    assert result.get("dry_run") is True
    assert result.get("written") is False
    assert len(result.get("content", "")) > 0


def test_gen_summary_with_parse_json():
    """gen-summary should fill parameter table from parse JSON."""
    parse_json = {
        "aclnn_name": "aclnnTest",
        "op_type": "Forward",
        "parameters": [
            {"name": "x", "source": "both", "type": "Tensor", "optionality": "REQUIRED", "category": "input"},
        ],
        "conflicts": [],
        "constraints_logic": [],
    }
    result = run_gen_summary("npu_test_op", parse_json=parse_json, dry_run=True)
    content = result.get("content", "")
    assert "x" in content
    assert "Tensor" in content


def test_gen_summary_with_conflicts():
    """gen-summary should include conflict section when conflicts exist."""
    parse_json = {
        "aclnn_name": "aclnnTest",
        "op_type": "Forward",
        "parameters": [],
        "conflicts": [
            {"param": "foo", "desc": "_def.cpp has foo but .md does not", "type": "missing_in_md", "resolution": "awaiting_user_confirmation"},
        ],
        "constraints_logic": [],
    }
    result = run_gen_summary("npu_test_op", parse_json=parse_json, dry_run=True)
    content = result.get("content", "")
    assert "foo" in content or "Conflict" in content or "⚠️" in content


# ── discover m_adapt_state (H17) ────────────────────────────────────

def test_m_adapt_state_per_entry_not_whole_file(tmp_path):
    """H17: m_adapt_state classification must check flags INSIDE the target op's
    YAML entry, not across the whole file. A pure acl_op: op must NOT be classified
    as B just because some other op in the same file has gen_opapi:."""
    from lib.discover import _extract_yaml_entry_block, _classify_m_adapt_state

    # Construct a fake YAML with two ops:
    #   - npu_target (acl_op: only — should be C or D depending on cpp presence)
    #   - npu_other  (gen_opapi: — was causing the whole-file false positive B)
    yaml_content = """\
- func: npu_other(Tensor x) -> Tensor
  gen_opapi:
    struct_name: NpuOther

- func: npu_target(Tensor self, Tensor out) -> Tensor
  acl_op: aclnnTarget
  dispatch:
    CPU: aclnnTarget

- func: npu_third(Tensor y) -> Tensor
  op_api: npu_third
"""
    yaml_file = tmp_path / "op_plugin_functions.yaml"
    yaml_file.write_text(yaml_content)

    # _extract_yaml_entry_block must isolate npu_target's own block
    block = _extract_yaml_entry_block(yaml_content, "npu_target")
    assert block is not None
    assert "acl_op: aclnnTarget" in block
    assert "npu_other" not in block
    assert "npu_third" not in block
    assert "gen_opapi:" not in block, "gen_opapi flag from another op leaked into target block"

    # Build fake paths with empty opapi/aclops dirs → no cpp found
    opapi_dir = tmp_path / "op_plugin" / "ops" / "opapi"
    opapi_dir.mkdir(parents=True)
    aclops_dir = tmp_path / "op_plugin" / "ops" / "aclops"
    aclops_dir.mkdir(parents=True)
    # Create a cpp file in aclops containing "npu_target"
    (aclops_dir / "target.cpp").write_text("void npu_target_fun() {}\n")

    paths = {
        "root": tmp_path,
        "yaml": yaml_file,
        "opapi_dir": opapi_dir,
    }

    state = _classify_m_adapt_state("npu_target", paths)
    # acl_op: only, no op_api:, no gen_opapi:, aclops/*.cpp exists → state C
    assert state == "C", f"expected C for acl_op-only op, got {state}"

    # An op NOT present in YAML → state D
    state_absent = _classify_m_adapt_state("npu_absent", paths)
    assert state_absent == "D"


def test_m_adapt_state_gen_opapi_is_B(tmp_path):
    """An op that has gen_opapi: in its own entry is classified B."""
    from lib.discover import _classify_m_adapt_state

    yaml_content = """\
- func: npu_candidate(Tensor x) -> Tensor
  gen_opapi:
    struct_name: NpuCandidate
"""
    yaml_file = tmp_path / "op_plugin_functions.yaml"
    yaml_file.write_text(yaml_content)
    opapi_dir = tmp_path / "op_plugin" / "ops" / "opapi"
    opapi_dir.mkdir(parents=True)
    aclops_dir = tmp_path / "op_plugin" / "ops" / "aclops"
    aclops_dir.mkdir(parents=True)
    paths = {"root": tmp_path, "yaml": yaml_file, "opapi_dir": opapi_dir}
    assert _classify_m_adapt_state("npu_candidate", paths) == "B"

