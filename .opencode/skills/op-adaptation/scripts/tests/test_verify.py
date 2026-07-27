"""Golden tests for op_adapt.py verify command.

Runs verify against existing operators in the repo and checks results.
Uses known_inconsistent.json to allowlist historical debt.

Run: pytest tests/test_verify.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.signature import verify_op
from lib.common import find_repo_root


# ── Known-good operators (should pass verify) ───────────────────────
# NOTE: npu_dynamic_quant was deleted from the repo; use _asymmetric variant.

KNOWN_GOOD = [
    "npu_dynamic_quant_asymmetric",
    "npu_fast_gelu",
    "npu_fast_gelu_backward",
    "npu_moe_token_permute_with_routing_map",
    "npu_moe_token_unpermute_with_routing_map",
]


# ── Known-inconsistent operators (historical debt) ──────────────────

def load_known_inconsistent() -> set:
    """Load the allowlist of operators with known inconsistencies."""
    p = Path(__file__).parent / "known_inconsistent.json"
    if not p.exists():
        return set()
    data = json.loads(p.read_text())
    return set(data.get("operators", []))


KNOWN_INCONSISTENT = load_known_inconsistent()


# ── Tests ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def repo_root():
    return find_repo_root()


@pytest.mark.parametrize("op_name", KNOWN_GOOD)
def test_verify_known_good(repo_root, op_name):
    """Verify known-good operators pass signature cross-check."""
    if op_name in KNOWN_INCONSISTENT:
        pytest.skip(f"{op_name} is in known_inconsistent allowlist")
    report = verify_op(op_name, repo_root)
    assert report["passed"], (
        f"{op_name} failed verify:\n"
        + "\n".join(f"  {m['check']}: {m['error']}" for m in report["mismatches"])
    )


def test_verify_yaml_cpp_param_names_match(repo_root):
    """YAML and C++ parameter names should match for npu_dynamic_quant_asymmetric."""
    report = verify_op("npu_dynamic_quant_asymmetric", repo_root)
    assert report["yaml"] is not None, "YAML entry not found"
    assert report["cpp"] is not None, "C++ signature not found"
    yaml_params = {p["name"] for p in report["yaml"]["params"]}
    cpp_params = {p["name"] for p in report["cpp"]["params"]}
    assert yaml_params == cpp_params, f"YAML params {yaml_params} != C++ params {cpp_params}"


def test_verify_yaml_meta_param_names_match(repo_root):
    """YAML and Meta parameter names should match for npu_dynamic_quant_asymmetric."""
    report = verify_op("npu_dynamic_quant_asymmetric", repo_root)
    assert report["yaml"] is not None, "YAML entry not found"
    assert report["meta"] is not None, "Meta registration not found"
    yaml_params = {p["name"] for p in report["yaml"]["params"]}
    meta_params = {p["name"] for p in report["meta"]["params"]}
    # Meta may rename with _dummy suffix
    meta_normalized = {p.replace("_dummy", "") for p in meta_params}
    assert yaml_params == meta_normalized, f"YAML params {yaml_params} != Meta params {meta_normalized}"


def test_verify_cpp_type_compatibility(repo_root):
    """C++ types should be compatible with YAML types for npu_dynamic_quant_asymmetric."""
    report = verify_op("npu_dynamic_quant_asymmetric", repo_root)
    assert report["yaml"] is not None, "YAML entry not found"
    assert report["cpp"] is not None, "C++ signature not found"
    # Should have no type mismatch mismatches
    type_mismatches = [
        m for m in report["mismatches"]
        if m["check"] == "yaml_vs_cpp" and "incompatible" in m["error"]
    ]
    assert len(type_mismatches) == 0, f"Type mismatches: {type_mismatches}"


def test_verify_type_compat_is_warning_not_hard_fail(repo_root):
    """Type-compat (heuristic) must be a warning, NOT a hard failure that flips passed.
    Guards the Tensor(b!) false-positive scenario (~52 ops): Tensor(b!) is stripped
    to Tensor by _strip_alias before matching, so no false alarm. Any remaining
    type-compat mismatch must be severity='warning' so passed reflects only hard mismatches.
    """
    report = verify_op("_amp_foreach_non_finite_check_and_unscale_", repo_root)
    tc = [m for m in report["mismatches"] if "incompatible" in m.get("error", "")]
    # After _strip_alias fix, Tensor(b!) → Tensor (in YAML_TO_CPP), no false alarm
    assert not tc, f"Tensor(b!) should be stripped to Tensor (no mismatch), got: {[m['error'][:60] for m in tc]}"
    # passed must equal (no hard mismatches) — warnings must not flip it
    hard = [m for m in report["mismatches"] if m.get("severity") not in ("warning", "known")]
    assert report["passed"] == (len(hard) == 0), (
        f"passed={report['passed']} but hard mismatches={len(hard)}; warnings must not flip passed")


def test_verify_known_inconsistent_op_is_downgraded(repo_root):
    """Allowlisted ops must be marked known_inconsistent, mismatches downgraded to
    severity='known', and passed stays true — verifies the runtime consults
    known_inconsistent.json (shared semantics with the test-suite skip)."""
    op = "npu_conv2d"
    if op not in KNOWN_INCONSISTENT:
        pytest.skip(f"{op} not in known_inconsistent allowlist")
    report = verify_op(op, repo_root)
    assert report["known_inconsistent"] is True, f"{op} should be flagged allowlisted"
    assert all(m.get("severity") == "known" for m in report["mismatches"]), (
        f"allowlisted op mismatches must be severity='known': "
        f"{[m.get('severity') for m in report['mismatches']]}")
    assert report["passed"] is True, f"allowlisted op {op} must pass (known mismatches accepted)"
