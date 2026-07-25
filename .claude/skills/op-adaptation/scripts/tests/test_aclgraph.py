"""Tests for gen-aclgraph (lib/aclgraph.py) and verify ACLGraph structural checks.

Golden: IFA v1/v2 仓内真身为唯一 conformance 参考；
FA v3 仅作 SymInt→int 降级 warning 路径的触发样本（未真实接入，不作 conformance 目标）。

Run: pytest tests/test_aclgraph.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.signature import verify_op, check_aclgraph_structure, find_yaml_func_raw, _is_aclgraph_aux_op
from lib.aclgraph import run_gen_aclgraph
from lib.common import find_repo_root, repo_paths


GOLDEN_OPS = [
    "npu_fused_infer_attention_score",
    "npu_fused_infer_attention_score_v2",
]

GOLDEN_ACLNN = {
    "npu_fused_infer_attention_score": {
        "aclnnFusedInferAttentionScoreV2",
        "aclnnFusedInferAttentionScoreV3",
        "aclnnFusedInferAttentionScoreV5",
    },
    "npu_fused_infer_attention_score_v2": {
        "aclnnFusedInferAttentionScoreV4",
        "aclnnFusedInferAttentionScoreV5",
    },
}


def _hard(mismatches, check=None):
    return [m for m in mismatches
            if m.get("severity") not in ("warning", "known")
            and (check is None or m["check"] == check)]


def _warnings(mismatches, check=None):
    return [m for m in mismatches
            if m.get("severity") == "warning"
            and (check is None or m["check"] == check)]


@pytest.fixture(scope="module")
def repo_root():
    return find_repo_root()


@pytest.fixture(scope="module")
def yaml_path(repo_root):
    return repo_paths(repo_root)["yaml"]


# ── golden case: verify 的 ACLGraph 结构检查全过（无 BLOCK、无降级 warning）──

@pytest.mark.parametrize("op_name", GOLDEN_OPS)
def test_verify_aclgraph_structure_golden(repo_root, op_name):
    report = verify_op(op_name, repo_root)
    assert report.get("aclgraph"), f"{op_name}: expected aclgraph section in verify report"
    assert report["aclgraph"]["out_entry"], f"{op_name}: .out entry not detected"
    assert report["aclgraph"]["get_max_workspace_entry"], f"{op_name}: get-max entry not detected"
    hard = _hard(report["mismatches"], check="aclgraph")
    assert not hard, f"{op_name}: ACLGraph structural BLOCKs on golden op:\n" + "\n".join(
        m["error"] for m in hard)
    warns = _warnings(report["mismatches"], check="aclgraph")
    assert not warns, f"{op_name}: IFA 无 SymInt 降级，不应有 aclgraph warning: {warns}"


# ── golden case: gen-aclgraph YAML 草稿与仓内真身逐字一致（硬回归断言）──

@pytest.mark.parametrize("op_name", GOLDEN_OPS)
def test_gen_aclgraph_yaml_matches_repo_verbatim(repo_root, yaml_path, op_name):
    result = run_gen_aclgraph(op_name, repo_root)
    assert "error" not in result, result.get("error")

    golden_out = find_yaml_func_raw(op_name + ".out", yaml_path)
    golden_gmw = find_yaml_func_raw("_" + op_name + "_get_max_workspace", yaml_path)
    assert golden_out and golden_gmw, "golden entries must exist in repo yaml"

    assert result["yaml_out"] == golden_out, (
        f"{op_name}.out 草稿与仓内真身不一致:\n  gen: {result['yaml_out']}\n  gold: {golden_out}")
    assert result["yaml_get_max_workspace"] == golden_gmw, (
        f"_{op_name}_get_max_workspace 草稿与仓内真身不一致")
    # SymInt 参数默认保留（IFA 形态），无降级分歧
    assert "int[]?" not in result["yaml_out"]
    assert "int[]?" not in result["yaml_get_max_workspace"]
    assert result["symint_params"], f"{op_name}: expected SymInt params to be flagged"


# ── golden case: C++ 骨架结构（不逐字，核函数签名/双路径/EXEC 宏/aclnn 名）──

@pytest.mark.parametrize("op_name", GOLDEN_OPS)
def test_gen_aclgraph_cpp_skeleton_structure(repo_root, op_name):
    result = run_gen_aclgraph(op_name, repo_root)
    assert "error" not in result, result.get("error")

    # aclnn 名机械提取正确
    assert set(result["aclnn_names"]) == GOLDEN_ACLNN[op_name], (
        f"{op_name}: aclnn names {result['aclnn_names']} != {GOLDEN_ACLNN[op_name]}")
    assert result["has_version_dispatch"] is True, f"{op_name}: IFA functional 有版本/SoC 门控"
    assert result["cpp_functional_name"].endswith("_symint")

    sk = result["cpp_out_skeleton"]
    assert f"{op_name}_out_symint(" in sk
    assert "workspace.has_value()" in sk, "缺双路径分支"
    assert "EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD" in sk, "workspace 路径必须用 UPDATE 宏"
    assert "EXEC_NPU_NO_FORMAT_CHECK_CMD" in sk, "无 workspace 普通路径必须保留"
    assert "numel() * workspace->element_size()" in sk, "workspace 字节数计算"
    assert "at::Tensor &attention_out" in sk and "at::Tensor &softmax_lse" in sk

    g = result["cpp_get_max_skeleton"]
    assert f"_{op_name}_get_max_workspace_symint(" in g
    assert "EXEC_GET_MAX_WORKSPACE_CMD" in g
    assert "at::kByte" in g, "新接口 workspace dtype 应为 at::kByte"
    assert "apply_tensor_without_format" in g

    # 多 aclnn 名 → 显式 TODO
    assert "TODO" in sk and "TODO" in g


# ── warning 路径样本: FA v3 的 prefix 降级报 warning 而非 BLOCK ──

def test_verify_fa_v3_symint_downgrade_is_warning_not_block(repo_root):
    report = verify_op("npu_fusion_attention_v3", repo_root)
    acl = [m for m in report["mismatches"] if m["check"] == "aclgraph"]
    assert acl, "FA v3 应触发 ACLGraph 结构检查（.out / get-max 条目存在）"

    hard = [m for m in acl if m.get("severity") not in ("warning", "known")]
    assert not hard, "FA v3 降级不应报 BLOCK:\n" + "\n".join(m["error"] for m in hard)

    warns = [m for m in acl if m.get("severity") == "warning"]
    assert warns, "FA v3 的 prefix 降级应报 warning"
    assert any("prefix" in m["error"] and "存疑形态" in m["error"] for m in warns), (
        f"warning 应指向 prefix 降级并标注存疑形态: {[m['error'] for m in warns]}")
    # .out 与 _get_max_workspace 两处降级都应被标出
    assert any(m["error"].startswith("npu_fusion_attention_v3.out") for m in warns)
    assert any(m["error"].startswith("_npu_fusion_attention_v3_get_max_workspace") for m in warns)


# ── 辅助接口 meta 缺失为预期（warning，不是 hard mismatch）──

@pytest.mark.parametrize("aux_op", [
    "npu_fused_infer_attention_score.out",
    "_npu_fused_infer_attention_score_get_max_workspace",
    "_npu_fused_infer_attention_score_infer_output",
])
def test_verify_aux_op_meta_missing_is_expected(repo_root, aux_op):
    report = verify_op(aux_op, repo_root)
    meta_mm = [m for m in report["mismatches"] if m["check"] == "meta"]
    assert meta_mm, f"{aux_op}: expected a meta note"
    assert all(m.get("severity") == "warning" for m in meta_mm), (
        f"{aux_op}: meta 缺失应降级为 warning（预期形态）: {meta_mm}")
    assert report["passed"], f"{aux_op}: aux 接口不应有 hard mismatch: {report['mismatches']}"


# ── 豁免作用域：普通 .out（无 workspace）缺 meta 仍 BLOCK，不得被误豁免 ──

def test_is_aclgraph_aux_op_normal_out_not_exempt(tmp_path):
    """普通 .out 变体（签名无 workspace）不是 ACLGraph aux，不豁免。"""
    yaml = _write_yaml(tmp_path, f"""\
custom:
  - func: npu_demo(Tensor q) -> Tensor
    op_api: [v2.1, newest]
  - func: npu_demo.out(Tensor q, Tensor(a!) out) -> Tensor(a!)
    op_api: [v2.1, newest]
""")
    assert _is_aclgraph_aux_op("npu_demo.out", yaml) is False
    # 无 YAML 可查时同样保守不豁免（保留原 BLOCK 行为）
    assert _is_aclgraph_aux_op("npu_demo.out", None) is False


def test_is_aclgraph_aux_op_aclgraph_out_with_workspace_exempt(tmp_path):
    """带 workspace 的 .out 是 ACLGraph aux，豁免；_get_max_workspace 按名豁免。"""
    yaml = _write_yaml(tmp_path, f"""\
custom:
  - func: npu_demo(Tensor q) -> (Tensor, Tensor)
    op_api: [v2.1, newest]
  - func: npu_demo.out(Tensor q, Tensor? workspace=None, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
    op_api: [v2.1, newest]
  - func: _npu_demo_get_max_workspace(Tensor q) -> Tensor
    op_api: [v2.1, newest]
""")
    assert _is_aclgraph_aux_op("npu_demo.out", yaml) is True
    assert _is_aclgraph_aux_op("_npu_demo_get_max_workspace", yaml) is True


def test_verify_normal_out_missing_meta_is_block(repo_root):
    """仓内普通 .out（无 workspace）缺 meta 必须报 BLOCK，不得被 ACLGraph 豁免误吞。"""
    for op in ("abs.out", "add.out", "_softmax.out"):
        report = verify_op(op, repo_root)
        meta_hard = [m for m in report["mismatches"]
                     if m["check"] == "meta" and m.get("severity") not in ("warning", "known")]
        assert meta_hard, f"{op}: 普通 .out 缺 meta 应报 BLOCK（不应被 ACLGraph 豁免误吞）"
        assert not report["passed"], f"{op}: 缺 meta 的普通 .out 不应 passed"


# ── 错误样本: 构造错误签名，断言报出 BLOCK/mismatch ──

def _write_yaml(tmp_path, entries: str) -> Path:
    p = tmp_path / "op_plugin_functions.yaml"
    p.write_text(entries)
    return p


FUNC_DEMO = "npu_demo(Tensor q, Tensor k, *, SymInt[]? seq=None, int heads=1) -> (Tensor, Tensor)"


def test_check_out_missing_workspace_is_block(tmp_path):
    yaml = _write_yaml(tmp_path, f"""\
custom:
  - func: {FUNC_DEMO}
    op_api: [v2.1, newest]

  - func: npu_demo.out(Tensor q, Tensor k, *, SymInt[]? seq=None, int heads=1, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
    op_api: [v2.1, newest]
""")
    errs = check_aclgraph_structure("npu_demo", yaml)
    hard = _hard(errs)
    assert any("workspace" in m["error"] for m in hard), (
        f".out 缺 workspace 应报 BLOCK: {[m['error'] for m in errs]}")


def test_check_out_param_mismatch_is_block(tmp_path):
    yaml = _write_yaml(tmp_path, f"""\
custom:
  - func: {FUNC_DEMO}
    op_api: [v2.1, newest]

  - func: npu_demo.out(Tensor q, Tensor k, *, SymInt[]? seq=None, int heads=2, Tensor? workspace=None, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
    op_api: [v2.1, newest]
""")
    errs = check_aclgraph_structure("npu_demo", yaml)
    hard = _hard(errs)
    assert any("heads=2" in m["error"] and "逐字一致" in m["error"] or "须一致" in m["error"]
               for m in hard), (
        f".out 参数段与 functional 不一致应报 BLOCK: {[m['error'] for m in errs]}")


def test_check_get_max_non_tensor_return_is_block(tmp_path):
    yaml = _write_yaml(tmp_path, f"""\
custom:
  - func: {FUNC_DEMO}
    op_api: [v2.1, newest]

  - func: _npu_demo_get_max_workspace(Tensor q, Tensor k, *, SymInt[]? seq=None, int heads=1) -> (Tensor, Tensor)
    op_api: [v2.1, newest]
""")
    errs = check_aclgraph_structure("npu_demo", yaml)
    hard = _hard(errs)
    assert any("-> Tensor" in m["error"] for m in hard), (
        f"get-max 返回非 Tensor 应报 BLOCK: {[m['error'] for m in errs]}")


def test_check_symint_downgrade_is_warning_not_block(tmp_path):
    """合成降级样本：与 FA v3 同形态，独立于仓内条目验证规则本身。"""
    yaml = _write_yaml(tmp_path, f"""\
custom:
  - func: {FUNC_DEMO}
    op_api: [v2.1, newest]

  - func: npu_demo.out(Tensor q, Tensor k, *, int[]? seq=None, int heads=1, Tensor? workspace=None, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
    op_api: [v2.1, newest]

  - func: _npu_demo_get_max_workspace(Tensor q, Tensor k, *, int[]? seq=None, int heads=1) -> Tensor
    op_api: [v2.1, newest]
""")
    errs = check_aclgraph_structure("npu_demo", yaml)
    assert not _hard(errs), f"降级不应报 BLOCK: {[m['error'] for m in _hard(errs)]}"
    warns = _warnings(errs)
    assert len(warns) == 2, f".out 与 get-max 两处降级都应报 warning: {warns}"
    assert all("存疑形态" in m["error"] for m in warns)


def test_check_correct_entries_no_findings(tmp_path):
    """正确的 2 接口 + functional：零发现。"""
    yaml = _write_yaml(tmp_path, f"""\
custom:
  - func: {FUNC_DEMO}
    op_api: [v2.1, newest]

  - func: npu_demo.out(Tensor q, Tensor k, *, SymInt[]? seq=None, int heads=1, Tensor? workspace=None, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
    op_api: [v2.1, newest]

  - func: _npu_demo_get_max_workspace(Tensor q, Tensor k, *, SymInt[]? seq=None, int heads=1) -> Tensor
    op_api: [v2.1, newest]
""")
    errs = check_aclgraph_structure("npu_demo", yaml)
    assert errs == [], f"正确条目应零发现: {errs}"


# ── gen-aclgraph 错误路径 ──

def test_gen_aclgraph_unknown_op_errors(repo_root):
    result = run_gen_aclgraph("npu_no_such_op_anywhere", repo_root)
    assert "error" in result
