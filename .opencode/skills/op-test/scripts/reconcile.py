#!/usr/bin/env python3
"""
覆盖对账 gate — step4 产出 UT 文件后、step6 跑测试前的强制前置校验。
把 SKILL.md:104(行数对账) + :86(增删日志) 从 model 自觉升级为脚本强制。

用法:
  python reconcile.py --checklist CHECKLIST.md --test-file test/{dir}/test_npu_{op}.py

退出码 0 = 通过; 非 0 = 阻断, 打印结构化缺口清单。

逻辑:
  1. AST 解析 test_npu_{op}.py → 收集 test_ / test_neg_ 方法名两集合
  2. 解析 CHECKLIST 三段:
     - Coverage Matrix → 正向期望集合(Test Item 列)
     - Negative Coverage Matrix → 负向期望集合
     - Finalization Add/Delete Log → 已记 Delete <item> / Add <item> 集合
  3. 四类判定:
     [MISS]       矩阵有、代码无、日志无 Delete → 硬失败 + 双路指引
     [DELETED-OK] 矩阵有、代码无、日志有 Delete → 放行
     [EXTRA]      代码有、矩阵无、日志无 Add → 警告
     [BAD-LOG]    Delete/Add 项不匹配任何矩阵行 → 警告
  4. 汇总: 任一 MISS > 0 → 退出码 1
"""

import ast
import re
import sys
import argparse
from pathlib import Path


# ─── AST: 收集测试方法名 ───────────────────────────────────────────

def collect_test_methods(test_file_path):
    """解析 test_npu_{op}.py, 收集 TestCase 子类的 test_ / test_neg_ 方法名。
    返回 (positives: set, negatives: set)。
    """
    tree = ast.parse(Path(test_file_path).read_text(encoding="utf-8"))
    pos, neg = set(), set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # 只收集 TestCase 子类: 类名须以 Test 开头
        if not node.name.startswith("Test"):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                name = item.name
                if name.startswith("test_neg_"):
                    neg.add(name)
                elif name.startswith("test_"):
                    pos.add(name)

    return pos, neg


# ─── AST: 随机种子 hard gate ─────────────────────────────────────────
# SKILL.md §4 从"model 自觉 self-check"升级为脚本强制:
# 任一 test_ 方法(含其调用的本类 helper)使用了 randn/rand/randint,
# 但该 test_ 方法体内无 manual_seed / np.random.seed → NO-SEED 阻断。
# test_neg_ 豁免: 负例只验错, 不比 golden, 种子对复现无意义。
# 跨 helper 传播: 随机常写在 generate_input 之类 helper 里, 种子须在
# test 方法调用 helper 前设置, 故"同作用域"须扩到"随机传递闭包"。

_RANDOM_CALLS = {"randn", "rand", "randint", "randn_like", "rand_like"}
_SEED_CALLS = {"manual_seed", "seed"}


def _call_attr_names(body_nodes):
    """收集一组 AST 节点里所有 Call 的末级函数名 (torch.randn→'randn')。"""
    names = set()
    for node in body_nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                if isinstance(fn, ast.Attribute):
                    names.add(fn.attr)
                elif isinstance(fn, ast.Name):
                    names.add(fn.id)
    return names


def _self_method_calls(body_nodes):
    """收集 self.<name>(...) 调用的 <name> 集合。"""
    names = set()
    for node in body_nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                val = sub.func.value
                if isinstance(val, ast.Name) and val.id == "self":
                    names.add(sub.func.attr)
    return names


def check_seeds(test_file_path):
    """返回无种子的 test_ 方法名列表(含跨 helper 随机传播)。"""
    tree = ast.parse(Path(test_file_path).read_text(encoding="utf-8"))
    violations = []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name.startswith("Test")):
            continue

        # 类内所有方法 name → FunctionDef
        methods = {it.name: it for it in node.body if isinstance(it, ast.FunctionDef)}

        # 每个方法: 直接是否含随机
        direct_random = {
            name: bool(_call_attr_names(fn.body) & _RANDOM_CALLS)
            for name, fn in methods.items()
        }
        # 固定点传播: 调用了"含随机" helper 的方法也标记为含随机
        changed = True
        while changed:
            changed = False
            for name, fn in methods.items():
                if direct_random[name]:
                    continue
                for callee in _self_method_calls(fn.body):
                    if direct_random.get(callee):
                        direct_random[name] = True
                        changed = True
                        break

        for name, fn in methods.items():
            if not name.startswith("test_"):
                continue
            # test_neg_ 豁免: 负例只验错(assertRaisesRegex), 不比 golden,
            # 随机种子对复现无意义 → 不强制 seed。
            if name.startswith("test_neg_"):
                continue
            if not direct_random[name]:
                continue
            # 种子须在 test 方法自身作用域内设置(调用 helper 前)
            if not (_call_attr_names(fn.body) & _SEED_CALLS):
                violations.append(name)

    return sorted(violations)


# ─── CHECKLIST 解析(复用 scaffold_coverage 解析函数) ────────────────

def parse_section(text, header, stop_at_next_section=True):
    """提取 header 到下一个 ## 或 ### 之间的内容。"""
    pattern = re.escape(header) + r'\n(.*?)(?=\n## |\n### |\Z)'
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1) if m else ""


def parse_markdown_table(text):
    """解析 markdown 表格, 返回 [{col: val}]。"""
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []
    header_line = None
    sep_line = None
    data_lines = []
    for i, line in enumerate(lines):
        if "|" in line and header_line is None:
            header_line = line
        elif "|" in line and re.match(r'^[\|\s\-:]+$', line) and sep_line is None:
            sep_line = line
        elif "|" in line and sep_line is not None:
            data_lines.append(line)
    if not header_line or not data_lines:
        return []

    headers = [h.strip() for h in header_line.split("|") if h.strip()]
    rows = []
    for dl in data_lines:
        cells = [c.strip() for c in dl.split("|") if c.strip() != ""]
        cells_clean = [c for c in cells if c]
        if len(cells_clean) >= len(headers):
            row = {}
            for j, h in enumerate(headers):
                row[h] = cells_clean[j] if j < len(cells_clean) else ""
            rows.append(row)
    return rows


def parse_coverage_matrix(text):
    """从 CHECKLIST 全文提取 Coverage Matrix 的 Test Item 集合。"""
    section = parse_section(text, "### Coverage Matrix")
    rows = parse_markdown_table(section)
    return {r["Test Item"] for r in rows if r.get("Test Item")}


def parse_negative_matrix(text):
    """从 CHECKLIST 全文提取 Negative Coverage Matrix 的 Test Item 集合。"""
    section = parse_section(text, "### Negative Coverage Matrix")
    rows = parse_markdown_table(section)
    return {r["Test Item"] for r in rows if r.get("Test Item")}


def parse_add_delete_log(text):
    """从 CHECKLIST 全文解析 Finalization Add/Delete Log。
    返回 (deletes: {item_name: reason}, adds: {item_name: reason}, raw_lines: [str])。
    格式契约: `Delete <exact_method_name>: <reason>` / `Add <exact_method_name>: <reason>`
    """
    deletes = {}
    adds = {}
    raw_lines = []

    section = parse_section(text, "### Finalization Add/Delete Log")
    if not section:
        return deletes, adds, raw_lines

    for line in section.strip().split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(">") or stripped.startswith("<!--"):
            continue

        raw_lines.append(stripped)

        # 匹配 Delete <exact_method_name>: <reason>
        m_del = re.match(r'^Delete\s+(\S+)\s*:\s*(.*)', stripped)
        if m_del:
            name = m_del.group(1).strip()
            reason = m_del.group(2).strip()
            deletes[name] = reason
            continue

        # 匹配 Add <exact_method_name>: <reason>
        m_add = re.match(r'^Add\s+(\S+)\s*:\s*(.*)', stripped)
        if m_add:
            name = m_add.group(1).strip()
            reason = m_add.group(2).strip()
            adds[name] = reason
            continue

    return deletes, adds, raw_lines


# ─── 对账核心 ──────────────────────────────────────────────────────

def reconcile(checklist_path, test_file_path):
    """执行对账, 返回 (exit_code, report_lines)。"""
    cl_text = Path(checklist_path).read_text(encoding="utf-8")
    pos_methods, neg_methods = collect_test_methods(test_file_path)
    all_methods = pos_methods | neg_methods
    seed_violations = check_seeds(test_file_path)

    pos_matrix = parse_coverage_matrix(cl_text)
    neg_matrix = parse_negative_matrix(cl_text)
    deletes, adds, raw_log_lines = parse_add_delete_log(cl_text)

    report = []
    pos_miss = 0
    neg_miss = 0
    undeleted = 0
    extra_warn = 0
    bad_log_warn = 0

    # ── 正向对账 ──
    report.append("=" * 60)
    report.append("正向覆盖对账 (Coverage Matrix vs test_*)")
    report.append("=" * 60)
    report.append(f"  矩阵预期: {len(pos_matrix)} 项")
    report.append(f"  代码实现: {len(pos_methods)} 个 test_ 方法")
    report.append(f"  删除日志: {len(deletes)} 条 Delete 记录")
    report.append("")

    for item in sorted(pos_matrix):
        if item not in all_methods:
            if item in deletes:
                report.append(f"  [DELETED-OK] {item} — 日志: Delete {item}: {deletes[item]}")
            else:
                report.append(f"  [MISS] {item} — 矩阵有, 代码无, 日志无")
                report.append(f"     → 二选一: (a) 补该 test_ 方法")
                report.append(f"               (b) 若该行不该测(平台专属/参数不适用), 在 CHECKLIST")
                report.append(f"                  ### Finalization Add/Delete Log 记 `Delete {item}: <原因>`")
                report.append(f"     补日志后重跑 → 转 [DELETED-OK] 放行")
                report.append("")
                pos_miss += 1
                undeleted += 1

    # 正向 EXTRA: 代码有但矩阵无且日志无 Add
    for method in sorted(pos_methods):
        if method not in pos_matrix and method not in adds:
            report.append(f"  [EXTRA] {method} — 代码有, 矩阵无, 日志无 Add 记录")
            extra_warn += 1

    report.append("")

    # ── 负向对账 ──
    report.append("=" * 60)
    report.append("负向覆盖对账 (Negative Coverage Matrix vs test_neg_*)")
    report.append("=" * 60)
    report.append(f"  矩阵预期: {len(neg_matrix)} 项")
    report.append(f"  代码实现: {len(neg_methods)} 个 test_neg_ 方法")
    report.append("")

    for item in sorted(neg_matrix):
        if item not in neg_methods:
            if item in deletes:
                report.append(f"  [DELETED-OK] {item} — 日志: Delete {item}: {deletes[item]}")
            else:
                report.append(f"  [MISS] {item} — 矩阵有, 代码无, 日志无")
                report.append(f"     → 二选一: (a) 补该 test_neg_ 方法")
                report.append(f"               (b) 若该行不该测, 在 CHECKLIST")
                report.append(f"                  ### Finalization Add/Delete Log 记 `Delete {item}: <原因>`")
                report.append("")
                neg_miss += 1
                undeleted += 1

    for method in sorted(neg_methods):
        if method not in neg_matrix and method not in adds:
            report.append(f"  [EXTRA] {method} — 代码有, 矩阵无, 日志无 Add 记录")
            extra_warn += 1

    report.append("")

    # ── BAD-LOG 检测 ──
    all_matrix_items = pos_matrix | neg_matrix
    for name in sorted(deletes):
        if name not in all_matrix_items:
            report.append(f"  [BAD-LOG] Delete '{name}' 不匹配任何矩阵行 — 方法名写错或用了自然语言描述")
            bad_log_warn += 1
    for name in sorted(adds):
        if name not in all_matrix_items and name not in all_methods:
            report.append(f"  [BAD-LOG] Add '{name}' 不匹配任何矩阵行或代码方法")
            bad_log_warn += 1

    if bad_log_warn > 0:
        report.append("")

    # ── 汇总 ──
    report.append("=" * 60)
    report.append("对账汇总")
    report.append("=" * 60)

    if pos_miss > 0:
        report.append(f"  ❌ 正向 MISS: {pos_miss} 项(矩阵预期但代码缺失且无删除日志)")
    if neg_miss > 0:
        report.append(f"  ❌ 负向 MISS: {neg_miss} 项(矩阵预期但代码缺失且无删除日志)")
    if undeleted > 0:
        report.append(f"  ❌ 无日志删除: {undeleted} 项(需补代码或补 Delete 日志)")

    if pos_miss == 0 and neg_miss == 0:
        report.append("  ✅ 正向/负向全覆盖: 所有矩阵行均已落地或已记录删除")

    if extra_warn > 0:
        report.append(f"  ⚠️  EXTRA 警告: {extra_warn} 项(代码有但矩阵无, 需补 Add 日志或删代码)")

    if bad_log_warn > 0:
        report.append(f"  ⚠️  BAD-LOG 警告: {bad_log_warn} 项(Delete/Add 日志项不匹配任何矩阵行, 格式可能不符合契约)")

    if raw_log_lines:
        n_del = len([l for l in raw_log_lines if l.startswith("Delete")])
        n_add = len([l for l in raw_log_lines if l.startswith("Add")])
        report.append(f"  📋 删除日志已记录: {n_del} 条 Delete, {n_add} 条 Add")
    else:
        report.append("  📋 删除日志: 空(未找到 ### Finalization Add/Delete Log 段或无有效条目)")

    # ── 随机种子 hard gate ──
    if seed_violations:
        report.append("")
        report.append("=" * 60)
        report.append("随机种子检查 (SKILL.md §4, hard gate)")
        report.append("=" * 60)
        for name in seed_violations:
            report.append(f"  [NO-SEED] {name} — 使用 randn/rand/randint(含所调 helper)"
                          f"但方法体内无 manual_seed/np.random.seed")
        report.append(f"  → 在每个上述 test_ 方法起始加 torch.manual_seed(0)"
                      f"(或 np.random.seed(0)), 保证确定可复现。")
        report.append(f"  ❌ 种子缺失: {len(seed_violations)} 项")

    blocked = undeleted > 0 or bool(seed_violations)

    if blocked:
        report.append("")
        reasons = []
        if undeleted > 0:
            reasons.append("[MISS] 项")
        if seed_violations:
            reasons.append("[NO-SEED] 项")
        report.append(f"❌ reconcile 阻断 — 请处理上述 {' 和 '.join(reasons)} 后重试。")
    else:
        report.append("")
        report.append("✅ reconcile 通过 — 可进入 Step 6 执行测试。")

    return (1 if blocked else 0), report


# ─── main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="覆盖对账 gate — 校验 UT 文件方法是否完整覆盖 CHECKLIST 矩阵"
    )
    p.add_argument("--checklist", required=True, help="CHECKLIST.md 路径")
    p.add_argument("--test-file", required=True, help="test_npu_{op}.py 路径")
    args = p.parse_args()

    cl = Path(args.checklist)
    tf = Path(args.test_file)

    if not cl.exists():
        print(f"[ERROR] CHECKLIST.md 不存在: {args.checklist}")
        return 2
    if not tf.exists():
        print(f"[ERROR] 测试文件不存在: {args.test_file}")
        return 2

    exit_code, report = reconcile(str(cl), str(tf))
    for line in report:
        print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
