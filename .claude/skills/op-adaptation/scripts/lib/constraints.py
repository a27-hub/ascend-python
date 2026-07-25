"""C-constraints: check that .md constraints are implemented as TORCH_CHECK in C++."""

import re
from pathlib import Path
from typing import Any


def run_check_constraints(parse_json: dict, cpp_file: str) -> dict:
    """Check constraint coverage: each constraint should have a corresponding TORCH_CHECK.

    Args:
        parse_json: Output from op_adapt.py parse (contains constraints_logic)
        cpp_file: Path to the C++ implementation file

    Returns:
        {
            "constraints_total": int,
            "covered": int,
            "uncovered": [...],
            "torch_checks_found": [...],
        }
    """
    constraints = parse_json.get("constraints_logic", [])
    cpp_content = Path(cpp_file).read_text()

    # Skip noise rows the parse step already flagged (param descriptions, error
    # codes, workspace/executor/stream) — they aren't constraints to implement
    # as TORCH_CHECK. Rows without the flag are kept (treated as checkable).
    checkable = [c for c in constraints if c.get("is_likely_constraint") is not False]
    noise_skipped = len(constraints) - len(checkable)

    # Extract all TORCH_CHECK statements
    torch_checks = _extract_torch_checks(cpp_content)

    # For each checkable constraint, try to match it with a TORCH_CHECK
    covered = []
    uncovered = []

    for c in checkable:
        rule = c["rule"]
        matched = _match_constraint_to_check(rule, torch_checks)
        if matched:
            covered.append({
                "rule": rule[:100],
                "confidence": c["confidence"],
                "matched_check": matched,
            })
        else:
            uncovered.append({
                "rule": rule[:100],
                "confidence": c["confidence"],
                "md_line": c.get("md_line"),
            })

    return {
        "constraints_total": len(checkable),
        "noise_skipped": noise_skipped,
        "covered": len(covered),
        "uncovered": len(uncovered),
        "covered_details": covered,
        "uncovered_details": uncovered,
        "torch_checks_found": torch_checks,
    }


# TORCH_CHECK call macros (excludes TORCH_CHECK_NAME, which registers a name,
# not a runtime check). Add variants here as CANN/PyTorch evolves.
_TORCH_CHECK_MACRO_RE = re.compile(r'TORCH_CHECK(?:_VALUE|_INDEX|_WITH|_LT|_LE|_GT|_GE|_EQ|_NE)?\s*\(')


def _extract_balanced_block(content: str, open_pos: int) -> str | None:
    """Return content between the '(' at open_pos and its matching ')'.

    Respects string literals (so ';' or '(' inside a message string do not
    truncate or skew depth) and nested parens (e.g. OPS_ERROR(ErrCode::PARAM)).
    Returns None if unbalanced (e.g. truncated file).
    """
    depth = 0
    in_string = False
    escape = False
    i = open_pos
    n = len(content)
    while i < n:
        ch = content[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return content[open_pos + 1:i]
        i += 1
    return None


def _extract_torch_checks(content: str) -> list[dict]:
    """Extract all TORCH_CHECK* statements from C++ code.

    Matches the TORCH_CHECK macro family (TORCH_CHECK, TORCH_CHECK_VALUE,
    TORCH_CHECK_INDEX, TORCH_CHECK_LT, ...) and uses balanced-paren scanning
    rather than `[^;]+`, so message strings containing ';' and nested macro
    args (OPS_ERROR(...)) are not truncated.
    """
    checks = []
    for m in _TORCH_CHECK_MACRO_RE.finditer(content):
        open_pos = m.end() - 1  # position of '('
        inner = _extract_balanced_block(content, open_pos)
        if inner is None:
            continue
        full = inner.strip()
        # Extract the message: the LAST string literal in the statement.
        # Convention: TORCH_CHECK(cond, "msg", OPS_ERROR(...)) — the message
        # string follows the condition; OPS_ERROR is not a string literal.
        # Taking the last string avoids grabbing a string inside the condition
        # (e.g. TORCH_CHECK(strcmp(s, "x") == 0, "msg") → "msg", not "x").
        msg_matches = re.findall(r'"([^"]+)"', full)
        msg = msg_matches[-1] if msg_matches else full[:80]
        macro = m.group(0).rstrip('(').strip()
        checks.append({
            "condition": full[:200],
            "message": msg,
            "line": content[:m.start()].count("\n") + 1,
            "macro": macro,
        })

    return checks


def _match_constraint_to_check(rule: str, torch_checks: list[dict]) -> dict | None:
    """Try to match a constraint rule to a TORCH_CHECK statement.

    Uses keyword matching: if significant words from the constraint appear
    in the TORCH_CHECK message, it's considered covered.
    """
    # Extract keywords from the rule (Chinese + English)
    keywords = _extract_keywords(rule)
    if not keywords:
        return None

    for check in torch_checks:
        check_text = check["message"] + " " + check["condition"]
        # Check if any keyword appears in the TORCH_CHECK
        matched_kw = [kw for kw in keywords if kw.lower() in check_text.lower()]
        if matched_kw:
            return {
                "message": check["message"],
                "line": check["line"],
                "matched_keywords": matched_kw,
            }

    return None


def _extract_keywords(rule: str) -> list[str]:
    """Extract significant keywords from a constraint rule for matching.

    Includes bilingual keyword mapping: Chinese constraint terms → English
    TORCH_CHECK keywords, so that .md (Chinese) can match C++ (English).
    """
    # Bilingual keyword map: Chinese → English equivalents
    ZH_TO_EN = {
        "空": "empty",
        "维度": "dim",
        "维": "dim",
        "数据类型": "dtype",
        "类型": "type",
        "格式": "format",
        "连续": "contiguous",
        "整除": "divisible",
        "偶数": "even",
        "最大值": "max",
        "最小值": "min",
        "范围": "range",
        "不支持": "not support",
        "仅支持": "only support",
        "必须": "must",
        "不能": "cannot",
        "大于": "greater",
        "小于": "less",
        "等于": "equal",
        "非负": "non-negative",
        "正数": "positive",
        "零": "zero",
        "形状": "shape",
        "输入": "input",
        "输出": "output",
        "参数": "param",
        "属性": "attr",
        "量化": "quant",
        "缩放": "scale",
        "偏移": "offset",
        "分组": "group",
        "索引": "index",
    }

    # Remove common words and table separators
    rule = re.sub(r'[|\-：:,]', ' ', rule)

    # Extract meaningful tokens
    tokens = rule.split()
    keywords = []

    skip = {"tensor", "acl", "int", "float", "bool", "nd", "-", "√", "×", "参数", "约束",
            "输入", "输出", "描述", "使用说明", "数据类型", "数据格式", "维度",
            "非连续", "参数名"}

    for t in tokens:
        t = t.strip("（()）{}[]")
        if len(t) >= 2 and t.lower() not in skip and not t.startswith("-"):
            keywords.append(t)
            # Add English equivalent if this is a Chinese keyword
            for zh, en in ZH_TO_EN.items():
                if zh in t:
                    keywords.append(en)

    return keywords  # no truncation: recall > precision for a coverage DRAFT the LLM verifies
