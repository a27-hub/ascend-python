"""C-summary: generate integration summary by filling the template.

The script writes the FULL template framework (all sections of
summary-template.md), filling only the deterministic parts:
  - identity fields (operator name, date, aclnn name)
  - C-parse parameter/conflict/constraint tables (from parse JSON)
  - C-verify results (from verify report)
Semantic / session-specific sections (functionality description, per-step
walkthrough for C-confirm/C-yaml/C-cpp/C-meta/C-build/C-test/C-docs/C-baseline,
file list, build/test results, key technical points, notes) are left as
placeholders for the LLM to complete per summary-template.md.
"""

from pathlib import Path
from datetime import datetime


TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent / "references" / "templates" / "summary-template.md"

# Exact template blocks the script replaces with filled content.
_PARSE_BLOCK = (
    "### C-parse: Parse Input Files\n"
    "- Input file paths\n"
    "- Parse results (parameters, constraints)"
)
_VERIFY_BLOCK = (
    "### C-verify: Signature Verification\n"
    "- Verification result (Passed/Failed)\n"
    "- Mismatches (if any)"
)


def run_gen_summary(op_name: str, parse_json: dict | None = None,
                    verify_report: dict | None = None,
                    notes_dir: str = "notes",
                    dry_run: bool = False) -> dict:
    """Generate an integration summary by filling summary-template.md.

    Args:
        op_name: Operator name (e.g. npu_fast_gelu)
        parse_json: Output from op_adapt.py parse (optional)
        verify_report: Output from op_adapt.py verify (optional)
        notes_dir: Directory for summary output
        dry_run: If True, return content without writing

    Returns:
        {op_name, summary_path, content, dry_run, written}
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    summary_path = Path(notes_dir) / f"summary_{date_str}.md"

    tpl = TEMPLATE_PATH.read_text()

    # ── Identity substitutions (deterministic) ──
    tpl = tpl.replace("# {OperatorName} Operator Integration Summary",
                      f"# {op_name} Operator Integration Summary")
    aclnn_name = parse_json.get("aclnn_name") if parse_json else None
    if aclnn_name:
        tpl = tpl.replace("**Operator Name**: aclnn{OperatorName}",
                          f"**Operator Name**: {aclnn_name}")
    tpl = tpl.replace("**Date**: {YYYY-MM-DD}", f"**Date**: {date_str}")
    tpl = tpl.replace("**torch_npu Function Name**: npu_{operator_name}",
                      f"**torch_npu Function Name**: {op_name}")

    # ── C-parse section: fill parameter/conflict/constraint table ──
    warnings = []
    if parse_json:
        if _PARSE_BLOCK in tpl:
            tpl = tpl.replace(_PARSE_BLOCK, _build_parse_section(parse_json))
        else:
            warnings.append(
                "C-parse block not found in template (template drifted from script's _PARSE_BLOCK) — "
                "C-parse section left as placeholder; fill the parameter/conflict/constraint table manually."
            )

    # ── C-verify section: fill results ──
    if verify_report is not None:
        if _VERIFY_BLOCK in tpl:
            tpl = tpl.replace(_VERIFY_BLOCK, _build_verify_section(verify_report))
        else:
            warnings.append(
                "C-verify block not found in template (template drifted from script's _VERIFY_BLOCK) — "
                "C-verify section left as placeholder; fill the verification result manually."
            )

    content = tpl

    # Append to existing file if same date, but REPLACE same op's section (H14 fix:
    # same op same-day re-run should not duplicate content)
    if summary_path.exists() and not dry_run:
        existing = summary_path.read_text()
        op_header = f"# {op_name} Operator Integration Summary"
        if op_header in existing:
            # Replace existing section for this op (from header to next --- or EOF)
            import re
            pattern = re.escape(op_header) + r'.*?(?=\n---\n|\Z)'
            content = re.sub(pattern, content.lstrip(), existing, flags=re.DOTALL)
        else:
            # Different op — append
            content = existing + "\n\n---\n\n" + content

    if dry_run:
        return {
            "op_name": op_name,
            "summary_path": str(summary_path),
            "content": content,
            "dry_run": True,
            "written": False,
            "warnings": warnings,
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(content)

    return {
        "op_name": op_name,
        "summary_path": str(summary_path),
        "content": content,
        "dry_run": False,
        "written": True,
        "warnings": warnings,
    }


def _build_parse_section(parse_json: dict) -> str:
    """Build the filled C-parse section (deterministic, from parse JSON)."""
    pm = parse_json.get("parse_meta", {})
    params = parse_json.get("parameters", [])
    conflicts = parse_json.get("conflicts", [])
    constraints = parse_json.get("constraints_logic", [])

    lines = ["### C-parse: Parse Input Files"]
    lines.append(f"- Input files: `{pm.get('md_path', '?')}`, `{pm.get('def_path', '?')}`")
    lines.append(f"- aclnn function: `{parse_json.get('aclnn_name', '?')}`")
    lines.append(f"- Operator type: {parse_json.get('op_type', '?')}")
    lines.append(f"- Parameters: {len(params)}")
    lines.append("")
    lines.append("| Parameter | Source | Type | Optionality | Category |")
    lines.append("|-----------|--------|------|-------------|----------|")
    for p in params:
        lines.append(
            f"| {p.get('name', '?')} | {p.get('source', '?')} | {p.get('type', '?')} | "
            f"{p.get('optionality', '?')} | {p.get('category', '?')} |"
        )
    lines.append("")

    if conflicts:
        lines.append(f"**Conflicts ({len(conflicts)}):**")
        for c in conflicts:
            lines.append(f"- ⚠️ {c.get('param', '?')}: {c.get('desc', '?')}")
        lines.append("")

    if constraints:
        likely = [c for c in constraints if c.get("is_likely_constraint") is not False]
        lines.append(f"**Constraints ({len(constraints)} extracted, {len(likely)} likely):**")
        for c in likely[:5]:
            lines.append(f"- [{c.get('confidence', '?')}] {c.get('rule', '?')[:80]}")
        if len(likely) > 5:
            lines.append(f"- ... and {len(likely) - 5} more")
        lines.append("")

    return "\n".join(lines)


def _build_verify_section(verify_report: dict) -> str:
    """Build the filled C-verify section (deterministic, from verify report)."""
    passed = verify_report.get("passed", False)
    mismatches = verify_report.get("mismatches", [])
    known = verify_report.get("known_inconsistent", False)
    warn_count = verify_report.get("warning_count", 0)

    lines = ["### C-verify: Signature Verification"]
    status = "✅ Passed" if passed else "❌ Failed"
    suffix = " (known_inconsistent)" if known else ""
    lines.append(f"- Verification result: {status}{suffix}")
    lines.append(f"- Mismatches: {len(mismatches)} total, {warn_count} warnings")
    if mismatches:
        lines.append("")
        lines.append("| Check | Severity | Error |")
        lines.append("|-------|----------|-------|")
        for m in mismatches:
            lines.append(
                f"| {m.get('check', '?')} | {m.get('severity', 'FAIL')} | {m.get('error', '?')[:80]} |"
            )
        lines.append("")

    return "\n".join(lines)
