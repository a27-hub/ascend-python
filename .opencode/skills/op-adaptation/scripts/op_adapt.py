#!/usr/bin/env python3
"""op_adapt.py — Unified CLI for op-adaptation skill scripts.

Usage:
    op_adapt.py parse             --md <path> --def <path> [--op-name <name>]
    op_adapt.py verify            --op-name <name> [--cpp-file <path>]
    op_adapt.py discover          --op-name <name>
    op_adapt.py gen-baseline      --op-name <name> [--write]
    op_adapt.py check-constraints --parse-json <path> --cpp-file <path>
    op_adapt.py gen-summary       --op-name <name> [--parse-json <path>] [--verify-json <path>]
    op_adapt.py gen-aclgraph      --op-name <name> [--out-names <a,b,...>]
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Ensure lib/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.parser import run_parse
from lib.signature import verify_op
from lib.discover import run_discover
from lib.baseline import run_gen_baseline
from lib.constraints import run_check_constraints
from lib.summary import run_gen_summary
from lib.aclgraph import run_gen_aclgraph
from lib.common import find_repo_root


def cmd_parse(args):
    """C-parse: parse .md + _def.cpp → JSON output."""
    try:
        result = run_parse(args.md, args.def_path, args.op_name)
        out = json.dumps(result, indent=2, ensure_ascii=False)
        print(out)
        save_path = args.save or f"notes/{result.get('npu_name','unknown')}.parse.json"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(out)
        return 0 if "error" not in result else 1
    except (FileNotFoundError, IOError) as e:
        print(json.dumps({"error": str(e)}))
        return 1


def cmd_verify(args):
    """C-verify: cross-check signatures across YAML/C++/Meta/Derivatives."""
    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = None

    report = verify_op(args.op_name, repo_root, args.cpp_file)
    out = json.dumps(report, indent=2, ensure_ascii=False)
    print(out)
    save_path = args.save or f"notes/{args.op_name}.verify.json"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(out)
    return 0 if report["passed"] else 1


def cmd_discover(args):
    """M-discover: grep search for an operator across the codebase."""
    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = None

    result = run_discover(args.op_name, repo_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_gen_baseline(args):
    """C-baseline: generate schema JSON + allowlist entries."""
    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = None

    result = run_gen_baseline(args.op_name, repo_root, args.write,
                              allowlist_list=args.allowlist_list)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if "error" not in result else 1


def _load_json_default(path: str | None, default_path: str) -> dict | None:
    """Load JSON from path if provided, else try default_path. Returns None if neither exists."""
    p = Path(path) if path else Path(default_path)
    if not p.exists():
        if path:
            raise FileNotFoundError(f"JSON file not found: {path}")
        return None
    return json.loads(p.read_text())


def cmd_check_constraints(args):
    """C-constraints: check constraint coverage in C++ TORCH_CHECK."""
    try:
        default_parse = f"notes/{args.op_name}.parse.json" if args.op_name else None
        if not args.parse_json and not default_parse:
            print(json.dumps({"error": "Either --parse-json or --op-name must be provided"}))
            return 1
        parse_json = _load_json_default(args.parse_json, default_parse)
        if parse_json is None:
            print(json.dumps({"error": f"Parse JSON not found at {default_parse}; pass --parse-json explicitly"}))
            return 1
        result = run_check_constraints(parse_json, args.cpp_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
        print(json.dumps({"error": str(e)}))
        return 1


def cmd_gen_summary(args):
    """C-summary: generate integration summary."""
    try:
        parse_json = None
        default_parse = f"notes/{args.op_name}.parse.json"
        if args.parse_json:
            parse_json = json.loads(Path(args.parse_json).read_text())
        else:
            p = Path(default_parse)
            if p.exists():
                parse_json = json.loads(p.read_text())

        verify_report = None
        default_verify = f"notes/{args.op_name}.verify.json"
        if args.verify_json:
            verify_report = json.loads(Path(args.verify_json).read_text())
        else:
            p = Path(default_verify)
            if p.exists():
                verify_report = json.loads(p.read_text())

        result = run_gen_summary(args.op_name, parse_json, verify_report, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if "error" not in result else 1
    except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
        print(json.dumps({"error": str(e)}))
        return 1


def cmd_gen_aclgraph(args):
    """gen-aclgraph: 生成 ACLGraph 手工 update 的 2 接口 YAML 草稿 + C++ 骨架（rules: yaml-guide/cpp-guide 对应小节）。"""
    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = None

    out_names = [n.strip() for n in args.out_names.split(",") if n.strip()] if args.out_names else None
    result = run_gen_aclgraph(args.op_name, repo_root, out_names=out_names)
    if "error" in result:
        print(json.dumps({"error": result["error"]}, ensure_ascii=False))
        return 1
    print(result["text"])
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="op_adapt",
        description="op-adaptation skill scripts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="Parse .md + _def.cpp → JSON")
    p_parse.add_argument("--md", required=True, help="Path to aclnnXxx.md")
    p_parse.add_argument("--def", dest="def_path", required=True, help="Path to xxx_def.cpp")
    p_parse.add_argument("--op-name", default=None, help="Operator name override")
    p_parse.add_argument("--save", default=None, help="Save JSON to path (default: notes/{op}.parse.json)")
    p_parse.set_defaults(func=cmd_parse)

    p_verify = sub.add_parser("verify", help="Cross-verify signatures")
    p_verify.add_argument("--op-name", required=True, help="Operator name")
    p_verify.add_argument("--cpp-file", default=None, help="Explicit C++ file path")
    p_verify.add_argument("--save", default=None, help="Save JSON to path (default: notes/{op}.verify.json)")
    p_verify.set_defaults(func=cmd_verify)

    p_discover = sub.add_parser("discover", help="Grep search for operator in codebase")
    p_discover.add_argument("--op-name", required=True, help="Operator name (without version suffix)")
    p_discover.set_defaults(func=cmd_discover)

    p_baseline = sub.add_parser("gen-baseline", help="Generate schema JSON + allowlist entries")
    p_baseline.add_argument("--op-name", required=True, help="Operator name")
    p_baseline.add_argument("--write", action="store_true", help="Write to files (default: dry-run)")
    p_baseline.add_argument("--allowlist-list", default=None,
                            help="Target allowlist list name (e.g. all_version, v2.1.0). "
                                 "Set this for version-specific ops so the script does a "
                                 "safe formatting-preserving insert. NEVER hand-edit the JSON.")
    p_baseline.set_defaults(func=cmd_gen_baseline)

    p_constraints = sub.add_parser("check-constraints", help="Check constraint coverage in C++")
    p_constraints.add_argument("--parse-json", default=None, help="Path to parse output JSON (default: notes/{op-name}.parse.json)")
    p_constraints.add_argument("--op-name", default=None, help="Operator name (used to locate default parse JSON)")
    p_constraints.add_argument("--cpp-file", required=True, help="Path to C++ implementation file")
    p_constraints.set_defaults(func=cmd_check_constraints)

    p_summary = sub.add_parser("gen-summary", help="Generate integration summary")
    p_summary.add_argument("--op-name", required=True, help="Operator name")
    p_summary.add_argument("--parse-json", default=None, help="Path to parse output JSON")
    p_summary.add_argument("--verify-json", default=None, help="Path to verify output JSON")
    p_summary.add_argument("--dry-run", action="store_true", help="Output to stdout only, don't write file")
    p_summary.set_defaults(func=cmd_gen_summary)

    p_aclgraph = sub.add_parser("gen-aclgraph",
                                help="Generate ACLGraph .out/_get_max_workspace YAML drafts + C++ skeletons")
    p_aclgraph.add_argument("--op-name", required=True, help="Functional operator name")
    p_aclgraph.add_argument("--out-names", default=None,
                            help="Comma-separated alias output names (default: reuse existing "
                                 ".out entry names if present, else out0/out1 placeholders)")
    p_aclgraph.set_defaults(func=cmd_gen_aclgraph)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
