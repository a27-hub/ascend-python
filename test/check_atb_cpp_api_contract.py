#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Checks ATB C++ source contracts, exported symbols, and a torch_npu-compatible
ABI smoke build.

Typical usage:

  python test/check_atb_cpp_api_contract.py --check-source
  python test/check_atb_cpp_api_contract.py --check-symbols --so /path/to/libop_plugin_atb.so
  python test/check_atb_cpp_api_contract.py --check-absent-symbols --so /path/to/libtorch_npu.so
  python test/check_atb_cpp_api_contract.py --artifact-root /path/to/op-plugin-artifacts --check-symbols
  python test/check_atb_cpp_api_contract.py --check-example-build --so /path/to/libop_plugin_atb.so \
      --torch-include-dir /path/to/libtorch/include \
      --torch-npu-include-dir /path/to/torch_npu_root_or_include_root \
      --torch-lib-dir /path/to/libtorch/lib \
      --cpp test/atb_cpp_interface_smoke.cpp \
      --cxx g++

The example-build check is an ABI smoke, not a standalone SDK proof. It writes
artifacts into ./atb_cpp_example_build, recreates that directory before each
run, links minimal PyTorch base libraries, and uses
-Wl,--allow-shlib-undefined so the full libtorch_npu runtime closure is not
part of this check.
"""

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


EXIT_OK = 0
EXIT_CONTRACT_FAILED = 1
EXIT_ERROR = 2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "op_plugin" / "config" / "atb_ops.yaml"
DEFAULT_HEADER = REPO_ROOT / "op_plugin" / "include" / "atb_ops.h"
DEFAULT_GENERATED_HEADER = REPO_ROOT / "op_plugin" / "include" / "AtbOpsInterface.h"
DEFAULT_SOURCE = REPO_ROOT / "op_plugin" / "ops" / "atb" / "AtbOpsInterface.cpp"
DEFAULT_BUILD_GENERATED_HEADER = REPO_ROOT / "build" / "pytorch" / "third_party" / "op-plugin" / "op_plugin" / "include" / "AtbOpsInterface.h"
DEFAULT_BUILD_SOURCE = REPO_ROOT / "build" / "pytorch" / "third_party" / "op-plugin" / "op_plugin" / "ops" / "atb" / "AtbOpsInterface.cpp"
DEFAULT_PACKAGE_GENERATED_HEADER = REPO_ROOT / "build" / "pytorch" / "build" / "packages" / "torch_npu" / "include" / "third_party" / "op-plugin" / "op_plugin" / "include" / "AtbOpsInterface.h"
DEFAULT_PACKAGE_SOURCE = REPO_ROOT / "build" / "pytorch" / "build" / "packages" / "torch_npu" / "include" / "third_party" / "op-plugin" / "op_plugin" / "ops" / "atb" / "AtbOpsInterface.cpp"
DEFAULT_NAMESPACE = "at_npu::native::atb"
DEFAULT_SO_NAME = "libop_plugin_atb.so"
DEFAULT_ATB_SRC_DIR = REPO_ROOT / "op_plugin" / "ops" / "atb"
DEFAULT_CPP_EXAMPLE = REPO_ROOT / "test" / "atb_cpp_interface_smoke.cpp"
DEFAULT_BUILD_DIR_NAME = "atb_cpp_example_build"


@dataclass(frozen=True)
class AtbCppApiFunction:
    cpp_name: str
    impl_name: str


@dataclass(frozen=True)
class ResolvedPaths:
    config: Path
    header: Optional[Path]
    source: Optional[Path]
    so: Optional[Path]


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_step(message: str) -> None:
    print(f"[STEP] {message}")


def print_sample(title: str, lines: Sequence[str]) -> None:
    print(f"[SNAPSHOT] {title}")
    if not lines:
        print("  <empty>")
        return
    for line in lines:
        print(f"  {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated ATB C++ API declarations, wrapper contracts, and smoke checks in torch_npu-compatible environments."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to op_plugin/config/atb_ops.yaml.")
    parser.add_argument("--header", help="Path to op_plugin/include/atb_ops.h.")
    parser.add_argument("--source", help="Path to op_plugin/ops/atb/AtbOpsInterface.cpp.")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Expected public C++ namespace.")
    parser.add_argument("--artifact-root", help="Root directory containing built op-plugin artifacts.")
    parser.add_argument("--check-source", action="store_true", help="Check generated header/source contract.")
    parser.add_argument("--check-symbols", action="store_true", help="Check that the target SO exports the expected ATB public symbols.")
    parser.add_argument("--check-absent-symbols", action="store_true", help="Check that the target SO does not export any ATB public symbols.")
    parser.add_argument("--check-example-build", action="store_true", help="Check that the documented C++ smoke can compile and link in a torch_npu-compatible environment.")
    parser.add_argument("--so", help="Path to the target shared library, for example libop_plugin_atb.so, libtorch_npu.so, or libnpu_profiler.so.")
    parser.add_argument("--cxx", default="g++", help="Path to the C++ compiler for --check-example-build.")
    parser.add_argument("--cpp", help="Path to the C++ source file used by --check-example-build.")
    parser.add_argument("--torch-include-dir", help="Path to the libtorch include root that contains ATen/ and c10/.")
    parser.add_argument("--torch-npu-include-dir", help="Path to the torch_npu source root or install/include root used by the smoke build.")
    parser.add_argument("--torch-lib-dir", help="Path to the libtorch library directory that contains libc10.so, libtorch_cpu.so, and libtorch.so.")
    parser.add_argument("--extra-include-dir", action="append", default=[], help="Extra include directory for --check-example-build. Repeat as needed.")
    parser.add_argument("--nm", default="nm", help="Path to nm.")
    parser.add_argument("--cxxfilt", default="c++filt", help="Path to c++filt.")
    return parser.parse_args()


def print_error(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return EXIT_ERROR


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def resolve_tool(tool: str) -> str:
    if any(sep in tool for sep in ("/", "\\")):
        tool_path = Path(tool)
        if not tool_path.is_file():
            raise FileNotFoundError(f"tool not found: {tool}")
        return str(tool_path)

    resolved = shutil.which(tool)
    if resolved is None:
        raise FileNotFoundError(f"tool not found in PATH: {tool}")
    return resolved


def resolve_candidate(explicit: Optional[str], candidates: Sequence[Path]) -> Optional[Path]:
    if explicit:
        return Path(explicit).resolve()
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_candidate_dir(explicit: Optional[str], candidates: Sequence[Path]) -> Optional[Path]:
    if explicit:
        return Path(explicit).resolve()
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def dedupe_existing_dirs(candidates: Sequence[Path], *, predicate=None) -> List[Path]:
    roots: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_dir():
            continue
        if predicate is not None and not predicate(resolved):
            continue
        key = str(resolved)
        if key in seen:
            continue
        roots.append(resolved)
        seen.add(key)
    return roots


def resolve_paths(args: argparse.Namespace) -> ResolvedPaths:
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else None

    header_candidates = [DEFAULT_HEADER, DEFAULT_BUILD_GENERATED_HEADER, DEFAULT_PACKAGE_GENERATED_HEADER]
    source_candidates = [DEFAULT_SOURCE, DEFAULT_BUILD_SOURCE, DEFAULT_PACKAGE_SOURCE]
    so_candidates: List[Path] = []

    if artifact_root is not None:
        header_candidates = [
            artifact_root / "op_plugin" / "include" / "atb_ops.h",
            artifact_root / "include" / "atb_ops.h",
        ] + header_candidates
        source_candidates = [
            artifact_root / "op_plugin" / "ops" / "atb" / "AtbOpsInterface.cpp",
            artifact_root / "build" / "pytorch" / "third_party" / "op-plugin" / "op_plugin" / "ops" / "atb" / "AtbOpsInterface.cpp",
        ] + source_candidates
        so_candidates = [
            artifact_root / DEFAULT_SO_NAME,
            artifact_root / "lib" / DEFAULT_SO_NAME,
            artifact_root / "build" / DEFAULT_SO_NAME,
        ]

    resolved_header = resolve_candidate(args.header, header_candidates)
    if args.source:
        resolved_source = Path(args.source).resolve()
    else:
        inferred_source_candidates = list(source_candidates)
        if resolved_header is not None:
            inferred_source_candidates = [
                resolved_header.parent.parent / "ops" / "atb" / "AtbOpsInterface.cpp",
                resolved_header.parent.parent.parent / "op_plugin" / "ops" / "atb" / "AtbOpsInterface.cpp",
            ] + inferred_source_candidates
        resolved_source = resolve_candidate(None, inferred_source_candidates)

    return ResolvedPaths(
        config=Path(args.config).resolve(),
        header=resolved_header,
        source=resolved_source,
        so=resolve_candidate(args.so, so_candidates),
    )


def infer_torch_npu_include_dir(args: argparse.Namespace) -> Optional[Path]:
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else None
    candidates: List[Path] = []
    if artifact_root is not None:
        candidates.extend(
            [
                artifact_root,
                artifact_root / "include",
                artifact_root.parent,
                artifact_root.parent.parent,
            ]
        )
    candidates.extend(
        [
            REPO_ROOT.parent / "pytorch-wzq",
            REPO_ROOT / "build" / "pytorch",
        ]
    )
    return resolve_candidate_dir(args.torch_npu_include_dir, candidates)


def infer_op_plugin_include_roots(torch_npu_include_dir: Path) -> List[Path]:
    candidates = [
        torch_npu_include_dir / "third_party" / "op-plugin",
        torch_npu_include_dir / "include" / "third_party" / "op-plugin",
    ]
    return dedupe_existing_dirs(candidates)


def infer_torch_include_dir(args: argparse.Namespace) -> Optional[Path]:
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else None
    candidates: List[Path] = []
    if artifact_root is not None:
        candidates.extend(
            [
                artifact_root / "include",
                artifact_root.parent / "include",
                artifact_root.parent.parent / "include",
            ]
        )
    candidates.extend(
        [
            REPO_ROOT.parent / "pytorch-wzq" / "third_party" / "incdir" / "include",
            REPO_ROOT.parent / "pytorch-wzq" / "dist" / "include",
        ]
    )
    return resolve_candidate_dir(args.torch_include_dir, candidates)


def infer_extra_torch_include_dirs(torch_include_dir: Path) -> List[Path]:
    torch_root = torch_include_dir.parent.parent
    candidates = [
        torch_root,
        torch_root / "build",
        torch_root / "torch" / "csrc" / "api" / "include",
    ]
    return dedupe_existing_dirs(
        candidates,
        predicate=lambda resolved: (
            (resolved / "c10").is_dir()
            or (resolved / "c10" / "macros" / "cmake_macros.h").is_file()
            or (resolved / "torch").is_dir()
        ),
    )


def infer_python_torch_include_dirs() -> List[Path]:
    try:
        import torch  # type: ignore
    except ImportError:
        return []

    torch_root = Path(torch.__file__).resolve().parent
    candidates = [
        torch_root / "include",
        torch_root / "include" / "torch" / "csrc" / "api" / "include",
    ]
    return dedupe_existing_dirs(candidates)


def infer_library_dirs(args: argparse.Namespace, so_path: Path) -> List[Path]:
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else None
    candidates: List[Path] = [so_path.parent]
    torch_npu_include_dir = infer_torch_npu_include_dir(args)
    if torch_npu_include_dir is not None:
        candidates.append(torch_npu_include_dir / "third_party" / "op-plugin" / "op_plugin" / "third_party" / "atb" / "libs")
    if artifact_root is not None:
        candidates.extend(
            [
                artifact_root,
                artifact_root / "lib",
                artifact_root / "build" / "packages" / "torch_npu" / "lib",
                artifact_root / "third_party" / "op-plugin" / "op_plugin" / "third_party" / "atb" / "libs",
            ]
        )
    return dedupe_existing_dirs(candidates)


def infer_torch_library_dirs(args: argparse.Namespace) -> List[Path]:
    candidates: List[Path] = []
    if args.torch_lib_dir:
        candidates.append(Path(args.torch_lib_dir).resolve())

    for include_root in infer_python_torch_include_dirs():
        candidates.append(include_root.parent / "lib")

    torch_include_dir = infer_torch_include_dir(args)
    if torch_include_dir is not None:
        torch_root = torch_include_dir.resolve().parent.parent
        candidates.extend(
            [
                torch_root / "build" / "lib",
                torch_root / "torch" / "lib",
                torch_root / "lib",
            ]
        )

    return dedupe_existing_dirs(candidates, predicate=lambda resolved: (resolved / "libc10.so").exists())


def load_api_functions(config_path: Path) -> List[AtbCppApiFunction]:
    require_file(config_path, "ATB ops config")
    functions = []
    current = {}
    key_pattern = re.compile(r"^\s*(?:-\s+)?(cpp_name|impl_name):\s*[\"']?([^\"'#\s]+)[\"']?\s*(?:#.*)?$")

    def flush_current() -> None:
        if not current:
            return
        if "cpp_name" not in current or "impl_name" not in current:
            raise ValueError(f"incomplete ATB C++ API entry in {config_path}: {current}")
        functions.append(AtbCppApiFunction(cpp_name=current["cpp_name"], impl_name=current["impl_name"]))
        current.clear()

    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = key_pattern.match(line)
        if match is None:
            continue
        key, value = match.group(1), match.group(2)
        if key == "cpp_name" and current:
            flush_current()
        current[key] = value
    flush_current()

    if not functions:
        raise ValueError(f"no cpp_name/impl_name entries found in config: {config_path}")

    cpp_names = [fn.cpp_name for fn in functions]
    duplicates = sorted({name for name in cpp_names if cpp_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate cpp_name entries found: {', '.join(duplicates)}")

    print_info(f"loaded {len(functions)} ATB public API entries from {config_path}")
    print_sample("ATB API config entries", [f"{fn.cpp_name} -> {fn.impl_name}" for fn in functions])
    return functions


def normalize_schema(schema: str) -> str:
    schema = re.sub(r"\s+", " ", schema).strip()
    schema = schema.replace(",*", ", *")
    schema = re.sub(r"\*,\s*", "*, ", schema)
    return schema


def normalize_cpp_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_cpp_type_spacing(text: str) -> str:
    text = normalize_cpp_whitespace(text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    text = re.sub(r"\s*&\s*(?=[,>)])", "&", text)
    text = re.sub(r"&\s+([A-Za-z_])", r"&\1", text)
    return text


def extract_atb_registry(src_dir: Path, generated_source_name: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    def_pattern = re.compile(r'm\.def\((?P<body>(?:\s*"[^"]*"\s*)+)\);', re.S)
    impl_pattern = re.compile(r'm\.impl\("([^"]+)",\s*TORCH_FN\(atb::([A-Za-z0-9_]+)\)\);')
    string_pattern = re.compile(r'"([^"]*)"')
    schema_by_op_name: Dict[str, str] = {}
    op_names_by_impl_name: Dict[str, List[str]] = {}

    for path in sorted(src_dir.glob("*.cpp")):
        if path.name == generated_source_name:
            continue
        text = path.read_text(encoding="utf-8")
        for match in def_pattern.finditer(text):
            schema = normalize_schema("".join(string_pattern.findall(match.group("body"))))
            op_name = schema.split("(", 1)[0]
            schema_by_op_name[op_name] = schema
        for op_name, impl_name in impl_pattern.findall(text):
            op_names_by_impl_name.setdefault(impl_name, []).append(op_name)
    return schema_by_op_name, op_names_by_impl_name


def split_top_level(text: str, delimiter: str = ",") -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    paren_depth = 0
    angle_depth = 0
    square_depth = 0
    brace_depth = 0
    for char in text:
        if char == delimiter and all(depth == 0 for depth in (paren_depth, angle_depth, square_depth, brace_depth)):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def extract_argument_name(parameter: str) -> str:
    parameter = re.sub(r"\s*=\s*.*$", "", parameter.strip())
    name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", parameter)
    if name_match is None:
        raise ValueError(f"failed to extract argument name from parameter: {parameter}")
    return name_match.group(1)


def quote_command(command: Sequence[str]) -> str:
    return " ".join(command)


def canonicalize_cpp_signature(signature: str) -> str:
    signature = normalize_cpp_whitespace(signature)
    open_paren = signature.find("(")
    close_paren = signature.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren < open_paren:
        return normalize_cpp_type_spacing(signature)
    prefix = normalize_cpp_whitespace(signature[:open_paren])
    prefix = re.sub(r"([&>*])([A-Za-z_][A-Za-z0-9_]*)$", r"\1 \2", prefix)
    parameter_text = signature[open_paren + 1:close_paren].strip()
    if not parameter_text:
        return f"{prefix}()"
    parameters = [normalize_cpp_type_spacing(item) for item in split_top_level(parameter_text)]
    return f"{prefix}({', '.join(parameters)})"


def strip_signature_defaults(signature: str) -> str:
    open_paren = signature.find("(")
    close_paren = signature.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren < open_paren:
        return signature
    prefix = signature[:open_paren]
    parameter_text = signature[open_paren + 1:close_paren].strip()
    if not parameter_text:
        return signature
    stripped_parameters = [re.sub(r"\s*=\s*.*$", "", item).strip() for item in split_top_level(parameter_text)]
    return f"{prefix}({', '.join(stripped_parameters)})"


def validate_signature_parameters(signature_text: str) -> None:
    open_paren = signature_text.find("(")
    close_paren = signature_text.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren < open_paren:
        raise ValueError(f"failed to locate parameter list in signature: {signature_text}")
    parameter_text = signature_text[open_paren + 1:close_paren].strip()
    if not parameter_text:
        return
    for parameter in split_top_level(parameter_text):
        extract_argument_name(parameter)


def collect_signature_text(text: str, impl_short_name: str) -> str:
    # Known limitation: this controlled regex/text scan assumes ATB cpp
    # function definitions keep their current style. If headers switch to
    # trailing returns, macro-wrapped signatures, or other shapes, update
    # this parser together with the contract checks.
    pattern = re.compile(rf"^\s*(?P<prefix>.*\b{re.escape(impl_short_name)}\s*\()", re.M)
    signature_texts: List[str] = []

    for match in pattern.finditer(text):
        open_paren = text.find("(", match.start())
        if open_paren == -1:
            continue
        depth = 1
        idx = open_paren + 1
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            idx += 1
        if depth != 0:
            continue
        end_paren = idx - 1
        tail_idx = end_paren + 1
        while tail_idx < len(text) and text[tail_idx].isspace():
            tail_idx += 1
        if tail_idx >= len(text) or text[tail_idx] != "{":
            continue
        signature_texts.append(text[match.start():end_paren + 1])

    normalized = sorted(set(normalize_cpp_whitespace(item) for item in signature_texts))
    if not normalized:
        raise ValueError(f"no C++ function definition found for atb::{impl_short_name}")
    if len(normalized) != 1:
        raise ValueError(f"multiple C++ function definitions found for atb::{impl_short_name}: {normalized}")
    return normalized[0]


def extract_impl_signature(src_dir: Path, generated_source_name: str, impl_name: str) -> str:
    impl_short_name = impl_name.removeprefix("atb::")
    matches: List[str] = []
    for path in sorted(src_dir.glob("*.cpp")):
        if path.name == generated_source_name:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            signature_text = collect_signature_text(text, impl_short_name)
        except ValueError as exc:
            if f"atb::{impl_short_name}" not in str(exc):
                raise
            continue
        validate_signature_parameters(signature_text)
        matches.append(signature_text)

    unique = sorted(set(matches))
    if not unique:
        raise ValueError(f"no C++ implementation signature found for {impl_name}")
    if len(unique) != 1:
        raise ValueError(f"multiple distinct C++ implementation signatures found for {impl_name}: {unique}")
    return unique[0]


def extract_header_signature(header_text: str, cpp_name: str) -> str:
    match = re.search(
        rf"^\s*TORCH_NPU_API\s+(?P<signature>[^\n;]*\b{re.escape(cpp_name)}\s*\([^;]*\))\s*;",
        header_text,
        re.M,
    )
    if match is None:
        raise ValueError(f"missing exported header declaration: {cpp_name}")
    return canonicalize_cpp_signature(match.group("signature"))


def extract_signature_argument_names(signature: str) -> List[str]:
    open_paren = signature.find("(")
    close_paren = signature.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren < open_paren:
        raise ValueError(f"failed to locate parameter list in signature: {signature}")
    parameter_text = signature[open_paren + 1:close_paren].strip()
    if not parameter_text:
        return []
    return [extract_argument_name(parameter) for parameter in split_top_level(parameter_text)]


def check_config_consistency(functions: Sequence[AtbCppApiFunction], src_dir: Path, generated_source_name: str) -> List[str]:
    failures = []
    schema_by_op_name, op_names_by_impl_name = extract_atb_registry(src_dir, generated_source_name)
    resolved = []
    implementation_signatures = []
    for fn in functions:
        impl_short_name = fn.impl_name.removeprefix("atb::")
        op_names = sorted(set(op_names_by_impl_name.get(impl_short_name, [])))
        if not op_names:
            failures.append(f"no dispatcher registration found for impl_name {fn.impl_name}")
            continue
        if len(op_names) != 1:
            failures.append(f"impl_name {fn.impl_name} maps to multiple dispatcher ops: {', '.join(op_names)}")
            continue
        op_name = op_names[0]
        schema = schema_by_op_name.get(op_name)
        if schema is None:
            failures.append(f"dispatcher op {op_name} has no matching m.def schema")
            continue
        try:
            impl_signature = extract_impl_signature(src_dir, generated_source_name, fn.impl_name)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        resolved.append(f"{fn.cpp_name} -> {schema}")
        implementation_signatures.append(f"{fn.cpp_name} -> {canonicalize_cpp_signature(impl_signature)}")
    print_info(f"resolved {len(resolved)}/{len(functions)} config entries to unique ATB dispatcher schemas")
    print_sample("config to dispatcher schema matches", resolved)
    print_info(f"resolved {len(implementation_signatures)}/{len(functions)} config entries to unique ::atb implementation signatures")
    print_sample("config to implementation signature matches", implementation_signatures)
    return failures


def resolve_registered_schema_for_function(
    fn: AtbCppApiFunction,
    src_dir: Path,
    generated_source_name: str,
) -> str:
    schema_by_op_name, op_names_by_impl_name = extract_atb_registry(src_dir, generated_source_name)
    impl_short_name = fn.impl_name.removeprefix("atb::")
    op_names = sorted(set(op_names_by_impl_name.get(impl_short_name, [])))
    if not op_names:
        raise ValueError(f"no dispatcher registration found for impl_name {fn.impl_name}")
    if len(op_names) != 1:
        raise ValueError(f"impl_name {fn.impl_name} maps to multiple dispatcher ops: {', '.join(op_names)}")
    schema = schema_by_op_name.get(op_names[0])
    if schema is None:
        raise ValueError(f"dispatcher op {op_names[0]} has no matching m.def schema")
    return schema


def run_command(command: Sequence[str], *, input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"failed to execute {' '.join(command)}: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}{detail}")
    return result


def make_example_source() -> str:
    return (
        '#include "third_party/op-plugin/op_plugin/include/atb_ops.h"\n\n'
        "int main()\n"
        "{\n"
        "    auto fn = &at_npu::native::atb::_npu_group_topk;\n"
        "    (void)fn;\n"
        "    return 0;\n"
        "}\n"
    )


def resolve_cpp_example_path(args: argparse.Namespace) -> Path:
    if args.cpp:
        return Path(args.cpp).resolve()
    return DEFAULT_CPP_EXAMPLE.resolve()


def check_example_build(args: argparse.Namespace, so_path: Path) -> List[str]:
    require_file(so_path, "shared library")
    cxx_path = resolve_tool(args.cxx)
    cpp_example_path = resolve_cpp_example_path(args)
    require_file(cpp_example_path, "C++ example source file")
    torch_include_dir = infer_torch_include_dir(args)
    torch_npu_include_dir = infer_torch_npu_include_dir(args)
    if torch_include_dir is None:
        raise FileNotFoundError("torch include directory could not be resolved; pass --torch-include-dir")
    if torch_npu_include_dir is None:
        raise FileNotFoundError("torch_npu include root could not be resolved; pass --torch-npu-include-dir")

    op_plugin_include_roots = infer_op_plugin_include_roots(torch_npu_include_dir)
    if not op_plugin_include_roots:
        raise FileNotFoundError(
            "op-plugin include root could not be inferred from torch_npu include root; "
            "expected a third_party/op-plugin directory beneath it"
        )

    include_dirs = dedupe_existing_dirs(
        [torch_npu_include_dir] + op_plugin_include_roots + [torch_include_dir]
        + infer_extra_torch_include_dirs(torch_include_dir)
        + infer_python_torch_include_dirs()
        + [Path(item).resolve() for item in args.extra_include_dir]
    )
    library_dirs = infer_library_dirs(args, so_path)
    torch_library_dirs = infer_torch_library_dirs(args)

    print_step(f"checking documented example build against {so_path}")
    print_info(f"using C++ compiler: {cxx_path}")
    print_info(f"using C++ example source: {cpp_example_path}")
    print_sample("example include directories", [str(path) for path in include_dirs])
    print_sample("example library directories", [str(path) for path in library_dirs])
    print_sample("example torch library directories", [str(path) for path in torch_library_dirs])
    source_text = cpp_example_path.read_text(encoding="utf-8")
    print_sample("documented example source", source_text.splitlines())

    build_dir = Path.cwd() / DEFAULT_BUILD_DIR_NAME
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_dir / "atb_cpp_example"

    compile_command: List[str] = [cxx_path, str(cpp_example_path), "-o", str(output_path)]
    for include_dir in include_dirs:
        compile_command.extend(["-I", str(include_dir)])
    for library_dir in library_dirs:
        compile_command.extend(["-L", str(library_dir)])
        compile_command.append(f"-Wl,-rpath,{library_dir}")
        compile_command.append(f"-Wl,-rpath-link,{library_dir}")
    for library_dir in torch_library_dirs:
        compile_command.extend(["-L", str(library_dir)])
        compile_command.append(f"-Wl,-rpath,{library_dir}")
        compile_command.append(f"-Wl,-rpath-link,{library_dir}")
    compile_command.append("-Wl,--allow-shlib-undefined")
    compile_command.append(str(so_path))
    compile_command.extend(["-lc10", "-ltorch_cpu", "-ltorch"])

    print_info(f"example build output directory: {build_dir}")
    print_info(f"example build command: {quote_command(compile_command)}")
    run_command(compile_command)

    return []


def extract_single_namespace_block(text: str, namespace: str) -> str:
    namespace_match = re.search(rf"\bnamespace\s+{re.escape(namespace)}\s*\{{", text)
    if namespace_match is None:
        raise ValueError(f"namespace {namespace} not found")

    body_start = namespace_match.end()
    depth = 1
    idx = body_start
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:idx]
        idx += 1
    raise ValueError(f"namespace {namespace} block is not closed")


def extract_namespace_path(text: str, namespace_path: str) -> str:
    body = text
    for name in namespace_path.split("::"):
        body = extract_single_namespace_block(body, name)
    return body


def find_function_body(namespace_body: str, cpp_name: str) -> str:
    name_match = re.search(rf"\b{re.escape(cpp_name)}\s*\(", namespace_body)
    if name_match is None:
        raise ValueError(f"wrapper function not found: {cpp_name}")

    open_brace = namespace_body.find("{", name_match.end())
    if open_brace == -1:
        raise ValueError(f"wrapper function body not found: {cpp_name}")

    depth = 1
    idx = open_brace + 1
    while idx < len(namespace_body):
        char = namespace_body[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return namespace_body[open_brace + 1:idx]
        idx += 1
    raise ValueError(f"wrapper function body is not closed: {cpp_name}")


def normalized_call_target(impl_name: str) -> str:
    return impl_name if impl_name.startswith("::") else f"::{impl_name}"


def extract_wrapper_call_argument_names(function_body: str, impl_name: str) -> List[str]:
    target = normalized_call_target(impl_name)
    call_match = re.search(rf"{re.escape(target)}\s*\(", function_body)
    if call_match is None:
        raise ValueError(f"wrapper body does not call expected implementation {target}")

    open_paren = function_body.find("(", call_match.start())
    if open_paren == -1:
        raise ValueError(f"failed to locate call argument list for {target}")

    depth = 1
    idx = open_paren + 1
    while idx < len(function_body) and depth > 0:
        char = function_body[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        idx += 1
    if depth != 0:
        raise ValueError(f"call argument list is not closed for {target}")

    argument_text = function_body[open_paren + 1:idx - 1].strip()
    if not argument_text:
        return []
    return [normalize_cpp_whitespace(argument) for argument in split_top_level(argument_text)]


def check_generated_file_format(header_text: str, source_text: str) -> List[str]:
    failures = []
    if "\n\n\n" in header_text:
        failures.append("generated header contains redundant consecutive blank lines")
    if "\n\n\n" in source_text:
        failures.append("generated source contains redundant consecutive blank lines")

    expected_namespace_gap = "}  // namespace atb\n\nnamespace at_npu {"
    if expected_namespace_gap not in source_text:
        failures.append("generated source does not use the expected single blank line between internal and public namespaces")

    return failures


def check_source_contract(
    functions: Sequence[AtbCppApiFunction],
    facade_header_path: Path,
    generated_header_path: Path,
    source_path: Path,
    namespace: str,
) -> List[str]:
    failures = []
    require_file(facade_header_path, "facade header file")
    require_file(generated_header_path, "generated header file")
    require_file(source_path, "source file")
    print_step(f"checking source contract in {facade_header_path}, {generated_header_path}, and {source_path}")

    facade_header_text = facade_header_path.read_text(encoding="utf-8")
    header_text = generated_header_path.read_text(encoding="utf-8")
    source_text = source_path.read_text(encoding="utf-8")

    if '#include "op_plugin/include/AtbOpsInterface.h"' not in facade_header_text:
        failures.append(f"facade header does not include generated interface header: {facade_header_path}")

    failures.extend(check_config_consistency(functions, DEFAULT_ATB_SRC_DIR, source_path.name))
    failures.extend(check_generated_file_format(header_text, source_text))

    try:
        extract_namespace_path(header_text, namespace)
    except ValueError as exc:
        failures.append(f"generated header namespace mismatch: {exc}")

    try:
        namespace_body = extract_namespace_path(source_text, namespace)
    except ValueError as exc:
        return failures + [f"source namespace mismatch: {exc}"]

    checked_wrappers = []
    for fn in functions:
        try:
            header_signature = extract_header_signature(header_text, fn.cpp_name)
            header_argument_names = extract_signature_argument_names(strip_signature_defaults(header_signature))
        except ValueError as exc:
            failures.append(str(exc))
            continue

        try:
            impl_signature = extract_impl_signature(DEFAULT_ATB_SRC_DIR, source_path.name, fn.impl_name)
        except ValueError as exc:
            failures.append(str(exc))
            continue

        expected_signature = canonicalize_cpp_signature(
            impl_signature.replace(fn.impl_name.removeprefix("atb::"), fn.cpp_name, 1)
        )
        actual_signature = canonicalize_cpp_signature(strip_signature_defaults(header_signature))
        if actual_signature != expected_signature:
            failures.append(
                f"generated header signature mismatch for {namespace}::{fn.cpp_name}: "
                f"expected `{expected_signature}`, got `{header_signature}`"
            )
            continue

        try:
            function_body = find_function_body(namespace_body, fn.cpp_name)
        except ValueError as exc:
            failures.append(str(exc))
            continue

        expected_call = f"{normalized_call_target(fn.impl_name)}("
        if expected_call not in function_body:
            failures.append(f"wrapper {namespace}::{fn.cpp_name} does not call expected implementation {normalized_call_target(fn.impl_name)}")
            continue
        try:
            wrapper_call_argument_names = extract_wrapper_call_argument_names(function_body, fn.impl_name)
        except ValueError as exc:
            failures.append(str(exc))
            continue

        if wrapper_call_argument_names != header_argument_names:
            failures.append(
                f"wrapper {namespace}::{fn.cpp_name} does not forward arguments in declaration order: "
                f"expected {header_argument_names}, got {wrapper_call_argument_names}"
            )
            continue

        checked_wrappers.append(
            f"{namespace}::{fn.cpp_name} -> {normalized_call_target(fn.impl_name)} "
            f"args={wrapper_call_argument_names} [{header_signature}]"
        )

    print_info(f"checked {len(checked_wrappers)}/{len(functions)} wrapper declarations and forwarding pairs")
    print_sample("wrapper forwarding pairs", checked_wrappers)
    return failures


def check_symbols(functions: Sequence[AtbCppApiFunction], so_path: Path, namespace: str, nm_tool: str, cxxfilt_tool: str) -> List[str]:
    require_file(so_path, "shared library")
    nm_path = resolve_tool(nm_tool)
    cxxfilt_path = resolve_tool(cxxfilt_tool)
    print_step(f"checking exported symbols from {so_path}")
    print_info(f"using nm tool: {nm_path}")
    print_info(f"using c++filt tool: {cxxfilt_path}")
    nm_output = run_command([nm_path, "-D", "--defined-only", str(so_path)]).stdout
    demangled = run_command([cxxfilt_path, "-n"], input_text=nm_output).stdout
    demangled_lines = [line for line in demangled.splitlines() if line.strip()]

    failures = []
    found_symbols = []
    for fn in functions:
        expected = f"{namespace}::{fn.cpp_name}("
        matched_line = next((line for line in demangled_lines if expected in line), None)
        if matched_line is None:
            failures.append(f"missing exported symbol: {expected}")
            continue
        found_symbols.append(matched_line)
    print_info(f"matched {len(found_symbols)}/{len(functions)} expected ATB public symbols")
    print_sample("matched ATB public symbol lines", found_symbols)
    return failures


def check_absent_symbols(functions: Sequence[AtbCppApiFunction], so_path: Path, namespace: str, nm_tool: str, cxxfilt_tool: str) -> List[str]:
    require_file(so_path, "shared library")
    nm_path = resolve_tool(nm_tool)
    cxxfilt_path = resolve_tool(cxxfilt_tool)
    print_step(f"checking that {so_path} does not export ATB public symbols")
    print_info(f"using nm tool: {nm_path}")
    print_info(f"using c++filt tool: {cxxfilt_path}")
    nm_output = run_command([nm_path, "-D", "--defined-only", str(so_path)]).stdout
    demangled = run_command([cxxfilt_path, "-n"], input_text=nm_output).stdout
    demangled_lines = [line for line in demangled.splitlines() if line.strip()]

    matched_lines = []
    for fn in functions:
        expected = f"{namespace}::{fn.cpp_name}("
        matched_line = next((line for line in demangled_lines if expected in line), None)
        if matched_line is not None:
            matched_lines.append(matched_line)

    print_info(f"matched 0/{len(functions)} expected symbols is required for absence check; actual matched count: {len(matched_lines)}")
    print_sample("unexpected ATB public symbol lines", matched_lines)
    if matched_lines:
        return [f"unexpected exported symbol: {line}" for line in matched_lines]
    return []


def run_selected_checks(args: argparse.Namespace, paths: ResolvedPaths, functions: Sequence[AtbCppApiFunction]) -> int:
    failures: List[str] = []
    print_info(f"public namespace under test: {args.namespace}")
    if paths.header is not None:
        print_info(f"resolved facade header path: {paths.header}")
        generated_header_path = DEFAULT_GENERATED_HEADER if paths.header == DEFAULT_HEADER else paths.header.parent / "AtbOpsInterface.h"
        if not generated_header_path.is_file() and DEFAULT_BUILD_GENERATED_HEADER.is_file():
            generated_header_path = DEFAULT_BUILD_GENERATED_HEADER
        print_info(f"resolved generated header path: {generated_header_path}")
    if paths.source is not None:
        print_info(f"resolved source path: {paths.source}")
    if paths.so is not None:
        print_info(f"resolved ATB shared library: {paths.so}")

    if args.check_source:
        if paths.header is None:
            return print_error("header file could not be resolved")
        if paths.source is None:
            return print_error("source file could not be resolved")
        generated_header_path = DEFAULT_GENERATED_HEADER if paths.header == DEFAULT_HEADER else paths.header.parent / "AtbOpsInterface.h"
        if not generated_header_path.is_file() and DEFAULT_BUILD_GENERATED_HEADER.is_file():
            generated_header_path = DEFAULT_BUILD_GENERATED_HEADER
        failures.extend(check_source_contract(functions, paths.header, generated_header_path, paths.source, args.namespace))

    if args.check_symbols:
        if paths.so is None:
            return print_error("shared library could not be resolved; pass --so or --artifact-root")
        failures.extend(check_symbols(functions, paths.so, args.namespace, args.nm, args.cxxfilt))

    if args.check_absent_symbols:
        if paths.so is None:
            return print_error("shared library could not be resolved; pass --so or --artifact-root")
        failures.extend(check_absent_symbols(functions, paths.so, args.namespace, args.nm, args.cxxfilt))

    if args.check_example_build:
        if paths.so is None:
            return print_error("shared library could not be resolved; pass --so or --artifact-root")
        failures.extend(check_example_build(args, paths.so))

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"  - {failure}")
        return EXIT_CONTRACT_FAILED

    checks = []
    if args.check_source:
        checks.append("source")
    if args.check_symbols:
        checks.append("symbols")
    if args.check_absent_symbols:
        checks.append("absent-symbols")
    if args.check_example_build:
        checks.append("example-build")
    print(f"PASS: {', '.join(checks)} checks succeeded for {len(functions)}/{len(functions)} ATB public APIs")
    return EXIT_OK


def main() -> int:
    args = parse_args()
    if not any((args.check_source, args.check_symbols, args.check_absent_symbols, args.check_example_build)):
        return print_error("no checks requested; use --check-source, --check-symbols, --check-absent-symbols, or --check-example-build")

    try:
        paths = resolve_paths(args)
        functions = load_api_functions(paths.config)
        return run_selected_checks(args, paths, functions)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return print_error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
