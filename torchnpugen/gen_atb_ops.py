"""
Generate the ATB public C++ interface wrappers in AtbOpsInterface.h/.cpp.

This generator only exposes ATB cpp functions that satisfy both constraints:
1. the implementation is registered through dispatcher schema in ops/atb/*.cpp
   via m.def(...) and m.impl(..., TORCH_FN(atb::<impl>));
2. the implementation is listed in op_plugin/config/atb_ops.yaml through the
   cpp_name -> impl_name mapping.

Only functions that satisfy both constraints are emitted into the public
at_npu::native::atb C++ interface. The generated public signatures use the
existing ::atb implementation function definitions as the source of truth.
"""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class AtbOpEntry:
    cpp_name: str
    impl_name: str


@dataclass(frozen=True)
class RenderedAtbOp:
    declaration: str
    forward_declaration: str
    definition: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ATB C++ public interfaces")
    parser.add_argument(
        "--config",
        default="op_plugin/config/atb_ops.yaml",
        help="Path to op_plugin/config/atb_ops.yaml.",
    )
    parser.add_argument("--header", default="op_plugin/include/AtbOpsInterface.h", help="Output header path.")
    parser.add_argument("--source", default="op_plugin/ops/atb/AtbOpsInterface.cpp", help="Output source path.")
    parser.add_argument("--atb-src-dir", default="op_plugin/ops/atb", help="Directory containing ATB op cpp files.")
    return parser.parse_args()


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


def strip_signature_defaults(signature: str) -> str:
    open_paren = signature.find("(")
    close_paren = signature.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren < open_paren:
        return signature
    prefix = signature[:open_paren]
    parameter_text = signature[open_paren + 1:close_paren].strip()
    if not parameter_text:
        return signature
    parameters = [re.sub(r"\s*=\s*.*$", "", item).strip() for item in split_top_level(parameter_text)]
    return f"{prefix}({', '.join(parameters)})"


def extract_signature_argument_names(signature: str) -> List[str]:
    open_paren = signature.find("(")
    close_paren = signature.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren < open_paren:
        raise RuntimeError(f"failed to locate parameter list in signature: {signature}")
    parameter_text = signature[open_paren + 1:close_paren].strip()
    if not parameter_text:
        return []
    argument_names: List[str] = []
    for parameter in split_top_level(parameter_text):
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", parameter.strip())
        if name_match is None:
            raise RuntimeError(f"failed to extract argument name from parameter: {parameter}")
        argument_names.append(name_match.group(1))
    return argument_names


def load_config(config_path: Path) -> List[AtbOpEntry]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [AtbOpEntry(cpp_name=item["cpp_name"], impl_name=item["impl_name"]) for item in data.get("functions", [])]


def extract_atb_registry(
    src_dir: Path,
    generated_source_name: str,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    schema_by_op_name: Dict[str, str] = {}
    op_names_by_impl_name: Dict[str, List[str]] = {}
    def_pattern = re.compile(r'm\.def\((?P<body>(?:\s*"[^"]*"\s*)+)\);', re.S)
    impl_pattern = re.compile(r'm\.impl\("([^"]+)",\s*TORCH_FN\(atb::([A-Za-z0-9_]+)\)\);')
    string_pattern = re.compile(r'"([^"]*)"')

    for path in sorted(src_dir.glob("*.cpp")):
        if path.name == generated_source_name:
            continue
        text = path.read_text(encoding="utf-8")
        for match in def_pattern.finditer(text):
            schema = normalize_schema("".join(string_pattern.findall(match.group("body"))))
            op_name = schema.split("(", 1)[0]
            if op_name in schema_by_op_name and schema_by_op_name[op_name] != schema:
                raise RuntimeError(f"duplicate schema definition for {op_name}")
            schema_by_op_name[op_name] = schema
        for op_name, impl_name in impl_pattern.findall(text):
            op_names_by_impl_name.setdefault(impl_name, []).append(op_name)
    return schema_by_op_name, op_names_by_impl_name


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
        raise RuntimeError(f"no C++ function definition found for atb::{impl_short_name}")
    if len(normalized) != 1:
        raise RuntimeError(f"multiple C++ function definitions found for atb::{impl_short_name}: {normalized}")
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
        except RuntimeError as exc:
            if f"atb::{impl_short_name}" not in str(exc):
                raise
            continue
        matches.append(signature_text)

    unique = sorted(set(matches))
    if not unique:
        raise RuntimeError(f"no C++ implementation signature found for {impl_name}")
    if len(unique) != 1:
        raise RuntimeError(f"multiple distinct C++ implementation signatures found for {impl_name}: {unique}")
    return canonicalize_cpp_signature(strip_signature_defaults(unique[0]))


def validate_registered_schema(
    entry: AtbOpEntry,
    schema_by_op_name: Dict[str, str],
    op_names_by_impl_name: Dict[str, List[str]],
) -> None:
    impl_short_name = entry.impl_name.removeprefix("atb::")
    op_names = sorted(set(op_names_by_impl_name.get(impl_short_name, [])))
    if not op_names:
        raise RuntimeError(f"no ATB dispatcher registration found for impl_name {entry.impl_name}")
    if len(op_names) != 1:
        raise RuntimeError(f"impl_name {entry.impl_name} maps to multiple dispatcher ops: {', '.join(op_names)}")
    if schema_by_op_name.get(op_names[0]) is None:
        raise RuntimeError(f"dispatcher op {op_names[0]} has no matching m.def schema")


def rename_signature(signature: str, old_name: str, new_name: str) -> str:
    renamed, count = re.subn(rf"\b{re.escape(old_name)}\b", new_name, signature, count=1)
    if count != 1:
        raise RuntimeError(f"failed to rename function {old_name} to {new_name} in signature: {signature}")
    return renamed


def render_entry(entry: AtbOpEntry, implementation_signature: str) -> RenderedAtbOp:
    impl_short_name = entry.impl_name.removeprefix("atb::")
    declaration_signature = rename_signature(implementation_signature, impl_short_name, entry.cpp_name)
    argument_names = extract_signature_argument_names(declaration_signature)
    call_args = ", ".join(argument_names)
    return_prefix = implementation_signature.split(impl_short_name, 1)[0].strip()
    if return_prefix == "void":
        definition = f"{declaration_signature}\n{{\n    ::{entry.impl_name}({call_args});\n}}"
    else:
        definition = f"{declaration_signature}\n{{\n    return ::{entry.impl_name}({call_args});\n}}"
    return RenderedAtbOp(
        declaration=f"TORCH_NPU_API {declaration_signature};",
        forward_declaration=f"{implementation_signature};",
        definition=definition,
    )


def validate_config(
    entries: List[AtbOpEntry],
    schema_by_op_name: Dict[str, str],
    op_names_by_impl_name: Dict[str, List[str]],
    src_dir: Path,
    generated_source_name: str,
) -> Dict[str, str]:
    implementation_signatures_by_cpp_name: Dict[str, str] = {}
    seen_cpp_names = set()
    for entry in entries:
        if entry.cpp_name in seen_cpp_names:
            raise RuntimeError(f"duplicate cpp_name in config: {entry.cpp_name}")
        seen_cpp_names.add(entry.cpp_name)
        validate_registered_schema(entry, schema_by_op_name, op_names_by_impl_name)
        implementation_signatures_by_cpp_name[entry.cpp_name] = extract_impl_signature(
            src_dir,
            generated_source_name,
            entry.impl_name,
        )
    return implementation_signatures_by_cpp_name


def read_template(template_name: str) -> Template:
    template_dir = Path(__file__).resolve().parent / "templates"
    return Template((template_dir / template_name).read_text(encoding="utf-8"))


def write_if_changed(path: Path, content: str) -> None:
    old_content = path.read_text(encoding="utf-8") if path.exists() else None
    if content != old_content:
        path.write_text(content, encoding="utf-8")


def write_outputs(header_path: Path, source_path: Path, rendered: List[RenderedAtbOp]) -> None:
    declarations = "\n".join(item.declaration for item in rendered).strip()
    forward_declarations = "\n".join(item.forward_declaration for item in rendered).strip()
    definitions = "\n\n".join(item.definition for item in rendered).strip()
    header_content = read_template("AtbOpsInterface.h").substitute(
        generated_comment="@generated by torchnpugen/gen_atb_ops.py",
        declarations=declarations,
    )
    source_content = read_template("AtbOpsInterface.cpp").substitute(
        generated_comment="@generated by torchnpugen/gen_atb_ops.py",
        forward_declarations=forward_declarations,
        definitions=definitions,
    )
    write_if_changed(header_path, header_content)
    write_if_changed(source_path, source_content)


def main() -> None:
    if not os.environ.get("PYTORCH_VERSION"):
        raise RuntimeError("PYTORCH_VERSION must be set before running gen_atb_ops.py")

    args = parse_args()
    config_path = Path(args.config).resolve()
    header_path = Path(args.header).resolve()
    source_path = Path(args.source).resolve()
    atb_src_dir = Path(args.atb_src_dir).resolve()

    entries = load_config(config_path)
    schema_by_op_name, op_names_by_impl_name = extract_atb_registry(atb_src_dir, source_path.name)
    implementation_signatures_by_cpp_name = validate_config(
        entries,
        schema_by_op_name,
        op_names_by_impl_name,
        atb_src_dir,
        source_path.name,
    )
    rendered = [render_entry(entry, implementation_signatures_by_cpp_name[entry.cpp_name]) for entry in entries]
    write_outputs(header_path, source_path, rendered)


if __name__ == "__main__":
    main()
