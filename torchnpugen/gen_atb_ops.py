import argparse
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple
"""
Generate the ATB public C++ interface wrappers in AtbOpsInterface.h/.cpp.

This generator consumes op_plugin/config/atb_ops.yaml to produce:
  - op_plugin/include/AtbOpsInterface.h
  - op_plugin/ops/atb/AtbOpsInterface.cpp

Field meanings in atb_ops.yaml:
  - cpp_name:
      The fully qualified public C++ function name generated under namespace
      at_npu::native::atb, for example:
      at_npu::native::atb::_npu_reshape_and_cache
  - impl_name:
      The existing ::atb implementation function that the generated wrapper
      forwards to.

This generator only exposes ATB cpp functions that satisfy both constraints:
1. the implementation is registered through dispatcher schema in ops/atb/*.cpp
   via m.def(...) and m.impl(..., TORCH_FN(atb::<impl>));
2. the implementation is listed in op_plugin/config/atb_ops.yaml through the
   cpp_name -> impl_name mapping.

Maintenance rules:
  - atb_ops.yaml is an interface-to-implementation mapping file, not a schema
    source of truth file.
  - ATB dispatcher schemas are discovered from op_plugin/ops/atb/*.cpp
    TORCH_LIBRARY_FRAGMENT(atb, m) / m.def(...) definitions.
  - Public C++ wrapper signatures are derived from the corresponding
    ::atb implementation function definitions.
"""


# args / constants
PUBLIC_NAMESPACE = "at_npu::native::atb"


@dataclass(frozen=True)
class AtbOpEntry:
    cpp_name: str
    cpp_short_name: str
    impl_name: str


@dataclass(frozen=True)
class ResolvedAtbOp:
    cpp_name: str
    cpp_short_name: str
    impl_name: str
    impl_signature: str


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


# text normalization helpers
def normalize_schema(schema: str) -> str:
    schema = re.sub(r"\s+", " ", schema).strip()
    schema = schema.replace(",*", ", *")
    schema = re.sub(r"\*,\s*", "*, ", schema)
    return schema


def normalize_cpp_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_cpp_parameter(text: str) -> str:
    text = normalize_cpp_whitespace(text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    text = re.sub(r"\s*&\s*(?=[,>)])", "&", text)
    text = re.sub(r"&\s+([A-Za-z_])", r"&\1", text)
    return text


def find_matching_paren(text: str, open_paren: int) -> int:
    depth = 1
    idx = open_paren + 1
    while idx < len(text) and depth > 0:
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        idx += 1
    return idx - 1 if depth == 0 else -1


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


def split_signature(signature: str, *, strip_defaults: bool = False) -> Tuple[str, str, List[str]]:
    open_paren = signature.find("(")
    close_paren = signature.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren < open_paren:
        raise RuntimeError(f"malformed implementation signature: {signature}")
    prefix = normalize_cpp_whitespace(signature[:open_paren])
    match = re.match(r"(?P<return_type>.+?)\s+(?P<function_name>[A-Za-z_][A-Za-z0-9_]*)$", prefix)
    if match is None:
        raise RuntimeError(f"malformed implementation signature: {signature}")
    parameter_text = signature[open_paren + 1:close_paren].strip()
    parameters = split_top_level(parameter_text) if parameter_text else []
    if strip_defaults:
        parameters = [re.sub(r"\s*=\s*.*$", "", item).strip() for item in parameters]
    return match.group("return_type"), match.group("function_name"), parameters


def canonicalize_cpp_signature(signature: str, *, strip_defaults: bool = False) -> str:
    return_type, function_name, parameters = split_signature(signature, strip_defaults=strip_defaults)
    prefix = normalize_cpp_whitespace(f"{return_type} {function_name}")
    prefix = re.sub(r"([&>*])([A-Za-z_][A-Za-z0-9_]*)$", r"\1 \2", prefix)
    if not parameters:
        return f"{prefix}()"
    normalized_parameters = [normalize_cpp_parameter(item) for item in parameters]
    return f"{prefix}({', '.join(normalized_parameters)})"


def parse_parameter_declaration(parameter_text: str) -> Tuple[str, str, str]:
    parts = split_top_level(parameter_text, delimiter="=")
    declaration_text = parts[0].strip()
    default_text = "=".join(parts[1:]).strip() if len(parts) > 1 else ""
    if not declaration_text:
        raise RuntimeError(f"malformed parameter declaration: {parameter_text}")

    identifier_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for match in reversed(list(identifier_pattern.finditer(declaration_text))):
        prefix = declaration_text[:match.start()]
        suffix = declaration_text[match.end():].strip()
        if suffix and re.fullmatch(r"(?:\[[^\]]*\]\s*)+", suffix) is None:
            continue
        if not prefix or (not prefix[-1].isspace() and prefix[-1] not in "&*"):
            continue
        type_text = prefix.strip()
        if not type_text:
            continue
        return normalize_cpp_whitespace(type_text), match.group(0), default_text

    raise RuntimeError(f"parameter declaration has no name: {parameter_text}")


def extract_signature_argument_names(signature: str, *, impl_name: str = "") -> List[str]:
    argument_names: List[str] = []
    for parameter in split_signature(signature)[2]:
        try:
            _, name, _ = parse_parameter_declaration(parameter)
        except RuntimeError as exc:
            detail = (
                f"implementation parameter must be named for public wrapper generation: "
                f"impl_name={impl_name}, signature='{signature}', parameter='{parameter}'"
                if impl_name
                else f"malformed implementation signature: {signature}"
            )
            raise RuntimeError(detail) from exc
        argument_names.append(name)
    return argument_names


# config loading
def parse_cpp_name(cpp_name: str) -> Tuple[str, str]:
    prefix = f"{PUBLIC_NAMESPACE}::"
    if not cpp_name.startswith(prefix):
        raise RuntimeError(f"config error: cpp_name must start with {prefix} {cpp_name}")
    cpp_short_name = cpp_name[len(prefix):].strip()
    if not cpp_short_name or "::" in cpp_short_name:
        raise RuntimeError(f"config error: cpp_name must end with a function name under {PUBLIC_NAMESPACE}: {cpp_name}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cpp_short_name):
        raise RuntimeError(f"config error: cpp_name has invalid function name: {cpp_name}")
    return cpp_name, cpp_short_name


def load_config(config_path: Path) -> List[AtbOpEntry]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    functions = data.get("functions") if isinstance(data, dict) else None
    if not isinstance(functions, list):
        raise RuntimeError(f"config error: `functions` must exist and be a list in {config_path}")
    if not functions:
        raise RuntimeError(f"config error: no ATB public API entries found in {config_path}")

    entries: List[AtbOpEntry] = []
    for index, item in enumerate(functions, 1):
        if not isinstance(item, dict):
            raise RuntimeError(f"config error: entry #{index} must be a mapping in {config_path}")
        cpp_name = item.get("cpp_name")
        impl_name = item.get("impl_name")
        if not cpp_name or not impl_name:
            raise RuntimeError(f"config error: entry #{index} must contain both cpp_name and impl_name in {config_path}")
        cpp_name, cpp_short_name = parse_cpp_name(cpp_name)
        entries.append(AtbOpEntry(cpp_name=cpp_name, cpp_short_name=cpp_short_name, impl_name=impl_name))
    return entries


# dispatcher registry extraction
def load_atb_cpp_texts(src_dir: Path, generated_source_name: str) -> Dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8")
        for path in sorted(src_dir.glob("*.cpp"))
        if path.name != generated_source_name
    }


def extract_atb_registry(cpp_texts: Dict[Path, str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    schema_by_op_name: Dict[str, str] = {}
    op_names_by_impl_name: Dict[str, List[str]] = {}
    def_pattern = re.compile(r'm\.def\((?P<body>(?:\s*"[^"]*"\s*)+)\);', re.S)
    impl_pattern = re.compile(r'm\.impl\("([^"]+)",\s*TORCH_FN\(atb::([A-Za-z0-9_]+)\)\);')
    string_pattern = re.compile(r'"([^"]*)"')

    for text in cpp_texts.values():
        for match in def_pattern.finditer(text):
            schema = normalize_schema("".join(string_pattern.findall(match.group("body"))))
            op_name = schema.split("(", 1)[0]
            if op_name in schema_by_op_name and schema_by_op_name[op_name] != schema:
                raise RuntimeError(f"duplicate schema definition for {op_name}")
            schema_by_op_name[op_name] = schema
        for op_name, impl_name in impl_pattern.findall(text):
            op_names_by_impl_name.setdefault(impl_name, []).append(op_name)
    return schema_by_op_name, op_names_by_impl_name


def collect_named_signatures(text: str, impl_name_pattern: re.Pattern[str]) -> Dict[str, List[str]]:
    signatures: Dict[str, List[str]] = {}
    for match in impl_name_pattern.finditer(text):
        impl_short_name = match.group("impl_name")
        open_paren = text.find("(", match.start())
        if open_paren == -1:
            continue
        end_paren = find_matching_paren(text, open_paren)
        if end_paren == -1:
            continue
        tail_idx = end_paren + 1
        while tail_idx < len(text) and text[tail_idx].isspace():
            tail_idx += 1
        if tail_idx >= len(text) or text[tail_idx] != "{":
            continue
        signatures.setdefault(impl_short_name, []).append(normalize_cpp_whitespace(text[match.start():end_paren + 1]))
    return signatures


def extract_impl_signature_index(cpp_texts: Dict[Path, str], impl_names: List[str]) -> Dict[str, str]:
    index: Dict[str, List[str]] = {}
    impl_short_names = sorted({impl_name.removeprefix("atb::") for impl_name in impl_names})
    name_union = "|".join(re.escape(name) for name in impl_short_names)
    impl_name_pattern = re.compile(rf"^\s*.*\b(?P<impl_name>{name_union})\s*\(", re.M)

    for text in cpp_texts.values():
        for impl_short_name, matches in collect_named_signatures(text, impl_name_pattern).items():
            index.setdefault(f"atb::{impl_short_name}", []).extend(matches)

    signatures: Dict[str, str] = {}
    for impl_name in impl_names:
        matches = index.get(impl_name, [])
        unique = sorted(set(matches))
        if not unique:
            raise RuntimeError(f"implementation error: no implementation found for impl_name {impl_name}")
        if len(unique) != 1:
            raise RuntimeError(f"implementation error: multiple implementations found for impl_name {impl_name}: {unique}")
        signatures[impl_name] = canonicalize_cpp_signature(unique[0], strip_defaults=True)
    return signatures


def validate_config(
    entries: List[AtbOpEntry],
    schema_by_op_name: Dict[str, str],
    op_names_by_impl_name: Dict[str, List[str]],
    impl_signatures: Dict[str, str],
) -> List[ResolvedAtbOp]:
    resolved_entries: List[ResolvedAtbOp] = []
    seen_cpp_names = set()
    for entry in entries:
        if entry.cpp_name in seen_cpp_names:
            raise RuntimeError(f"config error: duplicate cpp_name in config: {entry.cpp_name}")
        seen_cpp_names.add(entry.cpp_name)
        impl_short_name = entry.impl_name.removeprefix("atb::")
        op_names = sorted(set(op_names_by_impl_name.get(impl_short_name, [])))
        if not op_names:
            raise RuntimeError(
                f"dispatcher error: no ATB dispatcher registration found for cpp_name {entry.cpp_name}, "
                f"impl_name {entry.impl_name}"
            )
        if len(op_names) != 1:
            raise RuntimeError(
                f"dispatcher error: impl_name {entry.impl_name} for cpp_name {entry.cpp_name} "
                f"maps to multiple dispatcher ops: {', '.join(op_names)}"
            )
        if schema_by_op_name.get(op_names[0]) is None:
            raise RuntimeError(
                f"dispatcher error: dispatcher op {op_names[0]} for cpp_name {entry.cpp_name} "
                f"has no matching m.def schema"
            )
        impl_signature = impl_signatures.get(entry.impl_name)
        if impl_signature is None:
            raise RuntimeError(
                f"implementation error: no implementation found for impl_name {entry.impl_name}"
            )
        extract_signature_argument_names(impl_signature, impl_name=entry.impl_name)
        resolved_entries.append(
            ResolvedAtbOp(
                cpp_name=entry.cpp_name,
                cpp_short_name=entry.cpp_short_name,
                impl_name=entry.impl_name,
                impl_signature=impl_signature,
            )
        )
    return resolved_entries


# rendering / output
def rename_signature(signature: str, old_name: str, new_name: str) -> str:
    renamed, count = re.subn(rf"\b{re.escape(old_name)}\b", new_name, signature, count=1)
    if count != 1:
        raise RuntimeError(f"failed to rename function {old_name} to {new_name} in signature: {signature}")
    return renamed


def render_entry(entry: ResolvedAtbOp) -> RenderedAtbOp:
    _, impl_short_name, _ = split_signature(entry.impl_signature)
    declaration_signature = rename_signature(entry.impl_signature, impl_short_name, entry.cpp_short_name)
    call_args = ", ".join(extract_signature_argument_names(declaration_signature, impl_name=entry.impl_name))
    return_type, _, _ = split_signature(entry.impl_signature)
    call_statement = f"::{entry.impl_name}({call_args});"
    if return_type != "void":
        call_statement = f"return {call_statement}"
    definition_body = f"    {call_statement}\n"
    definition = f"{declaration_signature}\n{{\n{definition_body}}}"
    return RenderedAtbOp(
        declaration=f"TORCH_NPU_API {declaration_signature};",
        forward_declaration=f"{entry.impl_signature};",
        definition=definition,
    )


def read_template(template_name: str) -> Template:
    template_dir = Path(__file__).resolve().parent / "templates"
    return Template((template_dir / template_name).read_text(encoding="utf-8"))


def render_outputs(rendered: List[RenderedAtbOp]) -> Tuple[str, str]:
    header_template = read_template("AtbOpsInterface.h")
    source_template = read_template("AtbOpsInterface.cpp")
    declarations = "\n".join(item.declaration for item in rendered).strip()
    forward_declarations = "\n".join(item.forward_declaration for item in rendered).strip()
    definitions = "\n\n".join(item.definition for item in rendered).strip()
    header_content = header_template.substitute(
        generated_comment="@generated by torchnpugen/gen_atb_ops.py",
        declarations=declarations,
    )
    source_content = source_template.substitute(
        generated_comment="@generated by torchnpugen/gen_atb_ops.py",
        forward_declarations=forward_declarations,
        definitions=definitions,
    )
    return header_content, source_content


def write_if_changed(path: Path, content: str) -> None:
    old_content = path.read_text(encoding="utf-8") if path.exists() else None
    if content != old_content:
        path.write_text(content, encoding="utf-8")


def extract_generated_declarations(header_content: str) -> Dict[str, str]:
    declarations: Dict[str, str] = {}
    for line in header_content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("TORCH_NPU_API "):
            continue
        signature = stripped.removeprefix("TORCH_NPU_API ").rstrip(";").strip()
        _, function_name, _ = split_signature(signature)
        declarations[function_name] = canonicalize_cpp_signature(signature, strip_defaults=True)
    return declarations


def extract_generated_forwarding_args(source_content: str, declaration_signature: str, impl_name: str) -> List[str]:
    impl_short_name = impl_name.removeprefix("atb::")
    signature_pos = source_content.find(f"{declaration_signature}\n{{")
    if signature_pos == -1:
        raise RuntimeError(
            f"forwarding mismatch for {declaration_signature} -> {impl_name}: wrapper definition not found in generated source"
        )
    body_start = signature_pos + len(declaration_signature) + 2
    body_end = source_content.find("\n}", body_start)
    if body_end == -1:
        raise RuntimeError(
            f"forwarding mismatch for {declaration_signature} -> {impl_name}: wrapper body not found in generated source"
        )
    body = source_content[body_start:body_end]
    call_pattern = re.compile(rf"::atb::{re.escape(impl_short_name)}\((?P<args>.*)\);", re.S)
    call_match = call_pattern.search(body)
    if call_match is None:
        raise RuntimeError(
            f"forwarding mismatch for {declaration_signature} -> {impl_name}: wrapper call not found in generated source"
        )
    args = call_match.group("args").strip()
    if not args:
        return []
    return [item.strip() for item in split_top_level(args)]


def verify_generated_interface_contract(
    resolved_entries: List[ResolvedAtbOp],
    header_content: str,
    source_content: str,
) -> None:
    declarations = extract_generated_declarations(header_content)

    for entry in resolved_entries:
        _, impl_short_name, _ = split_signature(entry.impl_signature)
        expected_signature = canonicalize_cpp_signature(
            rename_signature(entry.impl_signature, impl_short_name, entry.cpp_short_name),
            strip_defaults=True,
        )
        actual_signature = declarations.get(entry.cpp_short_name)
        if actual_signature is None:
            raise RuntimeError(
                f"declaration mismatch for {entry.cpp_name} -> {entry.impl_name}: declaration not found in generated header"
            )
        if actual_signature != expected_signature:
            raise RuntimeError(
                f"declaration mismatch for {entry.cpp_name} -> {entry.impl_name}: "
                f"expected `{expected_signature}`, got `{actual_signature}`"
            )

        expected_args = extract_signature_argument_names(actual_signature)
        actual_args = extract_generated_forwarding_args(source_content, actual_signature, entry.impl_name)
        if actual_args != expected_args:
            raise RuntimeError(
                f"forwarding mismatch for {entry.cpp_name} -> {entry.impl_name}: "
                f"expected args {expected_args}, got {actual_args}"
            )


# main
# Main flow:
# 1. load config
# 2. load ATB cpp texts
# 3. build dispatcher registry
# 4. build implementation signature index
# 5. validate mappings
# 6. render header/source content
# 7. verify generated header/source content
# 8. write outputs
def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    header_path = Path(args.header).resolve()
    source_path = Path(args.source).resolve()
    atb_src_dir = Path(args.atb_src_dir).resolve()

    entries = load_config(config_path)
    cpp_texts = load_atb_cpp_texts(atb_src_dir, source_path.name)
    schema_by_op_name, op_names_by_impl_name = extract_atb_registry(cpp_texts)
    impl_signatures = extract_impl_signature_index(cpp_texts, [entry.impl_name for entry in entries])
    resolved_entries = validate_config(entries, schema_by_op_name, op_names_by_impl_name, impl_signatures)
    rendered = [render_entry(entry) for entry in resolved_entries]
    header_content, source_content = render_outputs(rendered)
    verify_generated_interface_contract(resolved_entries, header_content, source_content)
    write_if_changed(header_path, header_content)
    write_if_changed(source_path, source_content)


if __name__ == "__main__":
    main()
