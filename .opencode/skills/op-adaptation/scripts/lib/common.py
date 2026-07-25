"""Common utilities: repo root detection, type mapping tables."""

from pathlib import Path

# ── YAML ↔ C++ type mapping ──────────────────────────────────────────
# Maps YAML schema types to their C++ equivalents (and vice versa).
# Used by parser.py and signature.py for cross-file verification.

YAML_TO_CPP = {
    "Tensor": ["const at::Tensor&", "at::Tensor", "const at::Tensor &", "at::Tensor&"],
    "Tensor?": ["const c10::optional<at::Tensor>&", "const c10::optional<at::Tensor> &", "c10::optional<at::Tensor>"],
    "int": ["int64_t", "int", "long", "long long"],
    "int?": ["c10::optional<int64_t>", "c10::optional<int>", "c10::OptionalIntArrayRef"],
    "int[]": ["c10::IntArrayRef", "at::IntArrayRef", "c10::SmallVector<int64_t"],
    "int[]?": ["c10::OptionalIntArrayRef"],
    "bool": ["bool"],
    "bool?": ["c10::optional<bool>"],
    "float": ["double", "float"],
    "float?": ["c10::optional<double>", "c10::optional<float>"],
    "str": ["c10::string_view", "std::string", "const c10::string_view", "const std::string"],
    "str?": ["c10::optional<c10::string_view>", "c10::optional<std::string>"],
    "ScalarType?": ["c10::optional<at::ScalarType>", "c10::optional<c10::ScalarType>"],
    "ScalarType": ["at::ScalarType", "c10::ScalarType"],
    "Scalar": ["at::Scalar", "c10::Scalar"],
    "Scalar?": ["c10::optional<at::Scalar>", "c10::optional<c10::Scalar>"],
    "SymInt": ["c10::SymInt"],
    "SymInt[]": ["c10::SymIntArrayRef"],
    "SymInt?": ["c10::optional<c10::SymInt>"],
    "SymInt[]?": ["c10::optional<c10::SymIntArrayRef>"],
    "Device": ["c10::Device"],
    "Device?": ["c10::optional<c10::Device>"],
    "Generator": ["at::Generator", "c10::Generator"],
    "Generator?": ["c10::optional<at::Generator>", "c10::optional<c10::Generator>"],
    "MemoryFormat": ["at::MemoryFormat", "c10::MemoryFormat"],
    "MemoryFormat?": ["c10::optional<at::MemoryFormat>"],
    "Layout": ["c10::Layout"],
    "Layout?": ["c10::optional<c10::Layout>"],
    "Dimname": ["at::Dimname"],
    "Dimname[]": ["at::DimnameList", "c10::ArrayRef<at::Dimname>"],
    "Tensor[]": ["at::TensorList", "c10::List<at::Tensor>", "c10::ArrayRef<at::Tensor>"],
    "Tensor[]?": ["c10::OptionalArrayRef<at::Tensor>", "c10::optional<at::TensorList>"],
}

# Reverse lookup: C++ type fragment → YAML type
CPP_TO_YAML = {}
for yaml_type, cpp_list in YAML_TO_CPP.items():
    for cpp in cpp_list:
        CPP_TO_YAML[cpp] = yaml_type


def find_repo_root() -> Path:
    """Locate the op-plugin repository root.

    Walk up from CWD to find a directory named ``op-plugin``; that directory is
    the repo root. If none is found, raise FileNotFoundError (the caller should
    ask the user to confirm the project root).
    """
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        if d.name == "op-plugin":
            return d

    raise FileNotFoundError(
        "Cannot locate op-plugin repo root: no directory named 'op-plugin' found "
        "by walking up from CWD. Run from within the repo or confirm the project root."
    )


def repo_paths(root: Path) -> dict:
    """Return well-known file paths within the repo."""
    return {
        "root": root,
        "yaml": root / "op_plugin" / "config" / "op_plugin_functions.yaml",
        "derivatives": root / "op_plugin" / "config" / "derivatives.yaml",
        "meta": root / "op_plugin" / "python" / "meta" / "_meta_registrations.py",
        "opapi_dir": root / "op_plugin" / "ops" / "opapi",
        "output_size_h": root / "op_plugin" / "utils" / "KernelNpuOutputSize.h",
        "output_size_cpp": root / "op_plugin" / "utils" / "KernelNpuOutputSize.cpp",
        "schema_json": root / "test" / "core_tests" / "torch_npu_OpApi_schema_all.json",
        "allowlist_json": root / "test" / "allowlist_for_publicAPI.json",
        "fake_tensor_test": root / "test" / "core_tests" / "test_fake_tensor.py",
    }


def extract_balanced(content: str, open_pos: int) -> str | None:
    """Return content between '(' at open_pos and its matching ')'.

    Respects nested parens and double-quoted string literals (with escapes).
    Does NOT handle char literals '( ' or // comments — sufficient for
    _def.cpp Input()/Attr() chains where strings use double quotes only.
    Do not reuse for general C++ text.
    """
    depth = 0
    in_str = False
    escape = False
    i = open_pos
    n = len(content)
    while i < n:
        ch = content[i]
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return content[open_pos + 1:i]
        i += 1
    return None
