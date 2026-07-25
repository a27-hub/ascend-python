"""gen-aclgraph: 为 ACLGraph 手工 update 生成 2 个新接口的草稿（rules: yaml-guide.md §Attention ACLGraph Interfaces / cpp-guide.md §Attention ACLGraph Variants）。

实现策略：原文行拷贝 + 定点插入 —— 从 functional 的 YAML func 行直接拷贝参数段
原文（保留 ``Tensor(a!)``、``SymInt[]?`` 等原始写法，不解析重组后再拼）。

产出（stdout，结构化分节，便于直接贴入 yaml/cpp）：
  1. ``_get_max_workspace`` YAML 条目草稿（functional 参数段拷贝，返回改 ``-> Tensor``）
  2. ``.out`` YAML 条目草稿（参数段拷贝 + 尾部 ``Tensor? workspace=None`` + alias 输出；
     版本字段留空，⚠️ codegen-type 决策标记）
  3. ``.out`` / ``_get_max_workspace`` C++ 函数骨架（模板见 cpp-guide §Attention ACLGraph Variants，TODO 清单固定 5 项）
"""

import re
from pathlib import Path

from .common import find_repo_root, repo_paths
from .signature import (
    find_yaml_func_raw,
    find_cpp_file,
    parse_yaml_func_line,
    _split_respecting_brackets,
    _split_func_raw,
    _return_arity,
    _alias_letters,
)


# YAML 类型 → C++ 形参声明（骨架用；未覆盖的类型打 TODO，由 LLM 确认）
_YAML_TO_CPP_DECL = {
    "Tensor": "const at::Tensor &",
    "Tensor?": "const c10::optional<at::Tensor> &",
    "Tensor[]": "at::TensorList",
    "Tensor[]?": "c10::OptionalArrayRef<at::Tensor>",
    "int": "int64_t",
    "int?": "c10::optional<int64_t>",
    "int[]": "at::IntArrayRef",
    "int[]?": "c10::OptionalIntArrayRef",
    "SymInt": "c10::SymInt",
    "SymInt?": "c10::optional<c10::SymInt>",
    "SymInt[]": "c10::SymIntArrayRef",
    "SymInt[]?": "c10::OptionalArrayRef<c10::SymInt>",
    "float": "double",
    "float?": "c10::optional<double>",
    "bool": "bool",
    "bool?": "c10::optional<bool>",
    "str": "c10::string_view",
    "str?": "c10::optional<c10::string_view>",
    "Scalar": "const at::Scalar &",
    "Scalar?": "const c10::optional<at::Scalar> &",
    "ScalarType": "at::ScalarType",
    "ScalarType?": "c10::optional<at::ScalarType>",
}


# ── functional C++ 定位与 aclnn 名机械提取 ──────────────────────────

def _find_functional_cpp(op_name: str, opapi_dir: Path, has_symint: bool):
    """Locate the functional C++ function. SymInt 形态的实现名带 ``_symint`` 后缀
    优先按带后缀名查找。Returns (cpp_path, cpp_func_name) or (None, None).
    """
    candidates = [op_name + "_symint", op_name] if has_symint else [op_name, op_name + "_symint"]
    for name in candidates:
        p = find_cpp_file(name, opapi_dir)
        if p is not None:
            return p, name
    return None, None


def _extract_function_body(content: str, func_name: str) -> str | None:
    """Extract the body (between '{' and matching '}') of func_name's definition."""
    pattern = re.compile(r'\b' + re.escape(func_name) + r'\s*\(')
    for m in pattern.finditer(content):
        # find matching ')' of the parameter list
        open_pos = m.end() - 1
        depth = 0
        close_pos = -1
        for i in range(open_pos, len(content)):
            if content[i] == "(":
                depth += 1
            elif content[i] == ")":
                depth -= 1
                if depth == 0:
                    close_pos = i
                    break
        if close_pos == -1:
            continue
        # definition: ')' followed by '{'
        rest = content[close_pos + 1:close_pos + 8]
        bm = re.match(r'\s*\{', rest)
        if not bm:
            continue  # call site / declaration
        brace_start = close_pos + 1 + bm.end() - 1
        depth = 0
        for i in range(brace_start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    return content[brace_start + 1:i]
        return None
    return None


def _extract_aclnn_names(body: str) -> list[str]:
    """机械提取 EXEC_NPU* 宏中的 aclnn 调用名（保持出现顺序去重）。"""
    names = []
    for n in re.findall(r'EXEC_NPU\w*\s*\(\s*(aclnn\w+)', body):
        if n not in names:
            names.append(n)
    return names


def _detect_version_dispatch(body: str) -> bool:
    """functional 是否带 CANN 版本 / SoC 门控（is_gte_cann_version* / GetSocVersion）。"""
    return bool(re.search(r'is_gte_cann_version\w*|GetSocVersion', body))


# ── YAML 草稿 ────────────────────────────────────────────────────────

def _existing_out_names(op_name: str, yaml_path: Path, arity: int) -> list[str] | None:
    """仓内已有 .out 条目时，沿用其 alias 输出参数名（重新生成场景保持一致）。"""
    out_raw = find_yaml_func_raw(op_name + ".out", yaml_path)
    if not out_raw:
        return None
    _, out_args, _ = _split_func_raw(out_raw)
    if out_args is None:
        return None
    names = [
        m.group(1)
        for p in _split_respecting_brackets(out_args)
        if (m := re.match(r'^\s*Tensor\([a-z]+!\)\s+(\w+)\s*$', p))
    ]
    return names if len(names) == arity else None


def _symint_params(func_args: str) -> list[str]:
    """functional 参数段中含 SymInt 的参数原文（⚠️ codegen-type 决策标记对象）。"""
    return [p.strip() for p in _split_respecting_brackets(func_args) if "SymInt" in p]


def _has_star_separator(func_args: str) -> bool:
    return "*" in [p.strip() for p in _split_respecting_brackets(func_args)]


# ── C++ 骨架 ─────────────────────────────────────────────────────────

def _cpp_param_decls(params: list[dict]) -> list[str]:
    decls = []
    for p in params:
        ptype = re.sub(r'\(\w+!\)', '', p["type"])  # strip alias annotations
        cpp_t = _YAML_TO_CPP_DECL.get(ptype)
        if cpp_t is None:
            decls.append(f"/* TODO: 确认 C++ 类型（YAML: {p['type']}）*/ {p['name']}")
        else:
            decls.append(f"{cpp_t} {p['name']}")
    return decls


def _format_params(decls: list[str], indent: str = "    ") -> str:
    return (",\n" ).join(indent + d for d in decls)


def _aclnn_placeholder(aclnn_names: list[str]) -> str:
    if not aclnn_names:
        return "aclnnXxx /* TODO①: 未能从 functional C++ 提取 aclnn 名，请手工填写 */"
    if len(aclnn_names) > 1:
        return (aclnn_names[0]
                + f" /* TODO①②: functional 存在多个 aclnn 名 {aclnn_names}（版本/SoC 分派），"
                  f".out 两路径与 get-max 必须逐项复制同一分派 */")
    return aclnn_names[0]


def _gen_out_cpp_skeleton(op_name: str, params: list[dict], out_names: list[str],
                          symint_suffix: str, aclnn_names: list[str]) -> str:
    """`.out` 双路径骨架。"""
    arity = len(out_names)
    decls = _cpp_param_decls(params)
    decls.append("const c10::optional<at::Tensor> &workspace")
    decls.extend(f"at::Tensor &{n}" for n in out_names)
    param_block = _format_params(decls)

    if arity > 1:
        ret_type = "std::tuple<" + ", ".join("at::Tensor &" for _ in out_names) + ">"
        ret_stmt = f"std::tie({', '.join(out_names)})"
    else:
        ret_type = "at::Tensor &"
        ret_stmt = out_names[0]

    aclnn = _aclnn_placeholder(aclnn_names)
    call_args = ",\n            ".join(p["name"] for p in params)
    outs = ", ".join(out_names)

    return f"""{ret_type} {op_name}_out{symint_suffix}(
{param_block})
{{
    // TODO④: host 序列参数转换——CPU Tensor 先校验（CPU/一维/整数 dtype）再转
    // IntArrayRef，注意生命周期（不得指向已销毁的临时对象）；SymInt 直转公共宏 CopyTypesV2。
    // TODO⑤: 预处理审查结论转 TORCH_CHECK（layout 转换、dropout mask、cast 等需静态上界）。

    if (workspace.has_value()) {{
        void *workspace_addr = const_cast<void *>(workspace->storage().data());
        // workspace 字节数按 numel()*element_size() 计算，不能把 numel() 直接当字节数
        uint64_t workspace_size = static_cast<uint64_t>(workspace->numel() * workspace->element_size());
        EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(
            {aclnn},
            workspace_addr,
            workspace_size,
            {call_args},
            {outs});
    }} else {{
        EXEC_NPU_NO_FORMAT_CHECK_CMD(
            {aclnn},
            {call_args},
            {outs});
    }}

    return {ret_stmt};
}}"""


def _gen_get_max_cpp_skeleton(op_name: str, params: list[dict], out_names: list[str],
                              symint_suffix: str, aclnn_names: list[str]) -> str:
    """`_get_max_workspace` 骨架。"""
    decls = _cpp_param_decls(params)
    param_block = _format_params(decls)
    aclnn = _aclnn_placeholder(aclnn_names)
    call_args = ",\n            ".join(p["name"] for p in params)
    outs = ", ".join(out_names)

    first_tensor = next((p["name"] for p in params if p["type"].startswith("Tensor")), None)
    options_expr = (f"{first_tensor}.options().dtype(at::kByte)" if first_tensor
                    else "/* TODO: 无 Tensor 参数，确认 device options */ at::TensorOptions().dtype(at::kByte)")

    return f"""at::Tensor _{op_name}_get_max_workspace{symint_suffix}(
{param_block})
{{
    // TODO③: 只在 C++ 内部构造用于描述 shape/dtype/format 的临时输出 ——
    // 与 functional 共用同一输出构造 helper（此处为占位，按真实 helper 替换）。
    auto [{outs}] = construct_{op_name}_output(/* TODO③: helper 实参，与 functional 一致 */);

    uint64_t workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(
        {aclnn},
        {call_args},
        {outs});

    // 新接口统一使用 Byte Tensor，使 numel 与字节数一致；
    // workspace_size == 0 时仍返回合法 Tensor，由公共执行路径处理空 workspace。
    return npu_preparation::apply_tensor_without_format(
        {{static_cast<int64_t>(workspace_size)}},
        {options_expr});
}}"""


# ── 主流程 ───────────────────────────────────────────────────────────

def run_gen_aclgraph(op_name: str, repo_root: Path | None = None,
                     out_names: list[str] | None = None) -> dict:
    """Generate ACLGraph manual-update drafts for op_name.

    Returns a dict with the individual artifacts (yaml_out / yaml_get_max_workspace
    为可直接与仓内真身逐字 diff 的 func 行) plus the formatted stdout text.
    """
    if repo_root is None:
        repo_root = find_repo_root()
    paths = repo_paths(repo_root)

    func_raw = find_yaml_func_raw(op_name, paths["yaml"])
    if func_raw is None:
        return {"error": f"Operator '{op_name}' not found in {paths['yaml']}"}
    _, func_args, func_ret = _split_func_raw(func_raw)
    if func_args is None:
        return {"error": f"Cannot split functional func line for '{op_name}': {func_raw}"}

    arity = _return_arity(func_ret)
    if arity < 1:
        return {"error": f"Cannot determine return arity for '{op_name}' from '{func_ret}'"}
    letters = _alias_letters(arity)

    parsed = parse_yaml_func_line(func_raw)
    params = parsed.get("params", [])

    # out 参数名：显式指定 > 沿用仓内已有 .out 条目 > out0/out1 占位
    if out_names is None:
        out_names = _existing_out_names(op_name, paths["yaml"], arity)
        out_names_reused = out_names is not None
    else:
        out_names_reused = False
    if out_names is None:
        out_names = [f"out{i}" for i in range(arity)]
    if len(out_names) != arity:
        return {"error": f"--out-names 数量 {len(out_names)} != functional 返回数量 {arity}"}

    # ── YAML 草稿 ──
    yaml_get_max = f"_{op_name}_get_max_workspace({func_args}) -> Tensor"

    outs_decl = ", ".join(f"Tensor({l}!) {n}" for l, n in zip(letters, out_names))
    if _has_star_separator(func_args):
        # functional 自带 *, —— workspace/out 直接接在其后（IFA 形态）
        tail = f"Tensor? workspace=None, {outs_decl}"
    else:
        # 全位置参数 functional —— 推荐 workspace=None, *, outs（alias 输出必须 keyword-only）
        tail = f"Tensor? workspace=None, *, {outs_decl}"
    if arity > 1:
        ret_alias = "(" + ", ".join(f"Tensor({l}!)" for l in letters) + ")"
    else:
        ret_alias = f"Tensor({letters[0]}!)"
    yaml_out = f"{op_name}.out({func_args}, {tail}) -> {ret_alias}"

    symint_parts = _symint_params(func_args)
    has_symint = bool(symint_parts)
    symint_suffix = "_symint" if has_symint else ""

    # ── functional C++ 与 aclnn 名提取 ──
    cpp_path, cpp_func_name = _find_functional_cpp(op_name, paths["opapi_dir"], has_symint)
    aclnn_names: list[str] = []
    has_version_dispatch = False
    if cpp_path is not None:
        body = _extract_function_body(cpp_path.read_text(encoding="utf-8"), cpp_func_name)
        if body:
            aclnn_names = _extract_aclnn_names(body)
            has_version_dispatch = _detect_version_dispatch(body)

    # ── C++ 骨架 ──
    cpp_out = _gen_out_cpp_skeleton(op_name, params, out_names, symint_suffix, aclnn_names)
    cpp_get_max = _gen_get_max_cpp_skeleton(op_name, params, out_names, symint_suffix, aclnn_names)

    result = {
        "op_name": op_name,
        "return_arity": arity,
        "yaml_out": yaml_out,
        "yaml_get_max_workspace": yaml_get_max,
        "out_names": out_names,
        "out_names_reused": out_names_reused,
        "symint_params": symint_parts,
        "cpp_file": str(cpp_path) if cpp_path else None,
        "cpp_functional_name": cpp_func_name,
        "aclnn_names": aclnn_names,
        "has_version_dispatch": has_version_dispatch,
        "cpp_out_skeleton": cpp_out,
        "cpp_get_max_skeleton": cpp_get_max,
    }
    result["text"] = _format_output(result)
    return result


def _format_output(r: dict) -> str:
    """格式化 stdout 输出：结构化分节，便于直接贴入 yaml/cpp。"""
    op_name = r["op_name"]
    lines = []

    lines.append("=" * 70)
    lines.append(f"gen-aclgraph 草稿: {op_name}")
    lines.append(f"functional C++: {r['cpp_file'] or '未找到（骨架 aclnn 名需手工填写）'}"
                 + (f"  ({r['cpp_functional_name']})" if r["cpp_functional_name"] else ""))
    lines.append("策略: 原文行拷贝 + 定点插入；⚠️ 项需 LLM/用户确认，勿直接合入")
    lines.append("=" * 70)
    lines.append("")

    version_note = "# TODO: 版本字段按目标算子真实支持范围填写，不得照抄其他算子"
    lines.append("─" * 8 + f" 1/3 YAML 草稿: _{op_name}_get_max_workspace " + "─" * 8)
    lines.append(version_note)
    lines.append("# 内部接口以下划线开头，不设置 exposed")
    lines.append(f"  - func: {r['yaml_get_max_workspace']}")
    lines.append("    op_api: []  # TODO: 按真实支持范围填写，不得照抄")
    lines.append("    internal_format_opapi: []  # TODO: 同上；不支持 internal format 则删除本行")
    lines.append("    # device_check: ...  # TODO: 按真实支持范围决定是否填写")
    lines.append("")

    lines.append("─" * 8 + f" 2/3 YAML 草稿: {op_name}.out " + "─" * 8)
    lines.append(version_note)
    if r["out_names_reused"]:
        lines.append(f"# out 参数名沿用仓内已有 .out 条目: {r['out_names']}")
    else:
        lines.append(f"# out 参数名为占位 {r['out_names']} —— 按输出语义重命名"
                     f"（参照 IFA 真身: attention_out / softmax_lse）")
    lines.append(f"  - func: {r['yaml_out']}")
    lines.append("    op_api: []  # TODO: 按真实支持范围填写，不得照抄")
    lines.append("    internal_format_opapi: []  # TODO: 同上；不支持 internal format 则删除本行")
    lines.append("    # device_check: ...  # TODO: 按真实支持范围决定是否填写")
    lines.append("")

    lines.append("# ⚠️ codegen-type 决策标记")
    if r["symint_params"]:
        for p in r["symint_params"]:
            lines.append(f"#   - {p}")
        lines.append("#   默认保留 SymInt 写法（IFA 形态）：若 functional 在 `symint:` 分组，")
        lines.append("#   .out 与 _get_max_workspace 必须同步登记，生成的实现名带 `_symint` 后缀。")
        lines.append("#   注：仓内另有 FA v3 将 SymInt[]? 降级为 int[]? 的孤例，但 FA 未真实接入")
        lines.append("#   ACLGraph、不作先例；如需降级必须用户明确确认并记入 CHECKLIST。")
    else:
        lines.append("#   functional 无 SymInt 参数，无 codegen-type 决策项。")
    lines.append("")

    lines.append("─" * 8 + " 3/3 C++ 骨架（结构草稿非可编译代码）" + "─" * 8)
    lines.append("// ── 骨架 TODO 清单（固定 5 项，逐项完成后再合入）────────────────")
    lines.append(f"// TODO① aclnn 名核对: 脚本从 functional 提取到 {r['aclnn_names'] or '（无）'}，需 LLM 对照 functional 确认")
    if r["has_version_dispatch"]:
        lines.append("// TODO② 版本/SoC 分派复制: functional 存在 is_gte_cann_version*/GetSocVersion 门控，")
        lines.append("//       .out 两条路径与 get-max 必须逐项复制同一分派")
    else:
        lines.append("// TODO② 版本/SoC 分派复制: functional 未检测到门控；若有遗漏，三处必须一致")
    lines.append("// TODO③ helper 函数体: 内部输出构造 helper 与 functional 共用")
    lines.append("// TODO④ host 序列参数转换: 见 cpp-guide.md §Attention ACLGraph Variants → host 序列参数处理细则（CPU Tensor 五步校验 + IntArrayRef 生命周期 + layout 语义约束）；SymInt 直转 CopyTypesV2")
    lines.append("// TODO⑤ 预处理限制转 TORCH_CHECK")
    lines.append("")
    lines.append(r["cpp_out_skeleton"])
    lines.append("")
    lines.append(r["cpp_get_max_skeleton"])
    lines.append("")

    return "\n".join(lines)
