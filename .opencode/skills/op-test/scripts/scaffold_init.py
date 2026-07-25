#!/usr/bin/env python3
"""
CHECKLIST.md 初始化脚手架 — 替代 T-scaffold-pre + T-scaffold-post。
从 op-adaptation CHECKLIST.md + _def.cpp + .md 文档一次性提取所有确定性信息，
写入 CHECKLIST.md「## UT Generation Info」完整骨架。

用法:
  python scaffold_init.py [--checklist CHECKLIST路径] [--dry-run] [--force]
"""

import re, sys, json, argparse
from pathlib import Path
from constraint_info import ConstraintInfo
from dataclasses import asdict


GE_DTYPE_MAP = {
    "DT_FLOAT16": "float16", "DT_BF16": "bfloat16", "DT_FLOAT": "float32",
    "DT_INT8": "int8", "DT_INT4": "quint4x2", "DT_INT32": "int32",
    "DT_INT64": "int64", "DT_UINT8": "uint8", "DT_BOOL": "bool",
    "DT_DOUBLE": "float64",
    "DT_HIFLOAT8": "hifloat8", "DT_FLOAT8_E4M3FN": "float8_e4m3fn",
    "DT_FLOAT8_E5M2": "float8_e5m2",
}

CLASSIFY_KW = {
    "quant": "quantization", "dynamic_quant": "quantization",
    "group_quant": "quantization", "dequant": "quantization",
    "attention": "attention", "attn": "attention",
    "ffn": "matmul", "matmul": "matmul",
    "norm": "normalization", "rms_norm": "normalization",
    "layer_norm": "normalization", "softmax": "activation",
    "silu": "activation", "gelu": "activation", "relu": "activation",
    "moe": "matmul", "routing": "matmul",
}


# ─── step 1: 从 op-adaptation CHECKLIST.md 提取 ──────────────────

def extract_op_name(text):
    # 容忍 markdown 包裹: 名字周围的反引号/引号/星号(粗体), 冒号可为中/英文,
    # 字段名后可带 ** 粗体标记。用 [A-Za-z0-9_] 限定捕获, 防尾部反引号被 \S+ 吞入。
    # 覆盖: "torch_npu function name: `npu_x`" / "**torch_npu function name**：npu_x" / 裸 "npu_x"
    m = re.search(r'torch_npu\s+function\s+name\**\s*[:：]\s*["\'`*]*\s*npu_([A-Za-z0-9_]+)', text)
    if m:
        return m.group(1).strip().strip('`"\'*')
    m = re.search(r'aclnn\s+function\s+name\**\s*[:：]\s*["\'`*]*\s*aclnn([A-Za-z0-9]+)', text)
    if m:
        camel = m.group(1).strip().strip('`"\'*')
        return re.sub(r'(?<!^)(?=[A-Z])', '_', camel).lower()
    return None

def extract_mode(text):
    m = re.search(r'Mode:\s*(\w+)', text)
    return m.group(1).strip().lower() if m else None

def determine_op_type(fn):
    if fn and fn.startswith("npu_"): return "custom"
    if fn and fn.startswith("torch."): return "native"
    return "custom"

def determine_target_dir(op_type):
    return "test/test_custom_ops" if op_type == "custom" else "test/test_base_ops"

def classify(name):
    if not name: return "other"
    nl = name.lower()
    for kw, cls in CLASSIFY_KW.items():
        if kw in nl: return cls
    return "other"

def determine_mode(search_dir, fn, target_dir, from_cl):
    if from_cl: return from_cl
    if fn and target_dir:
        return "modify" if (Path(search_dir) / target_dir / f"test_{fn}.py").exists() else "create"
    return "create"


# ─── step 2: 文件发现 ─────────────────────────────────────────────

def find_def_cpp(name, search_dir):
    sd = Path(search_dir)
    for pat in [f"aclnn_input/{name}_def.cpp", f"aclnn_input/*{name}*_def.cpp", f"**/{name}_def.cpp"]:
        m = list(sd.glob(pat))
        if m: return m[0]
    return None

def find_md_doc(name, search_dir):
    sd = Path(search_dir)
    camel = "".join(w.capitalize() for w in name.split("_"))
    for pat in [f"aclnn_input/aclnn{camel}.md", f"aclnn_input/{name}.md",
                f"**/{name}.md", f"**/aclnn{camel}.md"]:
        m = list(sd.glob(pat))
        if m: return m[0]
    return None


# ─── step 3: _def.cpp 解析 ────────────────────────────────────────

def parse_def_cpp(filepath):
    """解析 _def.cpp，返回主块参数 + 平台扩展dtype。
    产出结构:
      { "operator_class": str|null, "inputs": [...], "outputs": [...], "attrs": [...],
        "configs": [...], "platform_dtypes": {plat: {param: [dtype]}} }
    """
    text = Path(filepath).read_text(encoding="utf-8")
    result = {"operator_class": None, "inputs": [], "outputs": [], "attrs": [], "configs": [],
              "platform_dtypes": {}}
    m = re.search(r"class\s+(\w+)\s*:\s*public\s+OpDef", text)
    if m: result["operator_class"] = m.group(1)

    # --- 主块: this->Input/Output/Attr (910b 主平台) ---
    for m in re.finditer(
        r'this->Input\("(\w+)"\)(.*?)(?=this->Input\("|this->Output\("|this->Attr\("|this->AICore|OpAICoreConfig)',
        text, re.DOTALL):
        name, block = m.group(1), m.group(2)
        pt = re.search(r'\.ParamType\((\w+)\)', block)
        dt = re.search(r'\.DataType\(\{(.*?)\}\)', block, re.DOTALL)
        dtypes = sorted(set(re.findall(r'ge::(DT_\w+)', dt.group(1)))) if dt else []
        result["inputs"].append({"name": name, "param_type": pt.group(1) if pt else "REQUIRED", "dtypes": dtypes})

    for m in re.finditer(
        r'this->Output\("(\w+)"\)(.*?)(?=this->Input\("|this->Output\("|this->Attr\("|this->AICore|OpAICoreConfig)',
        text, re.DOTALL):
        name, block = m.group(1), m.group(2)
        pt = re.search(r'\.ParamType\((\w+)\)', block)
        dt = re.search(r'\.DataType\(\{(.*?)\}\)', block, re.DOTALL)
        dtypes = sorted(set(re.findall(r'ge::(DT_\w+)', dt.group(1)))) if dt else []
        result["outputs"].append({"name": name, "param_type": pt.group(1) if pt else "REQUIRED", "dtypes": dtypes})

    for m in re.finditer(
        r'this->Attr\("(\w+)"\)(.*?)(?=this->Input\("|this->Output\("|this->Attr\("|this->AICore|OpAICoreConfig)',
        text, re.DOTALL):
        name, block = m.group(1), m.group(2)
        at = re.search(r'\.AttrType\((\w+)\)', block)
        dv = re.search(r'\.(Int|Float)\(([^)]+)\)', block)
        default_val = dv.group(2).strip().replace("ge::", "") if dv else None
        result["attrs"].append({"name": name, "attr_type": at.group(1) if at else "OPTIONAL", "default": default_val})

    result["configs"] = sorted(set(re.findall(r'AddConfig\("(\w+)"', text)))

    # --- 内联 config 块: 按 OpAICoreConfig 切分块,块内提 Input/Output ---
    # 切分: 从第一个 OpAICoreConfig 开始,每块到下一个 OpAICoreConfig 或类结束
    config_sections = []
    for m in re.finditer(r'OpAICoreConfig\s+(\w+)\s*;(.*?)(?=OpAICoreConfig\s+\w+\s*;|' + r'\n\};)', text, re.DOTALL):
        var = m.group(1)
        block = m.group(2)
        config_sections.append((var, block))

    for var, block in config_sections:
        # 找 AddConfig("plat", var) 建立平台映射
        plat_match = re.search(
            r'this->AICore\(\)\.AddConfig\("(\w+)",\s*' + re.escape(var), text)
        if not plat_match:
            continue
        plat = plat_match.group(1)
        plat_dtypes = {}

        # 块内提 Input/Output DataType(无复杂 lookahead——块已切好)
        for kind, pattern in [("Input", r'\.Input\("(\w+)"\)'), ("Output", r'\.Output\("(\w+)"\)')]:
            for m in re.finditer(pattern, block):
                name = m.group(1)
                # 从匹配位置往后找 DataType
                tail = block[m.end():m.end() + 2000]
                dt = re.search(r'\.DataType\(\s*\{(.*?)\}\)', tail, re.DOTALL)
                if dt:
                    dtypes = sorted(set(re.findall(r'ge::(DT_\w+)', dt.group(1))))
                    if name not in plat_dtypes:
                        plat_dtypes[name] = dtypes
                    else:
                        plat_dtypes[name] = sorted(set(plat_dtypes[name] + dtypes))

        if plat_dtypes:
            result["platform_dtypes"][plat] = plat_dtypes

    return result


# ─── step 4: .md 文档解析 ─────────────────────────────────────────

def parse_md_doc(filepath):
    text = Path(filepath).read_text(encoding="utf-8")
    result = {"dim_constraints": None, "notes": []}
    dm = re.search(r'维度.*?(\d+\s*-\s*\d+)', text, re.DOTALL)
    if dm: result["dim_constraints"] = dm.group(1)
    if "INT4" in text:
        result["notes"].append("INT4(quint4x2): output last dim = input last dim / 8 (packed as int32)")
    if re.search(r'INT32.*?整除.?8', text):
        result["notes"].append("INT32: input last dim must be divisible by 8")
    return result


def parse_md_param_table(md_path, input_params, output_specs):
    """解析 .md 文档第一个 <table>，提取每参数约束 → Dict[str, ConstraintInfo]。
    HTML 表 8 列固定: 参数名|输入/输出|描述|使用说明|数据类型|数据格式|维度|非连续Tensor
    列3(索引4)「使用说明」= 约束 <ul><li>，列6(索引7)「维度」= shape 信息。
    产出 key 统一为 snake_case(def 口径),非 md 驼峰。
    """
    text = Path(md_path).read_text(encoding="utf-8")
    tables = re.findall(r'<table[^>]*>(.*?)</table>', text, re.DOTALL)
    if not tables:
        return {}

    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tables[0], re.DOTALL)
    constraints = {}
    # def 参数信息: {snake_name: {"dtypes": [...], "optional": bool}}
    def_params = {}
    for p in input_params:
        def_params[p["name"]] = {"dtypes": p.get("dtypes", []), "optional": p.get("param_type") == "OPTIONAL"}
    for o in output_specs:
        def_params[o["name"]] = {"dtypes": o.get("dtypes", []), "optional": False}

    def _clean_html(tag_text):
        return re.sub(r'<[^>]+>', ' ', tag_text)

    def _to_snake(camel):
        """CamelCase → snake_case"""
        return re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', camel).lower()

    def _match_def_name(md_name):
        md_lower = md_name.lower().replace('_', '')
        # 1. 精确匹配(去下划线后相等)
        for dname in def_params:
            if dname.lower().replace('_', '') == md_lower:
                return dname
        # 2. 受控前缀匹配: md 名去掉常见后缀(optional/out/ref)后再精确比
        #    覆盖 smoothScalesOptional→smooth_scales、yOut→y、offsetOut→offset
        stripped = md_lower
        for suffix in ("optional", "out", "ref"):
            if stripped.endswith(suffix) and len(stripped) > len(suffix):
                stripped = stripped[: -len(suffix)]
                break
        for dname in def_params:
            if dname.lower().replace('_', '') == stripped:
                return dname
        # 不做双向子串模糊匹配——短名(x/y)会灾难性错配
        # (如 "x" 命中 groupIndexOptional/executor)。匹配不上交给 _to_snake 兜底。
        return None

    def _is_api_param(name):
        skip = {"workspacesize", "executor", "stream"}
        return name.lower().replace("（", "(").split("(")[0].strip().lower() not in skip

    for row in rows[1:]:
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
        if len(cells) < 7:
            continue
        raw_name = re.sub(r'<[^>]+>', '', cells[0]).strip()
        md_name = raw_name.split('（')[0].strip()
        if not md_name or not _is_api_param(raw_name):
            continue

        # C1: 标准化为 def snake 名
        param_name = _match_def_name(md_name)
        if param_name is None:
            param_name = _to_snake(md_name)

        usage = _clean_html(cells[3])
        usage = re.sub(r'\s+', ' ', usage).strip()
        dim_text = _clean_html(cells[6]).strip()
        dim_text = re.sub(r'\s+', '', dim_text)

        # C2: optional 取 def ParamType(确定性), 不靠 md 措辞
        def_info = def_params.get(param_name, {})
        ci = ConstraintInfo(param=param_name, optional=def_info.get("optional", False))

        # C3: supports_empty 只对该参数的约束文本
        if "支持空Tensor" in usage or "支持空tensor" in usage.lower():
            ci.supports_empty = True

        # dtype_link
        for sent in re.split(r'[。；]', usage):
            sent = sent.strip()
            dm = re.search(r'数据类型.*?(?:要和|与)([a-zA-Z0-9_]+).*?(?:一致|相同)', sent)
            if dm:
                ci.dtype_link = dm.group(1).strip('`')
                break

        # Bug2: 扫描全部句子,取最高优先级: same > same_except_last > last_dim > without_last_dim
        link_priority = {"same": 0, "same_except_last": 1, "last_dim": 2, "without_last_dim": 3}
        # 降维限定词: 句子含这些词时, "shape与X一致" 不是完全 same, 而是降维联动
        # (如 "维度比x少一维, shape与对应维度x一致" = without_last_dim, 不能判 same)
        DOWNRANK_HINT = re.compile(r'少一维|少1维|减一维|减1维|去除.*?维|剔除.*?维|去除尾轴|剔除尾轴|比.{0,6}少|对应维度|除.*?维外')
        best_rule, best_link = None, None
        for sent in re.split(r'[。；]', usage):
            sent = sent.strip()
            has_downrank = bool(DOWNRANK_HINT.search(sent))
            if 'shape与' in sent and '一致' in sent and '(' not in sent and not has_downrank:
                ref_m = re.search(r'shape与([a-zA-Z0-9_]+)', sent)
                best_link = {"ref": ref_m.group(1) if ref_m else "x", "rule": "same"}
                break  # same 是最高优先级,直接停
            elif ('其余维度' in sent and '一致' in sent) or ('对应维度' in sent and '一致' in sent and has_downrank):
                if best_rule is None or link_priority.get("same_except_last", 99) < link_priority.get(best_rule, 99):
                    ref_m = re.search(r'(?:与|和)([a-zA-Z0-9_]+)', sent)
                    best_rule, best_link = "same_except_last", {"ref": ref_m.group(1) if ref_m else "x", "rule": "same_except_last"}
            elif '最后一维相同' in sent:
                ref_m = re.search(r'与([a-zA-Z0-9_]+)', sent)
                if best_rule is None or link_priority.get("last_dim", 99) < link_priority.get(best_rule, 99):
                    best_rule = "last_dim"
                    best_link = {"ref": ref_m.group(1) if ref_m else "x", "rule": "last_dim"}
            elif '剔除最后一维' in sent or '去除尾轴' in sent or '剔除尾轴' in sent:
                ref_m = re.search(r'(?:与|和)([a-zA-Z0-9_]+)', sent)
                if best_rule is None or link_priority.get("without_last_dim", 99) < link_priority.get(best_rule, 99):
                    best_rule, best_link = "without_last_dim", {"ref": ref_m.group(1) if ref_m else "x", "rule": "without_last_dim"}
        if best_link:
            ci.shape_link = best_link

        if dim_text and dim_text.isdigit():
            ci.rank = int(dim_text)
        dm = re.match(r'(\d+)\s*-\s*(\d+)', dim_text)
        if dm:
            ci.dim_range = (int(dm.group(1)), int(dm.group(2)))
        # 自然语言维度约束 → dim_range (None 表该侧无界)
        # "大于N维"→下界 N+1; "小于N维"→上界 N-1; "N维及以上"/"至少N维"→下界 N
        elif dim_text:
            m_gt = re.search(r'大于(\d+)维', dim_text)
            m_lt = re.search(r'小于(\d+)维', dim_text)
            m_ge = re.search(r'(?:大于等于|至少)(\d+)维|(\d+)维及以上', dim_text)
            if m_ge:
                n = int(m_ge.group(1) or m_ge.group(2))
                ci.dim_range = (n, None)
            elif m_gt:
                ci.dim_range = (int(m_gt.group(1)) + 1, None)
            elif m_lt:
                ci.dim_range = (None, int(m_lt.group(1)) - 1)

        ci.valid_dtypes = def_info.get("dtypes", [])
        constraints[param_name] = ci

    return constraints


# ─── step 5: 构建 CHECKLIST.md 段落 ───────────────────────────────

def fmt_dtypes(dtypes):
    return ", ".join(GE_DTYPE_MAP.get(d, d) for d in dtypes) if dtypes else "unknown"

def build_all(operator_name, torch_npu_fn, op_type, target_dir, mode, classification,
              def_cpp_path, md_doc_path, def_info, md_info,
              constraints=None, platform_dtypes=None):
    """构建 CHECKLIST 段落。constraints = {param_name: ConstraintInfo}, platform_dtypes = {plat: {param: [dtype]}}"""
    if constraints is None:
        constraints = {}
    if platform_dtypes is None:
        platform_dtypes = {}

    def _dtype_with_platform(name, main_dtypes):
        """给 main dtypes 加平台扩展标注。"""
        parts = [fmt_dtypes(main_dtypes) if main_dtypes else "—"]
        for plat, pdict in platform_dtypes.items():
            if name in pdict:
                ext = [GE_DTYPE_MAP.get(d, d) for d in pdict[name]]
                ext_new = [d for d in ext if d not in parts[0]]
                if ext_new:
                    parts.append(f"{','.join(ext_new)}({plat})")
        return ', '.join(parts)

    lines = [
        "## UT Generation Info",
        "",
        "### Operator Basic Info",
        f"- Operator Name: {operator_name or '(to be filled manually)'}",
        f"- torch_npu function name: {torch_npu_fn or '(to be filled manually)'}",
        f"- Operator Type: {op_type}",
        f"- Target Directory: {target_dir}",
        f"- Mode: {mode}",
        f"- Operator Classification: {classification}",
        "",
        "### Source File Paths",
        f"- _def.cpp: {def_cpp_path or 'not found'}",
        f"- .md doc: {md_doc_path or 'not found'}",
        "",
        "### Input/Output Spec",
        "",
        "**Input parameter list**:",
        "",
        "| Param Name | Type | Required/Optional | Supported dtypes | Constraints |",
        "|------------|------|-------------------|------------------|-------------|",
    ]
    for inp in def_info["inputs"]:
        pt = "Required" if inp["param_type"] == "REQUIRED" else "Optional"
        dtype_str = _dtype_with_platform(inp['name'], inp['dtypes'])
        # Constraints column left for model to fill (step 2) — script does not pre-fill.
        # Note: structured constraints extracted from .md are still written to .constraint_info.json for negative cases (independent channel).
        lines.append(f"| {inp['name']} | Tensor | {pt} | {dtype_str} | (to be filled by model) |")
    for attr in def_info["attrs"]:
        at = "Required" if attr["attr_type"] == "REQUIRED" else "Optional"
        dv = attr["default"] or "None"
        # Default value from _def.cpp (deterministic fact; keep). Valid-value enumeration etc. left for model.
        lines.append(f"| {attr['name']} | int (Attr) | {at} | — | Default: {dv}; (to be filled by model) |")

    lines.extend([
        "",
        "**Output specs**:",
        "",
        "| Output Name | Type | Shape Rule | Dtype Rule |",
        "|-------------|------|------------|------------|",
    ])
    for out in def_info["outputs"]:
        dtype_str = _dtype_with_platform(out['name'], out['dtypes'])
        lines.append(f"| {out['name']} | Tensor | (to be filled by model) | {dtype_str} |")

    if md_info.get("notes"):
        lines.append("")
        lines.append("**Constraints extracted from operator doc**:")
        for n in md_info["notes"]:
            lines.append(f"- {n}")
    if md_info.get("dim_constraints"):
        lines.append(f"- 维度支持范围: {md_info['dim_constraints']}")

    lines.extend([
        "",
        "### Coverage Matrix",
        "> To be filled by T-scaffold-coverage or T-coverage.",
        "",
    ])
    return "\n".join(lines)



# ─── main ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="scaffold init — 一次性初始化 CHECKLIST.md UT 段落")
    p.add_argument("--checklist", default=None, help="CHECKLIST.md 路径")
    p.add_argument("--search-dir", default=None, help="搜索根目录")
    p.add_argument("--op-name", default=None,
                   help="显式算子名(torch_npu fn 去 npu_ 前缀, 如 dynamic_quant)。"
                        "CHECKLIST 无 'torch_npu function name:' 标记时用此定向。")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    search_dir = Path(args.search_dir) if args.search_dir else Path.cwd()
    checklist_path = args.checklist or str(search_dir / "CHECKLIST.md")
    cp = Path(checklist_path)

    if not cp.exists():
        print(f"[ERROR] CHECKLIST.md 不存在: {checklist_path}")
        return 1

    text = cp.read_text(encoding="utf-8")

    # step 1: 从 CHECKLIST 提取
    operator_name = args.op_name.strip() if args.op_name else extract_op_name(text)
    if args.op_name:
        print(f"[INFO] 算子名由 --op-name 显式指定: {operator_name}")
    mode_from_cl = extract_mode(text)
    torch_npu_fn = f"npu_{operator_name}" if operator_name else None
    op_type = determine_op_type(torch_npu_fn)
    target_dir = determine_target_dir(op_type)
    mode = determine_mode(search_dir, torch_npu_fn, target_dir, mode_from_cl)
    classification = classify(operator_name)

    # 合法性自检: classify 输出必须是 scaffold_coverage.CLASS_SHAPES 已知键,
    # 否则下游量化/attention 等专项分支静默失效。词表漂移 → 大声告警。
    _KNOWN_CLASSES = {"quantization", "attention", "matmul", "activation", "normalization", "other"}
    if classification not in _KNOWN_CLASSES:
        print(f"[WARN] classify('{operator_name}') = '{classification}' 不在已知分类 {sorted(_KNOWN_CLASSES)} "
              f"→ 下游覆盖矩阵专项分支将失效, 请核对 CLASSIFY_KW 与 scaffold_coverage.CLASS_SHAPES 同步")

    print(f"[INFO] operator_name:  {operator_name or '未提取到'}")
    print(f"[INFO] torch_npu_fn:   {torch_npu_fn or '未提取到'}")
    print(f"[INFO] op_type:        {op_type}")
    print(f"[INFO] target_dir:     {target_dir}")
    print(f"[INFO] mode:           {mode}")
    print(f"[INFO] classification: {classification}")

    # step 2-3: 找 + 解析源文件
    def_cpp = find_def_cpp(operator_name, search_dir) if operator_name else None
    md_doc = find_md_doc(operator_name, search_dir) if operator_name else None
    print(f"[FILE] _def.cpp: {def_cpp or '未找到'}")
    print(f"[FILE] .md:      {md_doc or '未找到'}")

    def_info = {"inputs": [], "outputs": [], "attrs": [], "configs": []}
    md_info = {"dim_constraints": None, "notes": []}
    if def_cpp:
        def_info = parse_def_cpp(def_cpp)
        print(f"[PARSE] in={len(def_info['inputs'])} out={len(def_info['outputs'])} attr={len(def_info['attrs'])}")
    if md_doc:
        md_info = parse_md_doc(md_doc)
        print(f"[PARSE] dim={md_info['dim_constraints']} notes={len(md_info['notes'])}")

    # step 4: .md 参数表约束抽取 + 序列化 ConstraintInfo
    constraints = {}
    if md_doc:
        # 合并 def 解析信息构造 input_params/output_specs 供 valid_dtypes 匹配
        _input_params = [{"name": inp["name"], "dtypes": [GE_DTYPE_MAP.get(d, d) for d in inp["dtypes"]],
                          "param_type": inp.get("param_type", "REQUIRED")}
                         for inp in def_info["inputs"]]
        _output_specs = [{"name": out["name"], "dtypes": [GE_DTYPE_MAP.get(d, d) for d in out["dtypes"]]}
                         for out in def_info["outputs"]]
        constraints = parse_md_param_table(md_doc, _input_params, _output_specs)
        # 缺陷D: valid_dtypes 合并全平台 union(主块 + platform_dtypes)
        pd = def_info.get("platform_dtypes", {})
        for ci in constraints.values():
            merged = set(ci.valid_dtypes)
            for plat, pdict in pd.items():
                if ci.param in pdict:
                    merged.update(GE_DTYPE_MAP.get(d, d) for d in pdict[ci.param])
            ci.valid_dtypes = sorted(merged)
        print(f"[PARSE] md constraints: {len(constraints)} params extracted")

        # 序列化 ConstraintInfo → .constraint_info.json
        if not args.dry_run and constraints:
            json_path = Path(checklist_path).parent / ".constraint_info.json"
            ci_list = [asdict(ci) for ci in constraints.values()]
            json_path.write_text(json.dumps(ci_list, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[OK] constraint_info → {json_path}")

    # step 5: 构建全部段落
    section = build_all(operator_name, torch_npu_fn, op_type, target_dir, mode, classification,
                        str(def_cpp) if def_cpp else None,
                        str(md_doc) if md_doc else None,
                        def_info, md_info,
                        constraints=constraints,
                        platform_dtypes=def_info.get("platform_dtypes", {}))

    # step 5: 写入
    current = cp.read_text(encoding="utf-8")
    if "## UT Generation Info" in current and not args.force:
        print("[SKIP] '## UT Generation Info' 已存在（幂等）。用 --force 覆盖。")
        return 0

    if args.force and "## UT Generation Info" in current:
        current = re.sub(r'\n+## UT Generation Info.*', '', current, flags=re.DOTALL)
        print("[OK] 已移除旧的 '## UT Generation Info'")

    if args.dry_run:
        print("=" * 60)
        print("[DRY RUN] 将写入:")
        print("=" * 60)
        print(section)
        print("=" * 60)
        return 0

    cp.write_text(current.rstrip() + "\n\n" + section + "\n", encoding="utf-8")
    print(f"[OK] 已写入 '## UT Generation Info' -> {checklist_path}")
    print()
    print("✅ scaffold_init complete. Step 2 can now read full Operator Basic Info + Source File Paths + Input/Output Spec.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
