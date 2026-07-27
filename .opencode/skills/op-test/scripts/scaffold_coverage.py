#!/usr/bin/env python3
"""
CHECKLIST.md 覆盖矩阵脚手架 — 在 scaffold_init 之后执行。
从 CHECKLIST.md 已写入的结构化参数表读取信息，按算子分类和覆盖规则
生成半成品覆盖矩阵，写入 CHECKLIST.md。

用法:
  python scaffold_coverage.py [--checklist CHECKLIST路径] [--dry-run]

前置条件: scaffold_init 已写入算子基本信息和输入输出规格。
产出:     CHECKLIST.md「覆盖矩阵」— 测试项已枚举。

脚本职责(A 侧): 只做机械穷举的基线矩阵(dtype 全覆盖 / 可选参数 product 组合 /
大 shape / 平台扩展 / 维度覆盖)。**不解析「约束」列** —— 行增删(不适用删行)、
交互展开全由步骤3 模型读「约束」列自然语言处理。
负向用例走独立通道(constraint_info.json → generate_negative_cases)。

脚本保证: 每个可选参数至少1个测试、每个输出dtype至少1个测试、分类必含场景全部列出。
"""

import re
import sys
import json
import argparse
import itertools
from pathlib import Path
from collections import defaultdict
from constraint_info import ConstraintInfo


# ─── dtype → torch short name ────────────────────────────────────

DTYPE_SHORT = {
    "float16": "fp16", "bfloat16": "bf16", "float32": "fp32",
    "int8": "int8", "quint4x2": "int4", "int32": "int32",
    "hifloat8": "hf8", "float8_e4m3fn": "f8e4m3", "float8_e5m2": "f8e5",
}

# 非法 dtype 候选池(优先级序)，供负向 invalid_dtype 选取
# 取 CANDIDATES 中首个不在该参数 valid_dtypes 的 dtype
INVALID_DTYPE_CANDIDATES = ["float32", "int64", "int32", "int8", "uint8", "bool", "float64"]

# ─── 分类标签归一 (RC9) ──────────────────────────────────────────
# scaffold_init.CLASSIFY_KW 产出全称(quantization/...), 但 CHECKLIST「算子分类」
# 字段历史上出现过短写(quant/norm/...)与上游手填值。消费侧若直接 == "quantization"
# 比较, 短写会静默落 other 分支 → 量化专项全跳过、shape 退化。统一在入口归一。
CLASS_ALIAS = {
    "quant": "quantization", "dynamic_quant": "quantization",
    "group_quant": "quantization", "dequant": "quantization",
    "quantize": "quantization",
    "attn": "attention",
    "mm": "matmul", "ffn": "matmul", "moe": "matmul", "routing": "matmul",
    "norm": "normalization", "rms_norm": "normalization", "layer_norm": "normalization",
    "act": "activation",
}


def normalize_classification(raw):
    """归一算子分类标签 → CLASS_SHAPES 已知键。
    返回 (canonical, warning|None)。非法且无别名 → 落 other 但回传告警(不静默)。"""
    c = (raw or "").strip().lower()
    canonical = CLASS_ALIAS.get(c, c)
    if canonical in CLASS_SHAPES:
        warn = None if canonical == c else f"分类标签 '{raw}' 归一为 '{canonical}'"
        return canonical, warn
    # other 是 CLASS_SHAPES 合法键但语义=兜底; 未知标签明确告警
    return "other", f"未知算子分类 '{raw}' → 落 'other' 兜底分支(量化/attention 等专项不生效)"


# ─── 分类 → 必含场景 ──────────────────────────────────────────────

CLASS_SCENARIOS = {
    "quantization": [
        ("dst_type all values", "each dst_type value tested individually"),
        ("smooth_scales", "smooth_scales passed and smooth_scales=None both tested"),
        ("group_index", "MoE group_index param test (if operator has this param)"),
        ("INT4 shape verify", "INT4 output packed shape check (if operator supports INT4)"),
    ],
    "attention": [
        ("input_layout all values", "BSH/BNSD/BSND/TND/NTD etc."),
        ("PFA/IFA/PageAttention", "multiple attention modes"),
        ("atten_mask", "attention mask param"),
        ("actual_seq_lengths", "actual sequence lengths param"),
        ("block_table", "PageAttention block_table param"),
    ],
    "matmul": [
        ("split_item all values", "each split_item value"),
        ("bias", "bias param"),
        ("quantization params", "scale/offset/antiquant_scale etc."),
        ("group_list", "group_list param"),
    ],
    "activation": [
        ("all supported dtypes", "at least 1 test per dtype"),
        ("in-place/out-of-place", "if operator supports"),
        ("boundary input values", "max/min/zero values"),
    ],
    "normalization": [
        ("all supported dtypes", "at least 1 test per dtype"),
        ("epsilon values", "default, very small, relatively large"),
        ("bias/no-bias", "with bias and without bias"),
        ("different dimensions", "1D/2D/3D normalization"),
    ],
}

# ─── 分类 → 推荐 shape ────────────────────────────────────────────

CLASS_SHAPES = {
    "quantization":    {"2D": (32, 256), "3D": (2, 32, 256), "large": (3072, 7168), "large_3d": (8, 512, 7168)},
    "attention":       {"2D": (2, 128), "3D": (1, 16, 128), "large": (8, 512, 64)},
    "matmul":          {"2D": (256, 256), "3D": (2, 256, 128), "large": (1024, 1024), "large_3d": (8, 1024, 1024)},
    "activation":      {"1D": (4096,), "2D": (128, 128), "3D": (4, 32, 128)},
    "normalization":   {"2D": (2, 256), "3D": (2, 32, 256), "large": (32, 2048), "large_3d": (8, 512, 2048)},
    "other":           {"2D": (4, 32), "3D": (2, 4, 32)},
}


# ─── CHECKLIST.md 解析 ────────────────────────────────────────────

def read_checklist(path):
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None


def parse_section(text, header, stop_at_next_section=True):
    """提取 ## header 到下一个 ## 之间的内容。"""
    pattern = re.escape(header) + r'\n(.*?)(?=\n## |\Z)'
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1) if m else ""


def parse_kv_list(text):
    """解析 '- key: value' 列表。"""
    result = {}
    for line in text.strip().split("\n"):
        m = re.match(r'- (.+?):\s*(.+)', line)
        if m:
            result[m.group(1).strip()] = m.group(2).strip()
    return result


def parse_markdown_table(text):
    """解析 markdown 表格，返回 [{col: val}]。"""
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []
    # 找到表头行（包含 |）
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
        # first/last may be empty from leading/trailing |
        cells_clean = [c for c in cells if c]
        if len(cells_clean) >= len(headers):
            row = {}
            for j, h in enumerate(headers):
                row[h] = cells_clean[j] if j < len(cells_clean) else ""
            rows.append(row)
    return rows


def find_table_in_section(text, table_label=None):
    """在文本中定位 markdown 表格。"""
    # 如果有标签（如 **Input parameter list**:），先定位
    if table_label:
        idx = text.find(table_label)
        if idx >= 0:
            text = text[idx + len(table_label):]
    # 找到 |---| 表格分隔行
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r'^\|[\-\s\|:]+\|$', line) and i > 0:
            # 向上找表头
            header = lines[i-1] if i > 0 else ""
            data_lines = []
            j = i + 1
            while j < len(lines) and "|" in lines[j]:
                data_lines.append(lines[j])
                j += 1
            return header + "\n" + lines[i] + "\n" + "\n".join(data_lines)
    return ""


def parse_input_params(text):
    """从输入输出规格段落提取输入参数列表。"""
    table_text = find_table_in_section(text, "**Input parameter list**:")
    if not table_text:
        # fallback: find first table
        table_text = find_table_in_section(text)
    return parse_markdown_table(table_text)


def parse_output_specs(text):
    """从输入输出规格段落提取输出规格列表。"""
    table_text = find_table_in_section(text, "**Output specs**:")
    if not table_text:
        # fallback: find second table
        tables = list(re.finditer(r'^\|[\-\s\|:]+\|$', text, re.MULTILINE))
        if len(tables) >= 2:
            idx = tables[1].start()
            # search backwards for header
            before = text[:idx].rstrip().split("\n")
            header = before[-1] if before else ""
            after = text[idx:].split("\n")
            data = [after[0]]  # separator
            for line in after[1:]:
                if "|" in line:
                    data.append(line)
                else:
                    break
            table_text = header + "\n" + "\n".join(data)
    return parse_markdown_table(table_text)


def extract_constraints(text):
    """从输入输出规格段落提取约束备注。"""
    notes = []
    m = re.search(r'\*\*Constraints extracted from operator doc\*\*:(.*?)(?=\n###|\n##|\Z)', text, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            line = line.strip("- ").strip()
            if line:
                notes.append(line)
    return notes


# ─── dtype 提取 ──────────────────────────────────────────────────

def parse_dtype_list(dtype_str):
    """'int8, quint4x2' / 'float16; bfloat16' → ['int8','quint4x2']
    分隔符容错: 英文逗号/分号 + 中文顿号/分号。step2 模型常把 ', ' 改成 '; '。"""
    if not dtype_str or dtype_str in ("—", "-", ""):
        return []
    return [d.strip() for d in re.split(r'[,;、；]', dtype_str) if d.strip()]


# ─── 类型列容错 (RC2/RC3): 机器格式 vs 模型签名格式 ────────────────
# scaffold_init 写 "Tensor" / "int (Attr)"; step2 模型常改成 "Tensor?" /
# "Tensor? =None" / "int?" / "int? =None"。用前缀判别, 两种格式都吃。

def is_tensor_type(t):
    return str(t).strip().lower().startswith("tensor")


def is_int_attr_type(t):
    return str(t).strip().lower().startswith("int")


# ─── 散文 dtype 抽 token (RC4/RC5) ────────────────────────────────
# 输出「dtype规则」列是 step2 模型写的自然语言(如
# "dst_type 决定: 默认int8; INT4→int32; FP8_E4M3→float8_e4m3fn ...")。
# 不能当逗号枚举切, 改成扫已知 dtype 词 + 大写别名。

_DTYPE_ALIAS = {
    "int4": "quint4x2", "fp8_e4m3": "float8_e4m3fn", "fp8e4m3": "float8_e4m3fn",
    "fp8_e5m2": "float8_e5m2", "fp8e5m2": "float8_e5m2", "hifloat8": "hifloat8",
}


def extract_dtype_tokens(cell):
    """从散文/枚举单元格抽规范 dtype 集合。先试纯枚举(逗号/分号), 再扫散文词。"""
    if not cell or cell in ("—", "-", ""):
        return set()
    # 纯枚举优先(每段都是已知 dtype 名 → 直接用)
    parts = parse_dtype_list(cell)
    if parts and all(p in DTYPE_SHORT for p in parts):
        return set(parts)
    found = set()
    for tok in DTYPE_SHORT:
        if re.search(rf'(?<![a-z0-9_]){re.escape(tok)}(?![a-z0-9_])', cell, re.I):
            found.add(tok)
    for alias, canon in _DTYPE_ALIAS.items():
        if re.search(rf'(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])', cell, re.I):
            found.add(canon)
    return found


def parse_dtype_list_from_cell(cell_str):
    """从表格单元格提取 dtype 列表，支持 'float16, bfloat16' 格式。"""
    return parse_dtype_list(cell_str)


# ─── 覆盖矩阵生成 ────────────────────────────────────────────────

def dtype_short(dtype_name):
    return DTYPE_SHORT.get(dtype_name, dtype_name)


def get_primary_name(input_params):
    """主输入参数名 —— 首个必选 Tensor 的「参数名」。
    本算子叫 input(不是 x), 不能 hardcode。兜底链: 必选Tensor→首个输入→"x"。"""
    required = [p for p in input_params if p.get("Required/Optional") == "Required"]
    for p in required:
        if is_tensor_type(p.get("Type")):
            n = p.get("Param Name", "").strip()
            if n:
                return n
    if input_params:
        n = input_params[0].get("Param Name", "").strip()
        if n:
            return n
    return "x"


def shape_for_dim(classification, ndim):
    shapes = CLASS_SHAPES.get(classification, CLASS_SHAPES["other"])
    if ndim == 1:
        return shapes.get("1D", shapes.get("2D"))
    elif ndim == 2:
        return shapes.get("2D")
    else:
        return shapes.get("3D", shapes.get("2D"))


def generate_coverage(input_params, output_specs, classification, constraints, constraint_infos=None):
    """核心：按规则生成覆盖矩阵行。"""
    rows = []
    used_names = set()

    def add(test_name, params, dtype_, shape, scenario):
        # 去重
        if test_name in used_names:
            return
        used_names.add(test_name)
        rows.append({
            "Test Item": test_name,
            "Covered Params": params,
            "Data Type": dtype_,
            "Dimensions": str(shape),
            "Scenario": scenario,
        })

    # --- 提取信息 ---
    optional_params = [p for p in input_params if p.get("Required/Optional") == "Optional"]
    required_params = [p for p in input_params if p.get("Required/Optional") == "Required"]

    # 主输入名 —— 取首个必选 Tensor 的「参数名」(本算子叫 input, 不能 hardcode "x")。
    # 覆盖参数列/命名/约束查找全用此名。
    primary_name = get_primary_name(input_params)

    # 主输入(x)的 dtype —— 只用第一个必选 Tensor 的 dtype
    primary_input_dtypes = set()
    for p in required_params:
        if is_tensor_type(p.get("Type")):
            primary_input_dtypes = set(parse_dtype_list_from_cell(p.get("Supported dtypes", "")))
            break
    # 兜底
    if not primary_input_dtypes and input_params:
        primary_input_dtypes = set(parse_dtype_list_from_cell(input_params[0].get("Supported dtypes", "")))
    if not primary_input_dtypes:
        primary_input_dtypes = {"float16"}

    # 可选 Tensor 参数的 dtype（如 smooth_scales 的 float16/bf16）——用于参数组合
    optional_dtypes = set()
    for p in optional_params:
        if is_tensor_type(p.get("Type")):
            optional_dtypes.update(parse_dtype_list_from_cell(p.get("Supported dtypes", "")))

    output_dtypes_all = set()
    for o in output_specs:
        output_dtypes_all.update(extract_dtype_tokens(o.get("Dtype Rule", "")))
    # scale dtype (float32) 不算用户可控的输出 dtype
    primary_output_dtypes = {d for d in output_dtypes_all if d != "float32"}

    has_smooth = any("smooth" in p.get("Param Name", "").lower() for p in input_params)
    has_group_index = any("group" in p.get("Param Name", "").lower() for p in input_params)
    has_dst_type = any(is_int_attr_type(p.get("Type", "")) and "dst" in p.get("Param Name", "").lower() for p in input_params)
    has_int4 = "quint4x2" in output_dtypes_all
    dim_info = None
    for c in constraints:
        m = re.search(r'(\d+\s*-\s*\d+)', c)
        if m:
            dim_info = m.group(1)
            break
    # RC6 兜底: notes 段无维度信息时, 扫主输入「约束」列(模型常把 "dim 2-8" 写这)
    if not dim_info:
        for p in (required_params + input_params):
            m = re.search(r'(\d+)\s*-\s*(\d+)\s*维?', p.get("Constraints", ""))
            if m:
                dim_info = f"{m.group(1)}-{m.group(2)}"
                break

    def _find_ci_for_coverage(name, ci_list):
        """generate_coverage 内约束查找(与负向的 _find_ci 同语义)。"""
        for ci in ci_list:
            if ci.param == name:
                return ci
        name_lower = name.lower().replace('_', '')
        for ci in ci_list:
            if ci.param.lower().replace('_', '') == name_lower:
                return ci
        return None

    default_dtype = next(iter(primary_input_dtypes)) if primary_input_dtypes else "float16"
    shapes = CLASS_SHAPES.get(classification, CLASS_SHAPES["other"])

    # --- 通用：基本功能（只用主输入 dtype）---
    for dt in sorted(primary_input_dtypes):
        dtype_label = dtype_short(dt)
        shape_2d = shapes.get("2D", (4, 32))
        add(f"test_basic_{dtype_label}", primary_name, dt, shape_2d, "Basic functionality")

    # --- 通用：维度覆盖 ---
    if dim_info:
        dims = [int(x) for x in re.findall(r'\d+', dim_info)]
        min_dim = min(dims)
        max_dim = max(dims)
        if min_dim <= 2:
            add("test_min_dimension", primary_name, default_dtype,
                shapes.get("2D", (4, 32)), f"边界-最小维度({min_dim}D)")
        if max_dim >= 3:
            add("test_3d_input", primary_name, default_dtype,
                shapes.get("3D", (2, 4, 32)), "3D输入")

    # --- 通用：可选参数组合(product) ---
    optional_tensor_params = [p for p in optional_params
                              if is_tensor_type(p.get("Type")) and "dst" not in p.get("Param Name", "").lower()]
    # 限制组合数: 最多 3 个可选 Tensor → 2^N ≤ 8
    combo_params = optional_tensor_params[:3]
    for combo in itertools.product([True, False], repeat=len(combo_params)):
        present = [combo_params[i]["Param Name"] for i, on in enumerate(combo) if on]
        if not present:
            continue  # 全不传由基本功能覆盖
        param_str = f"{primary_name}, " + ", ".join(present)
        combo_label = "_".join(present)
        combo_desc = "可选参数-" + "+".join(present)
        # 每个组合覆盖主流 dtype(f16/bf16 各一)
        for dt in sorted(primary_input_dtypes)[:2]:
            dt_label = dtype_short(dt)
            add(f"test_{combo_label}_{dt_label}", param_str, dt,
                shapes.get("3D", (2, 4, 32)), f"{combo_desc}({dt_label})")

    # --- 量化：dst_type 全值 = 输出 dtype 覆盖（二合一，避免重复）---
    if classification == "quantization" and has_dst_type:
        for odt in sorted(primary_output_dtypes):
            dtype_label = dtype_short(odt)
            # INT4 用偶数末维
            if "int4" in dtype_label or "quint4" in dtype_label.lower():
                test_shape = shapes.get("2D", (4, 64))
                # 确保末维能被 8 整除（int4 packing）
                last_dim = test_shape[-1]
                if last_dim % 8 != 0:
                    test_shape = tuple(list(test_shape[:-1]) + [(last_dim // 8 + 1) * 8])
            else:
                test_shape = shapes.get("2D", (4, 32))
            add(f"test_dst_type_{dtype_label}", f"{primary_name}, dst_type", default_dtype,
                test_shape, f"dst_type={odt}")
        if has_int4:
            # INT4 形状验证用能被 8 整除的末维
            add("test_int4_output_shape", f"{primary_name}, dst_type", default_dtype,
                (4, 64), "INT4-输出形状验证(末维/8)")
    else:
        # 非量化或无 dst_type：输出 dtype 覆盖
        for odt in sorted(primary_output_dtypes):
            dtype_label = dtype_short(odt)
            add(f"test_output_dtype_{dtype_label}", f"{primary_name}, dst_type" if has_dst_type else primary_name,
                default_dtype, shapes.get("2D", (4, 32)), f"输出dtype-{odt}")

    # --- 量化专项(组合覆盖已由 product 逻辑处理,此处只保留专项验证) ---
    if classification == "quantization":
        # scale 数值验证(非纯 shape)：cpu_op_exec 的 scale golden 对任意维度都能算，
        # 用例须 assertRtolEqual(cpu_scale, npu_scale)，不得降级成只查 scale.shape。
        add("test_scale_value", primary_name, default_dtype,
            shapes.get("3D", (2, 4, 32)), "scale数值验证(比golden,非纯shape)")
        # 维度5: large shape — 保持 CLASS 基准量级,按 dim_range 决定秩,不膨胀元素量
        # min==2: 算子支持 2D → 直接用 2D CLASS large(方案A,无膨胀)
        # min>2 : 算子要求更高秩 → 用 3D 专用标定值(方案C,量级不变)
        large = shapes.get("large", (3072, 7168))
        if constraint_infos:
            x_ci = _find_ci_for_coverage(primary_name, constraint_infos)
            if x_ci and x_ci.dim_range and x_ci.dim_range[0] > 2:
                large = shapes.get("large_3d", shapes.get("large", (8, 512, 7168)))
        add("test_large_input", primary_name, default_dtype, large, "Large data (must compare scale golden, not pure shape smoke)")

    # --- attention 专项 ---
    if classification == "attention":
        add("test_input_layout_bsh", f"{primary_name}, input_layout", "float16",
            shapes.get("3D", (2, 128, 64)), "input_layout=BSH")
        add("test_input_layout_bnsd", f"{primary_name}, input_layout", "float16",
            shapes.get("3D", (2, 16, 128)), "input_layout=BNSD")
        add("test_with_atten_mask", f"{primary_name}, atten_mask", "float16",
            shapes.get("3D", (2, 128, 64)), "atten_mask")

    # --- matmul 专项 ---
    if classification == "matmul":
        add("test_with_bias", f"{primary_name}, bias", "float16",
            shapes.get("2D", (256, 256)), "可选参数-bias")
        add("test_split_item", f"{primary_name}, split_item", "float16",
            shapes.get("2D", (256, 256)), "split_item 覆盖")

    # --- normalization 专项 ---
    if classification == "normalization":
        add("test_epsilon_large", f"{primary_name}, epsilon", "float16",
            shapes.get("2D", (2, 256)), "epsilon=1e-3")
        add("test_epsilon_small", f"{primary_name}, epsilon", "float16",
            shapes.get("2D", (2, 256)), "epsilon=1e-6")

    # --- 通用：输出 shape/dtype 验证 ---
    # 量化算子跳过：默认 int8 结构已由 test_dst_type_int8(dst_type循环) + test_basic_*(值比对) 覆盖，
    # 再发一条通用 shape 行 = 三重冗余。仅非量化分类保留此结构兜底行。
    if classification != "quantization":
        add("test_output_shape_dtype", primary_name, "float16",
            shapes.get("3D", (2, 4, 32)), "输出shape+dtype断言")

    # --- 通用：旧弱负向已迁移到 generate_negative_cases ---

    # 不解析「约束」列——行增删与交互展开由步骤3 模型处理(脚本与约束列零耦合)。

    return rows


# ─── 写入 ─────────────────────────────────────────────────────────

def generate_negative_cases(input_params, output_specs, constraint_infos, classification,
                            default_dtype, shapes, dim_info):
    """从约束信号派生负向用例。每类有对应约束才生成,无则跳过。返回 7 字段负向行列表。"""
    rows = []

    # 约束索引: {param_name: ConstraintInfo}
    ci_map = {}
    for ci in constraint_infos:
        ci_map[ci.param] = ci

    def _find_ci(name):
        """模糊匹配 def param name → ConstraintInfo。"""
        if name in ci_map:
            return ci_map[name]
        name_lower = name.lower().replace('_', '')
        for md_name, ci in ci_map.items():
            if md_name.lower().replace('_', '') == name_lower:
                return ci
        return None

    base_shape_2d = shapes.get("2D", (4, 32))
    base_shape_3d = shapes.get("3D", (2, 4, 32))
    main_dtype = default_dtype

    # 主输入名(本算子叫 input, 不能 hardcode "x") —— 覆盖参数列/命名/dim约束查找全用此名。
    primary_name = get_primary_name(input_params)

    # 收集 Tensor 参数(必选+可选,为负向 invalid_dtype 全量覆盖)
    all_tensors = [p for p in input_params if is_tensor_type(p.get("Type"))]

    # 收集输出dtype选择 Attr (dst_type 类)
    dst_attrs = [p for p in input_params if is_int_attr_type(p.get("Type", "")) and "dst" in p.get("Param Name", "").lower()]
    all_output_dtypes = set()
    for o in output_specs:
        all_output_dtypes.update(extract_dtype_tokens(o.get("Dtype Rule", "")))

    # --- 1. Tensor 参数 → invalid_dtype(必选+可选) ---
    for p in all_tensors:
        pname = p.get("Param Name", "")
        valid = set(parse_dtype_list_from_cell(p.get("Supported dtypes", "")))
        if not valid:
            continue
        # 取 CANDIDATES 中首个不在 valid 的
        invalid = None
        for cand in INVALID_DTYPE_CANDIDATES:
            if cand not in valid:
                invalid = cand
                break
        if invalid:
            rows.append({
                "Test Item": f"test_neg_{pname}_invalid_dtype",
                "Covered Params": f"{primary_name}, {pname}" if pname != primary_name else primary_name,
                "Data Type": main_dtype,
                "Dimensions": str(base_shape_2d),
                "Scenario": "Negative-dtype domain",
                "Violated Field": pname,
                "Violation Construction": f"dtype={invalid}",
                "Expected": "raises:PARAM_INVALID",
            })

    # --- 2. 输出 shape_link=same → shape_mismatch ---
    for o in output_specs:
        oname = o.get("Output Name", "")
        ci = _find_ci(oname)
        if ci and ci.shape_link and ci.shape_link.get("rule") == "same":
            # 末维×2
            wrong_shape = list(base_shape_2d)
            wrong_shape[-1] *= 2
            rows.append({
                "Test Item": f"test_neg_{oname}_shape_mismatch",
                "Covered Params": f"{primary_name}, {oname}",
                "Data Type": main_dtype,
                "Dimensions": str(base_shape_2d),
                "Scenario": "Negative-shape link",
                "Violated Field": oname,
                "Violation Construction": f"shape={tuple(wrong_shape)}",
                "Expected": "raises:PARAM_INVALID",
            })

    # --- 3. rank=N → invalid_rank(从 ConstraintInfo,覆盖所有有秩约束的参数) ---
    for p in all_tensors + output_specs:
        pname = p.get("Param Name", "") if "Param Name" in p else (p.get("Output Name", "") if "Output Name" in p else "")
        ci = _find_ci(pname)
        if ci and ci.rank is not None:
            wrong_shape = list(base_shape_2d)
            wrong_shape.insert(0, 1)  # prepend dim → rank+1
            rows.append({
                "Test Item": f"test_neg_{pname}_invalid_rank",
                "Covered Params": f"{primary_name}, {pname}" if pname != primary_name else primary_name,
                "Data Type": main_dtype,
                "Dimensions": str(base_shape_2d),
                "Scenario": "Negative-shape rank",
                "Violated Field": pname,
                "Violation Construction": f"shape={tuple(wrong_shape)}",
                "Expected": "raises:PARAM_INVALID",
            })

    # --- 4. dim_range → below/above (None 边界表该侧无界, 跳过该侧) ---
    x_ci = _find_ci(primary_name)
    if x_ci and x_ci.dim_range:
        lo, hi = x_ci.dim_range
        if lo is not None and lo > 1:
            below_shape = tuple([2] * (lo - 1))  # 比下界少 1 维
            rows.append({
                "Test Item": "test_neg_dim_below_min",
                "Covered Params": primary_name,
                "Data Type": main_dtype,
                "Dimensions": str(base_shape_2d),
                "Scenario": "Negative-dim out of bounds",
                "Violated Field": primary_name,
                "Violation Construction": f"shape={below_shape}",
                "Expected": "raises:PARAM_INVALID",
            })
        if hi is not None:
            above_shape = tuple([2] * (hi + 1))
            rows.append({
                "Test Item": "test_neg_dim_above_max",
                "Covered Params": primary_name,
                "Data Type": main_dtype,
                "Dimensions": str(base_shape_2d),
                "Scenario": "Negative-dim out of bounds",
                "Violated Field": primary_name,
                "Violation Construction": f"shape={above_shape}",
                "Expected": "raises:PARAM_INVALID",
            })

    # --- 5. dst_type Attr → invalid value(显式优先级,对标官方 ACL_BF16) ---
    INVALID_OUTPUT_DTYPES = ["bfloat16", "float32", "int64", "int32"]
    for attr in dst_attrs:
        aname = attr.get("Param Name", "")
        if all_output_dtypes:
            invalid = None
            for cand in INVALID_OUTPUT_DTYPES:
                if cand not in all_output_dtypes:
                    invalid = cand
                    break
            if invalid:
                rows.append({
                    "Test Item": f"test_neg_{aname}_invalid",
                    "Covered Params": f"{primary_name}, {aname}",
                    "Data Type": main_dtype,
                    "Dimensions": str(base_shape_2d),
                    "Scenario": "Negative-dtype domain",
                    "Violated Field": aname,
                    "Violation Construction": f"dtype={invalid}",
                    "Expected": "raises:PARAM_INVALID",
                })

    # --- 6. mutex 非空 → conflict ---
    for p in all_tensors:
        pname = p.get("Param Name", "")
        ci = _find_ci(pname)
        if ci and ci.mutex:
            for a, b in ci.mutex:
                rows.append({
                    "Test Item": f"test_neg_{a}_{b}_conflict",
                    "Covered Params": f"{primary_name}, {a}, {b}",
                    "Data Type": main_dtype,
                    "Dimensions": str(base_shape_2d),
                    "Scenario": "Negative-param conflict",
                    "Violated Field": f"{a},{b}",
                    "Violation Construction": "conflict",
                    "Expected": "raises:PARAM_INVALID",
                })

    return rows


def check_required_scenarios(rows, classification, input_params, output_specs):
    """RC10: Verify required scenarios from CLASS_SCENARIOS[classification] are covered.
    Code-enforced acceptance of OPTIMIZATION-DESIGN "operator category required scenario full coverage".
    Returns self-check text lines + missing scenario list (for warnings aggregation).

    Matching: use ascii tokens from scenario label (dst_type/smooth_scales/group_index/INT4...)
    to search in rows text. Conditional scenarios (desc contains "if operator has/supports") —
    if probe token not found in input param names / output dtypes → mark N/A not applicable, not counted as missing."""
    scenarios = CLASS_SCENARIOS.get(classification, [])
    if not scenarios:
        return [], []
    row_text = " ".join(
        (r.get("Test Item", "") + " " + r.get("Covered Params", "") + " " + r.get("Scenario", "")).lower()
        for r in rows
    )
    param_text = " ".join(p.get("Param Name", "").lower() for p in (input_params or []))
    out_text = " ".join(o.get("Dtype Rule", "").lower() for o in (output_specs or []))
    applicable_text = param_text + " " + out_text

    lines = ["- Required scenario check (category=%s):" % classification]
    missing = []
    for label, desc in scenarios:
        tokens = [t.lower() for t in re.findall(r'[A-Za-z][A-Za-z0-9_]+', label)]
        tokens = [t for t in tokens if t not in ("moe", "int", "moe")]  # filter generic tokens
        conditional = ("if operator has" in desc) or ("if operator supports" in desc)
        covered = any(t in row_text for t in tokens) if tokens else False
        applicable = (not conditional) or any(t in applicable_text for t in tokens)
        if not applicable:
            lines.append(f"    - {label}: ⊘ N/A (operator lacks this param/capability)")
        elif covered:
            lines.append(f"    - {label}: ✅ covered")
        else:
            lines.append(f"    - {label}: ⚠️ missing")
            missing.append(label)
    return lines, missing


def format_negative_table(rows):
    """Format negative coverage matrix as markdown table (7 fields)."""
    if not rows:
        return ""
    lines = [
        "",
        "### Negative Coverage Matrix",
        "",
        "> Auto-generated by generate_negative_cases(), derived from constraint signals for illegal input combinations.",
        "",
        "| Test Item | Covered Params | Data Type | Dimensions | Scenario | Violated Field | Violation Construction | Expected |",
        "|-----------|----------------|-----------|------------|----------|----------------|------------------------|----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['Test Item']} | {r['Covered Params']} | {r['Data Type']} | {r['Dimensions']} | {r['Scenario']} | "
            f"{r['Violated Field']} | {r['Violation Construction']} | {r['Expected']} |"
        )
    lines.append("")
    neg_categories = set(r["Scenario"] for r in rows)
    lines.append(f"**Negative category coverage**: {len(neg_categories)} categories, {len(rows)} rows total")
    lines.append("")
    return "\n".join(lines)


def format_coverage_table(rows, input_params=None, output_specs=None, classification="other", degraded=False):
    lines = [
        "### Coverage Matrix",
        "",
        "> Auto-generated by scaffold_coverage.py, add/delete applied per Constraints column markers.",
        "",
        "| Test Item | Covered Params | Data Type | Dimensions | Scenario |",
        "|-----------|----------------|-----------|------------|----------|",
    ]
    for r in rows:
        lines.append(f"| {r['Test Item']} | {r['Covered Params']} | {r['Data Type']} | {r['Dimensions']} | {r['Scenario']} |")

    # summary
    lines.append("")
    lines.append(f"**Auto-generated stats**: {len(rows)} test items.")
    lines.append("")

    # ── 规则合规自检 (RC8): 按真实展开行数实算, 漏展开打 ⚠️ ──
    lines.append("**Rule compliance self-check**:")
    warnings = []

    # 1. 基本功能: 数据类型列不得残留分隔符(坍缩信号)
    basic_rows = [r for r in rows if r.get("Scenario") == "Basic functionality"]
    collapsed = [r for r in basic_rows if re.search(r'[;,、；]', r.get("Data Type", ""))]
    if collapsed:
        warnings.append(f"Basic functionality dtype not split (possible separator collapse): {[r['Data Type'] for r in collapsed]}")
    lines.append(f"- Basic functionality dtype coverage: {len(basic_rows)} rows" + (" ⚠️" if collapsed else " ✅"))

    # 2. dst_type/输出 dtype 全值: 实际展开行数 vs 期望 dtype 数
    if output_specs is not None:
        expected = set()
        for o in output_specs:
            expected.update(extract_dtype_tokens(o.get("Dtype Rule", "")))
        expected.discard("float32")  # scale dtype 不可控
        dst_rows = [r for r in rows if r.get("Test Item", "").startswith(("test_dst_type_", "test_output_dtype_"))]
        ok = len(dst_rows) >= len(expected) and len(expected) > 0
        mark = " ✅" if ok else " ⚠️ 漏展开"
        lines.append(f"- Output dtype full coverage: expected {len(expected)} values {sorted(expected)}, expanded {len(dst_rows)} rows{mark}")
        if not ok and expected:
            warnings.append(f"Output dtype expected {sorted(expected)}, only {len(dst_rows)} rows expanded")
    else:
        lines.append(f"- Output dtype full coverage: see 'test_dst_type_*' items above")

    # 3. 可选 Tensor 参数: 每个至少 1 行覆盖
    if input_params is not None:
        opt_tensors = [p.get("Param Name", "") for p in input_params
                       if p.get("Required/Optional") == "Optional" and is_tensor_type(p.get("Type"))
                       and "dst" not in p.get("Param Name", "").lower()]
        all_cov = " ".join(r.get("Covered Params", "") for r in rows)
        missing = [n for n in opt_tensors if n and n not in all_cov]
        mark = " ✅" if not missing else f" ⚠️ missing coverage: {missing}"
        lines.append(f"- Optional param coverage: {len(opt_tensors)} optional Tensors{mark}")
        if missing:
            warnings.append(f"Optional param zero coverage: {missing}")
    else:
        lines.append(f"- Optional param coverage: checking passed params across {len(rows)} items")

    lines.append(f"- Interaction / not-applicable: Step 3 handles row add/delete and expansion per `Interaction: A×B` / `Not applicable: param` anchors in Constraints column")

    # 4. Required scenario check (RC10): enforce category-specific scenarios, mark missing with ⚠️
    scen_lines, scen_missing = check_required_scenarios(rows, classification, input_params, output_specs)
    lines.extend(scen_lines)
    if scen_missing:
        warnings.append(f"Required scenarios missing ({classification}): {scen_missing}")

    # 5. Negative degradation banner (RC7+): .constraint_info.json missing → incomplete negative matrix, visible in file
    if degraded:
        warnings.append("Negative degradation: .constraint_info.json missing, only dtype+dim negative recoverable; "
                        "rank/shape_link/mutex negative lost → re-run scaffold_init to generate .constraint_info.json then re-run this script")

    if warnings:
        lines.append("")
        lines.append("**⚠️ Self-check warnings (step 3 review or upstream format fix needed)**:")
        for w in warnings:
            lines.append(f"  - {w}")
    lines.append("")

    return "\n".join(lines)


def patch_checklist(checklist_path, coverage_text, dry_run=False):
    cp = Path(checklist_path)
    if not cp.exists():
        print(f"[ERROR] CHECKLIST.md not found: {checklist_path}")
        return False
    current = cp.read_text(encoding="utf-8")
    if "## UT Generation Info" not in current:
        print("[ERROR] Missing '## UT Generation Info', run scaffold_pre.py first")
        return False

    # Replace '### Coverage Matrix' content up to next '###' or '##'
    marker = "### Coverage Matrix"
    pattern = re.escape(marker) + r'.*?(?=\n###|\n## |\Z)'
    if marker not in current:
        print("[WARN] '### Coverage Matrix' not found, appending")
        new_text = current.rstrip() + "\n\n" + coverage_text + "\n"
    else:
        existing = re.search(pattern, current, re.DOTALL)
        new_text = current[:existing.start()] + coverage_text.rstrip() + "\n" + current[existing.end():]

    if dry_run:
        print("=" * 60)
        print("[DRY RUN] Coverage matrix:")
        print("=" * 60)
        print(coverage_text)
        print("=" * 60)
        return True

    cp.write_text(new_text, encoding="utf-8")
    print(f"[OK] Coverage matrix updated → {checklist_path}")
    return True


# ─── main ────────────────────────────────────────────────────────

def rebuild_constraint_infos(input_params, output_specs):
    """RC7 降级路径: .constraint_info.json 缺失时, 从 CHECKLIST 输入/输出表
    重建最小 ConstraintInfo, 使 dim/dtype 负向用例仍能生成(不是静默丢覆盖)。
    重建项: valid_dtypes(dtype列) + dim_range(约束列 'N-M 维')。
    缺项: shape_link/rank/mutex(需 .md 结构化抽取, 无法从摘要表恢复)。"""
    cis = []
    for p in input_params:
        if not is_tensor_type(p.get("Type")):
            continue
        name = p.get("Param Name", "")
        if not name:
            continue
        ci = ConstraintInfo(param=name, optional=(p.get("Required/Optional") == "Optional"))
        ci.valid_dtypes = parse_dtype_list_from_cell(p.get("Supported dtypes", ""))
        m = re.search(r'(\d+)\s*-\s*(\d+)\s*维?', p.get("Constraints", ""))
        if m:
            ci.dim_range = (int(m.group(1)), int(m.group(2)))
        cis.append(ci)
    return cis


def main():
    p = argparse.ArgumentParser(
        description="Coverage matrix scaffold — generate semi-finished coverage matrix from CHECKLIST structured param tables"
    )
    p.add_argument("--checklist", default=None, help="Path to CHECKLIST.md")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    search_dir = Path.cwd()
    checklist_path = args.checklist or str(search_dir / "CHECKLIST.md")
    text = read_checklist(checklist_path)
    if not text:
        print(f"[ERROR] CHECKLIST.md not found: {checklist_path}")
        return 1

    ut_section = parse_section(text, "## UT Generation Info")
    if not ut_section:
        print("[ERROR] Missing '## UT Generation Info' section")
        return 1

    # Parse operator basic info
    basic_info = parse_section(ut_section, "### Operator Basic Info")
    basic = parse_kv_list(basic_info)
    raw_classification = basic.get("Operator Classification", "other")
    classification, class_warn = normalize_classification(raw_classification)
    operator_name = basic.get("Operator Name", "unknown")

    print(f"[INFO] Operator: {operator_name}")
    if class_warn:
        print(f"[WARN] {class_warn}")
    print(f"[INFO] Classification: {classification}" + (f" (raw: {raw_classification})" if classification != raw_classification.strip().lower() else ""))

    # Parse input/output spec
    io_section = parse_section(ut_section, "### Input/Output Spec")
    input_params = parse_input_params(io_section)
    output_specs = parse_output_specs(io_section)
    constraints = extract_constraints(io_section)

    # Load structured constraints from .constraint_info.json (phase 1+2 output)
    constraint_info_json = Path(checklist_path).parent / ".constraint_info.json"
    constraint_infos = []
    degraded = False  # RC7+: negative degradation flag, written into CHECKLIST self-check area (not just stdout)
    if constraint_info_json.exists():
        try:
            raw = json.loads(constraint_info_json.read_text(encoding="utf-8"))
            constraint_infos = [ConstraintInfo(**d) for d in raw]
            print(f"[INFO] constraint_info: {len(constraint_infos)} params loaded")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[WARN] .constraint_info.json parse error: {e}")
    # RC7: missing/parse failure → loud degradation, rebuild minimal constraints from input table (don't silently drop negative coverage)
    if not constraint_infos:
        degraded = True
        constraint_infos = rebuild_constraint_infos(input_params, output_specs)
        print(f"[WARN] .constraint_info.json missing/empty → rebuilt {len(constraint_infos)} minimal constraints "
              f"from input table dtype+Constraints columns (dtype negative preserved; dim negative only recoverable "
              f"if 'N-M dim' filled in Constraints column; shape_link/rank/mutex negative unrecoverable, "
              f"re-run scaffold_init to generate .constraint_info.json)")

    print(f"[INFO] Input params: {len(input_params)}")
    for ip in input_params:
        print(f"        {ip.get('Param Name', '?')}: {ip.get('Required/Optional', '?')}, dtype={ip.get('Supported dtypes', '?')}")
    print(f"[INFO] Output specs: {len(output_specs)}")
    for op in output_specs:
        print(f"        {op.get('Output Name', '?')}: dtype={op.get('Dtype Rule', '?')}")
    if constraints:
        print(f"[INFO] Constraint notes: {len(constraints)} entries")

    if not input_params:
        print("[WARN] Input param table is empty, generating minimal coverage matrix.")

    # 生成
    rows = generate_coverage(input_params, output_specs, classification, constraints, constraint_infos)
    coverage_text = format_coverage_table(rows, input_params, output_specs, classification, degraded)

    # 负向用例派生(从 .constraint_info.json)
    shapes = CLASS_SHAPES.get(classification, CLASS_SHAPES["other"])
    default_dtype = next(iter(
        set(parse_dtype_list_from_cell(input_params[0].get("Supported dtypes", "")))
    )) if input_params else "float16"
    dim_info = None
    for c in constraints:
        m = re.search(r'(\d+\s*-\s*\d+)', c)
        if m:
            dim_info = m.group(1)
            break
    if not dim_info:
        for p in input_params:
            m = re.search(r'(\d+)\s*-\s*(\d+)\s*维?', p.get("Constraints", ""))
            if m:
                dim_info = f"{m.group(1)}-{m.group(2)}"
                break
    neg_rows = generate_negative_cases(input_params, output_specs, constraint_infos,
                                       classification, default_dtype, shapes, dim_info)
    neg_text = format_negative_table(neg_rows)

    print(f"[INFO] Generated {len(rows)} positive test items + {len(neg_rows)} negative test items")

    # 预置空 Add/Delete Log 段(带格式注释), 给 step3 model 有坑可填、降低漏写率
    add_delete_log_placeholder = (
        "\n### Finalization Add/Delete Log\n\n"
        "> 格式契约: `Delete <exact_method_name>: <reason>` / `Add <exact_method_name>: <reason>`\n"
        "> `<exact_method_name>` 必须与 Coverage Matrix 的 Test Item 列值**逐字一致**, 禁自然语言概述。\n"
        "> reconcile.py 依赖此格式做删除放行判定, 格式不符 → [BAD-LOG] 告警。\n"
        "\n"
        "<!-- 在此下方逐行记录增删, 一行一项 -->\n"
    )
    # Write (positive + negative + log placeholder)
    full_text = coverage_text + neg_text + add_delete_log_placeholder
    ok = patch_checklist(checklist_path, full_text, dry_run=args.dry_run)
    if ok:
        print()
        print("✅ scaffold_coverage complete.")
        print(f"   Coverage matrix: {len(rows)} test items, at least 1 per optional param, at least 1 per output dtype.")
        print("   Model only needs to supplement: shape adjustments, param combination decisions, boundary case refinement.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
