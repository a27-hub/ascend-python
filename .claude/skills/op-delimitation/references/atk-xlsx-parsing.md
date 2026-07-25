# ATK xlsx 中文表头解析模板（权威单一展开点）

> **入口 skill**：[`tools/op-atk-adapt/SKILL.md`](../tools/op-atk-adapt/SKILL.md)  
> orchestrator / analyze-* / report **只引用本节，不重复展开**。

## 读 xlsx：中文表头 + 中文单元格的解析模板（**权威单一展开点**）

> 本节是 ATK `report/*.xlsx` 的**单一权威源**。orchestrator / analyze-single-op / analyze-functional / precision / report 等下游 skill **只引用本节、不重复展开**。

### 给模型的备注（开干前先读这条）

- ATK 产出的 `report/*.xlsx` 是**真 xlsx**（OOXML 压缩 zip），不是 csv；表头、`运行结果` / `失败原因` / `精度详情` 等关键单元格**大量中文**。
- `cat` / `head` / `tail` / `less` / `grep` / `xlsx2csv` 这类**纯文本 / 字节流**工具读 xlsx 会得到二进制乱码或字符丢失，**实际不可用**——这不是"为规矩而禁",是物理事实。
- **推荐做法**：在用户机器上用 python（`openpyxl` 或 `pandas + openpyxl`）解析，把判读必需字段（`case_id` / `运行结果` / `失败原因` 等）转写到 `运行记录.md` 与下游分析 md。
- 不强制使用某个具体库；只要是"能正确解 xlsx + 能输出 UTF-8 中文"的 python 工具都可（`openpyxl` / `pandas` / `polars.read_excel` / `xlrd`<sup>(xlsx 须 ≥ 2.0)</sup> 等任选）。

### 最简一行式（先扫一眼整张表）

适合刚拿到 xlsx 路径、想先看"表头 + 总行数 + `运行结果` 分布"的场景，**不需要写脚本文件**：

```bash
python -c "
from openpyxl import load_workbook
import sys
ws = load_workbook(sys.argv[1], read_only=True, data_only=True).active
h = [c.value for c in next(ws.iter_rows(max_row=1))]
print('表头:', h)
i = h.index('运行结果')
c = {}
for r in ws.iter_rows(min_row=2, values_only=True):
    c[str(r[i])] = c.get(str(r[i]), 0) + 1
print('运行结果分布:', c)
" "<xlsx 路径>"
```

后续如需把 FAILED 行 / `*精度通过=false` 行原文转写出来，再切到下面「解析模板 A / B」的完整版本。

### 何时启用 python 解析

- xlsx 路径**来自用户**（绝对路径 / 远端真实路径），不是从 DTS URL 或截图推断；
- 用户机器上有 python（远端 NPU 机器一般有 conda 环境；本机 Windows 也常有 `python.exe`）；
- 解析目的**有限**：拉表头、统计 `运行结果` 分布、筛 FAILED 行 / SUCCESS+`*精度通过`=false 行、列出失败 `case_id`、贴 `失败原因` 原文 —— **不**整表 dump，**不**回贴到聊天里的字段超出"判读必需"。

> 仍**禁止**：对 DTS 页 URL / heaven URL / 内网 wiki / CodeHub issue / 截图做抓取或 OCR（属于 [hard-constraints · 统一口径](../knowledge/hard-constraints.md#统一口径--外部资源获取方式) 范围）。

### 解析模板 A — `openpyxl` 直读（推荐；NPU 机器 conda 环境通常自带）

把下段拷给用户、或助手通过 shell 工具直接在用户机器上跑：

```python
# 用法：python read_atk_xlsx.py <xlsx 路径> [sheet 名/索引，默认第一张]
# 输出：表头一次 + 关键列分布 + FAILED / SUCCESS+精度未通过 行原文（按需收口）
import sys, os, json
from openpyxl import load_workbook

xlsx = sys.argv[1]
sheet_arg = sys.argv[2] if len(sys.argv) > 2 else None

wb = load_workbook(xlsx, read_only=True, data_only=True)
ws = wb[sheet_arg] if sheet_arg else wb[wb.sheetnames[0]]

rows = list(ws.iter_rows(values_only=True))
header = [str(c) if c is not None else "" for c in rows[0]]
data = rows[1:]

def col(name):
    return header.index(name) if name in header else None

i_result = col("运行结果")
i_reason = col("失败原因")
i_id = col("编号") if "编号" in header else col("case_id")
acc_cols = [(idx, h) for idx, h in enumerate(header) if h.endswith("精度通过")]

print("== 表头 ==")
print(header)
print(f"\n== sheet 名 ==  {ws.title}    总行数（不含表头）= {len(data)}")
print("\n== 运行结果 分布 ==")
counter = {}
for r in data:
    v = r[i_result] if i_result is not None else None
    counter[str(v)] = counter.get(str(v), 0) + 1
print(json.dumps(counter, ensure_ascii=False, indent=2))

def is_fail(r):
    return i_result is not None and str(r[i_result]).strip().upper() == "FAILED"

def is_acc_fail(r):
    if i_result is None or str(r[i_result]).strip().upper() != "SUCCESS":
        return False
    for ai, _ in acc_cols:
        v = str(r[ai]).strip().lower()
        if v in ("false", "failed", "0", "no"):
            return True
    return False

print("\n== FAILED 行（前 50） ==")
shown = 0
for r in data:
    if is_fail(r) and shown < 50:
        cid = r[i_id] if i_id is not None else ""
        reason = r[i_reason] if i_reason is not None else ""
        print(f"id={cid} | 失败原因={reason}")
        shown += 1

print("\n== SUCCESS + 任一 *精度通过=false 的行（前 50） ==")
shown = 0
for r in data:
    if is_acc_fail(r) and shown < 50:
        cid = r[i_id] if i_id is not None else ""
        acc_state = {h: r[ai] for ai, h in acc_cols}
        print(f"id={cid} | 精度列={acc_state}")
        shown += 1
```

> 命中字段名变体（如 `运行结果` / `测试结果`、`编号` / `case_id`）时，**让用户确认一次列名映射**再锁定，不臆造列名。

### 解析模板 B — `pandas + openpyxl`（看表更顺手时用）

```python
import sys
import pandas as pd

xlsx = sys.argv[1]
df = pd.read_excel(xlsx, sheet_name=0, engine="openpyxl")
print("== 表头 ==")
print(list(df.columns))
print("\n== 运行结果 分布 ==")
print(df["运行结果"].value_counts(dropna=False).to_string())

acc_cols = [c for c in df.columns if c.endswith("精度通过")]

print("\n== FAILED 行 ==")
print(df[df["运行结果"].astype(str).str.upper() == "FAILED"][
    ["编号" if "编号" in df.columns else "case_id", "运行结果", "失败原因"]
].head(50).to_string(index=False))

print("\n== SUCCESS + 任一 *精度通过=false ==")
mask_succ = df["运行结果"].astype(str).str.upper() == "SUCCESS"
mask_acc = False
for c in acc_cols:
    mask_acc = mask_acc | df[c].astype(str).str.lower().isin(["false", "failed", "0", "no"])
print(df[mask_succ & mask_acc][
    ["编号" if "编号" in df.columns else "case_id"] + acc_cols
].head(50).to_string(index=False))
```

### 中文打印的注意点（**Windows / PowerShell 上常见**）

- 在 cmd.exe / PowerShell 上 stdout 默认 GBK，中文会乱码；跑模板前先 `set PYTHONIOENCODING=utf-8`（cmd）或 `$env:PYTHONIOENCODING="utf-8"`（PowerShell），或在脚本顶端加 `sys.stdout.reconfigure(encoding="utf-8")`；
- 用 `print(json.dumps(..., ensure_ascii=False))` 保留中文原文；
- 在远端 Linux 上一般是 utf-8 locale，可省此步；若 `LC_ALL` 为 `C` / `POSIX`，临时 `export LC_ALL=C.UTF-8`。

### 用解析结果做证据的规矩

- 解析后**只把判读必需字段**（`case_id` / `运行结果` / `失败原因` 摘要 / 命中的 `*精度通过` 列名）转写到 `运行记录.md` 与下游分析 md；**不**整表 dump 进聊天 / 仓库；
- 若用户随后**用截图、终端粘贴等"非 xlsx"方式**给出同一行原文，以**用户粘贴的纯文本**为准；解析脚本输出仅作交叉验证；
- 任何"运行结果分布"统计写进 md 时，要带上 xlsx 真实路径 + sheet 名 + 总行数，便于复核。
