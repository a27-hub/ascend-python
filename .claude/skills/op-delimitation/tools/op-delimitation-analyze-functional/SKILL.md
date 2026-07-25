---
name: op-delimitation-analyze-functional
description: >-
  Use for ATK-only functional or environment triage after flow-atk identifies
  FAILED xlsx rows, including aclnn errors, environment failures, A2/A3 lookup
  prompts, OOM/timeout evidence, and retest routing.
---

# 3.2 — 功能 / 环境问题确认（ATK-only）

> **分层阅读**：先做 [`skills/op-delimitation-flow-atk.md`](../../skills/op-delimitation-flow-atk.md) 中的 **两遍主流程判定**；**同一卡点连续两轮仍无法在 3.1/3.2/4 间落地** → 必须 **Read 本 SKILL 全文**。  
>
> **前提**：orchestrator Gate 0 已通过；输入为 **ATK 全量跑**得到的 `report/*.xlsx`。
>
> **入口条件**：xlsx 中 **`运行结果` = `FAILED`** 的行。与"`运行结果`=`SUCCESS` 且某 `*精度通过` 列=false"的 **4 真精度** 路径互斥（见 [op-delimitation-flow-atk](../../skills/op-delimitation-flow-atk.md#atk-xlsx-decision) §一屏决策表）。

## 给模型的两条前置备注（开干前先读）

1. **读 xlsx → 走 python 解析**：ATK `report/*.xlsx` 的**表头与关键单元格是中文**（`运行结果` / `失败原因` / `编号` / `*精度通过` 等）；`cat` / `head` / `grep` / `xlsx2csv` 类纯文本/字节流工具读 xlsx 会得到二进制乱码或丢字（**物理上不可用**，不是禁令）。**筛 `运行结果=FAILED` 行、按 `失败原因` 分类的所有操作均走 python**（`openpyxl` 直读或 `pandas + openpyxl`，最简形式即 `python -c "from openpyxl import load_workbook; ..."` 一行式），完整模板见 [op-atk-adapt「读 xlsx 的解析模板」](../op-atk-adapt/SKILL.md#读-xlsx中文表头--中文单元格的解析模板权威单一展开点)，本 skill 不重复展开。
2. **用户粘贴的 ATK 命令是骨架，按 xlsx 真实证据可微调**：本 skill 内涉及"OOM/timeout 行 → 回 op-atk-adapt 调 `-mt`/`-to` 复跑"或"跨向 3.1 做 `-wl` 单跑"时，`-wl '[case_id,...]'` 的下标**应按当下 xlsx 的真实 FAILED `case_id` 重写**、`-bl` / `-mt` / `-to` / `--save_data` 按本次目标按需叠；但 `-c <case_json>` / `-n <yaml>` / `-p <python_script>` 路径、容器入口、`source <set_env.sh>`、工作目录、API 名**必须保持用户粘贴原文不动**。完整对照见 [op-atk-adapt「用户粘贴 ATK 命令的可调整范围」](../op-atk-adapt/SKILL.md#atk-参数可调整范围)。

## 输入（必须先与用户对齐一次）

- xlsx 路径（真实，非占位）。
- 表头含：**`运行结果`**、**`失败原因`**（常必需）、`编号`/`case_id`（用于单跑）。若列名变体，请用户映射后锁定。**筛行/取列**走 python（见上方前置备注 1）。

## 判定（按顺序）

1. **筛行**：`运行结果 == FAILED` 为**功能/环境候选**；若无 → 功能问题=false，交回编排器。
2. **读 `失败原因`**，按下表分类（**识别模式仅用于分类提示，不替代用户确认**）：

| 分类 | 典型关键词 / 模式 | 处理 |
|------|-------------------|------|
| **aclnn 类失败** | `aclnn` 名 + `failed` / `EZ*` 错误码 / `Operator execute failed` | 归为单算子/功能问题；可选 [3.1 single-op](../op-delimitation-analyze-single-op/SKILL.md) 做 `-wl` 单跑确认 → 5.end |
| **算子 dtype 不支持** | `EZ3002` / `Unsupported_Operator` / `data type DT_* is not supported` / `op type * is not found in this op store` / `build op model failed` | CANN kernel 缺该 dtype 白名单或适配层缺 cast → 进 flow-fix / 转 CANN，**非框架竞态**。**xlsx `失败原因` 常只记末端 `FileNotFoundError`/`accuracy dc failed`，根因在 atk.log 更早的 `EZ*`** —— 见 [末端症状回溯](#末端症状回溯) |
| **环境 / 驱动 / 依赖** | `CANN` / `ascend-toolkit` / `ImportError` / `libxxx.so not found` / `set_env` / `HCCL` / `RANK_TABLE_FILE` / `No module named torch_npu` | 环境问题；分析 → 5.end；**不要**当成 API 精度问题 |
| **用例脚本 / 配置** | `yaml` 解析错 / `case_json` 缺字段 / `FileNotFoundError` 指向用例输入 | 归为功能；引导用户提供命令 / 路径原文供复核。**注意**：`FileNotFoundError: input.bin not exists` 可能是**派生症状**（case 因算子不支持先崩溃 → ATK 清理产物 → input.bin 被删）→ 见 [末端症状回溯](#末端症状回溯) |
| **超时 / OOM** | `timeout` / `OOM` / `memory` | 先回 [op-atk-adapt](../op-atk-adapt/SKILL.md) 走证据驱动调优：OOM → `-mt` 从 `5` 开始向下；timeout → `-to` 从 60s 向上。重跑时沿用 `unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2`；再回本步判定是否仍归功能/环境 |
| **A2 / A3 已知问题** | 疑似与已知问题匹配 | **仅提示用户自搜**（见下节）；助手不代查、不编造命中 |

<a id="末端症状回溯"></a>

> **末端症状回溯（防误判）**：xlsx `失败原因` 列常只记 error 链**末端**（`FileNotFoundError: input.bin not exists` / `accuracy dc failed` / `run opp task failed`），根因级错误（`Unsupported_Operator(EZ3002)` / `build op model failed`）只在 `atk_output/.../log/atk.log` 的 `[ERROR]` traceback 里。对 `FileNotFoundError` 指向 `input.bin` 的行，**必须先 grep 同 case 在 atk.log 里更早的 `EZ*` / `Unsupported` / `build op model failed`**：
>
> ```bash
> grep -nE 'EZ[0-9]{4}|Unsupported|not supported|build op model failed|working operator name' atk_output/<CaseSet>_<ts>/log/atk.log
> ```
>
> - 有更早 op error → 归"算子 dtype 不支持 / aclnn 类失败"，**不是脚本配置**。
> - 真实因果链：算子不支持 → 崩溃 → ATK `cleanup case artifacts, reason=run_opp_task_failed` 正常清理（删 input.bin）→ 后续 `load_dataset` 报 `FileNotFoundError`。清理是**正常失败处理**，非并发竞态。

### ATK 错误速查表（先匹配再做深度分析）

> 以下为 ATK 运行中**可自修复**（不需要回 op-atk-adapt 重跑全量）的常见错误模式。命中任一条 → 按"动作"列修复后重跑当前命令，**不视为执行失败**。

| 错误关键字 / 模式 | 诊断 | 动作 |
|------------------|------|------|
| `KeyError: "自定义API'xxx'找不到或者import失败"` / `请检查用例yaml中的api_type字段` | `-p` 自定义 API 插件路径缺失 | 检查 issue 目录下 `.py` 文件（含 `@register("api_type")` 注册）→ 加 `-p <file.py>` 重跑。**不要**在未尝试加 `-p` 的情况下直接写"ATK 插件缺失 → early-stop"结论 |
| `TypeError: Argument 'case_list' has incorrect type (expected list, got str)` | `-bl`/`-wl` 格式被 shell 错误转义（常见 `\""[...]"\"`） | 改为 `-bl '[...]'` / `-wl '[...]'` 格式（单引号包裹 JSON 数组）。详见 [op-atk-adapt §`-bl`/`-wl` 格式自检](../op-atk-adapt/SKILL.md#blwl-格式自检跑前必做) |
| `sqlite3.OperationalError: disk I/O error` | Celery worker 的 SQLite 后端临时写入失败（非磁盘满） | ① `df -h` 确认磁盘空间充足；② 若有可用 xlsx 则直接复用不进重跑；③ 否则换 `atk_output/` 父目录重跑 |
| `IndexError: list index out of range`（`-wl` 单跑时） | `-wl` 格式错误（如 `'[[id1],[id2]]'` 多了一层嵌套列表） | 改为 `-wl '[id1,id2]'`（离散值）或 `-wl '[[start,end]]'`（连续区间）。**不是** case 不存在 |
| `Error: Invalid value for '--save_data': Ellipsis` | `--save_data` 当 flag 裸写，实际需要参数值 | 改为 `--save_data input --save_data output`（或简写 `-sd`） |

3. **必要时跨向 3.1** 做 `-wl` 单跑，补齐证据（是单个 case 还是一批 case 同模式失败）。

## A2 / A3 已知问题处理（硬约束）

- 本 Agent **不接入** A2/A3 表格。
- 需要对照时，**仅给用户**"建议检索词清单"：
  - case 名（`atk_output` 里的 `all_<CaseSet>_*`）、
  - ATK `case_json` 文件名、
  - API 名（如 `torch.Tensor.remainder`）、
  - `失败原因` 中截取的报错片段/错误码。
- 不得在 md/结论中写入"已在 A2/A3 找到 xxx"类字样，除非用户自己给出命中行原文。

## 外部链接写法

遵循 [hard-constraints · 统一口径](../../knowledge/hard-constraints.md#统一口径--外部资源获取方式)：**当前默认助手无权/不主动访问** wiki / CodeHub issue 等内网 URL。正文中的步骤要点、前置条件、关键命令**由用户粘贴原文**；URL 完整保留于 `运行记录.md` / `分析结论.md` 作存档。**严禁助手凭 URL 自造摘要。**

## 结单最小字段（进入 5.end 前）

向 [report](../../report/op-delimitation-report/SKILL.md) 传递：

- 分类（aclnn 失败 / 环境 / 脚本配置 / 其它）。
- 证据行列表：`case_id` + `失败原因` 摘要（1~2 行）+ 命中行所在 xlsx 路径。
- 若做了 3.1 单跑：附**单跑汇总表路径**与关键结论（单 vs 多算子 / 内存踩踏）。
- 对照 A2/A3 的**建议检索词**（若用户有此需要）。

## 原则

- **不造表**：`失败原因` 的文字以 xlsx 为准，不改写。
- **xlsx 读取一律走 python**（[op-atk-adapt 的解析模板](../op-atk-adapt/SKILL.md#读-xlsx中文表头--中文单元格的解析模板权威单一展开点)）；中文表头/单元格用 `cat`/`grep` 等文本工具直读会乱码丢字，是物理事实，不是禁令；不重复展开模板。
- **用户粘贴的 ATK 命令按需微调**：`-wl` / `-bl` / `-mt` / `-to` / `--save_data` 可改且**应**按当下 xlsx 真实 `case_id` 改写；`-c <case_json>` / `-n <yaml>` / `-p <python_script>` / 容器入口 / `set_env` / 工作目录 / API 名**保持原文不动**（[op-atk-adapt 可调整范围表](../op-atk-adapt/SKILL.md#atk-参数可调整范围)）。
- **不越界**：`运行结果=SUCCESS` 的行交给 [precision](../op-delimitation-analyze-precision/SKILL.md)。
- **不跳过基线**：涉及 ATK 重跑时沿用 [op-atk-adapt 基线](../op-atk-adapt/SKILL.md#atk-跑前环境基线权威本仓库唯一展开点)（`unset http_proxy https_proxy` + `export ATK_BIND_CPU_TYPE=2`）。
- **不盲调参数**：`-mt` / `-to` 只按 OOM/timeout 证据调整，且把取值与依据回填 `运行记录.md`。
- **early-stop** 时在分析结论 md 写清："功能类提前结束；依据：xlsx 第 N 行 `失败原因` = ..."。
- **末端症状不当根因**：`FileNotFoundError: input.bin not exists` / `accuracy dc failed` 可能是算子崩溃后的派生症状，必须回溯 atk.log 更早的 `EZ*`/`Unsupported`（见 [末端症状回溯](#末端症状回溯)）；禁止见 `FileNotFoundError` 就归"脚本配置"或"celery 竞态"。
- **确定性 False 按 dtype 分桶**：若 FAILED 行外还有 `确定性计算结果=False` 的 SUCCESS 行，必按 self dtype 交叉表（False/True 各 dtype 计数），分两路定性：① False 聚集某 dtype 且 atk.log 有 `EZ*`/`Unsupported` → 该 dtype 系统性不支持，非并发问题；② 支持型 dtype 仍 False（混合信号残余）→ 用 `--save_data` 固定输入比对 indices，区分 `max(dim)` ties-indices 规则差异 vs 真内存踩踏，**不得**与 dtype 不支持混为一谈。
