---
name: op-delimitation-report
description: >-
  Use when writing final op-delimitation ticket closure artifacts, including
  分析结论.md and 运行记录.md, for ATK script reproduction after evidence has
  been collected.
---

# 5.end — 结束与产物（ATK 通用）

> **前提**：orchestrator Gate 0 已通过；任务路由到 **分支 A · ATK**（主命令 `atk task` / `atk case`）。
>
> 目标：把前面步骤的证据一次性沉淀成**分析结论 md** + **运行记录 md**（以及 **`analyze/`** 下的中间分析产物；ATK 必要时 + **`single-run/`**），命名 / 路径规范、可追溯、无密码。
>
> **外部资源（DTS / heaven / wiki / CodeHub issue / A5 xlsx 等）获取方式**：统一遵循 [hard-constraints · 统一口径](../../knowledge/hard-constraints.md#统一口径--外部资源获取方式) —— **URL 仅存档、用户粘贴纯文本为唯一正文来源**。本 skill 的每个 md 模板均按该口径组织「证据」段与「外部参考」段。

## 产出位置（硬约束）

1. **默认**：当前工作目录或用户指定的**pta 目录（{opName}_pta_atk/）**下**新建子目录**存放本次产出。
2. **广根目录风险**：若用户要求写到 `home/`、`data/`、`root/` 等根级目录 → **先提示**数据散落/权限/误删风险并**明确询问**最终位置，获得确认后再写。
3. **命名规则**：目录 / 文件名**必须包含 opName + 日期**（示例 `npu_swiglu_group_quant_backward_20260421`；分隔符与用户确认一次后固定）。

## 目录层级建议（分支通用骨架）

```text
{opName}_pta_atk/
├── 分析结论.md
├── 运行记录.md
├── 运行日志/                ← tee / 重定向 log
│   └── atk_full_<ts>.log        (分支 A · ATK)
├── analyze/                 ← **分析中间结果**（草稿表、分段 grep/trace、比对笔记、未定稿摘录；不进终稿 `分析结论.md` 的可先放此处并在终稿交叉引用文件名）
├── single-run/              ← 仅分支 A：3.1 / 4.1 单跑产物与汇总表
│   ├── 单跑汇总表.csv
│   └── case_<id>/           ← 可选：每个 case 的 input/output 素材
└── README.md                ← 可选：目录导览
```

说明：

- **`analyze/`**：在pta 目录子目录 **`{opName}_{日期}/analyze/`** 下持久化「**分析的中间结果**」——包括但不限于：grep/`readelf`/`nm`/`objdump`/栈片段采集、两轮重试的差异对比表草案、未写入终稿的结论草稿、从大日志截出的可读切片（**仍可含敏感时请用户自检**）。命名建议带阶段或日期前缀（如 `03_grep_observer.txt`、`match_round2.md`），终稿 **`分析结论.md`** 正文只摘要结论，**可追溯路径**写到 `analyze/…`。**不要**把整个 `analyze/` 当终稿复述进聊天刷屏；结单自检要求 **目录存在且关键中间步骤可追溯**（见下）。

## 目标文档（通用骨架 + 分支特定段）

### 1. `分析结论.md`

必含（共用）：

- **摘要**：**分支（A）** + 分类 + 是否 early-stop + 关键结论一句话。
- **证据**：见下方各分支模板。
- **外部参考（含摘要）**：相关 wiki / issue 的**要点** + URL。
- **待办**：A2/A3 需用户自查时给"建议检索词"。

### 2. `运行记录.md`

必含（共用）：

- op-atk 输入：CANN ATK 用例目录 / op-plugin 仓根 / opName / api_type；
- op-atk 4 步执行记录：① auto_adapt.py 输出（opName / 接口原型 / @register api_type）② atk case 生成 result 路径（模式 B/3file 下为「跳过（预生成 case.json）」，须记录 case.json 路径）③ executor.py npu 分支改写摘要 ④ atk task 命令（`-c` 按模式：A=`result/<api_type>.json`；B=传入的 case.json）+ tee 路径 + xlsx 路径（按 `atk_output/all_<api_type>_<ts>/` 拼）；
- 实际在目标机执行的命令块（含 tee 规范）；
- 生成物真实路径列表（含 **`analyze/`** 下各中间文件名，若有）；
- 关键环境变量（按分支适配：`TORCH_NPU_LOGS=op_plugin`、`ASCEND_GLOBAL_LOG_LEVEL`、`ASCEND_WORK_PATH` 等）；
- 环境版本：默认本地 `local_torch_npu_dir` / `local_torch_npu_version` + `op_plugin_repo_root` / `op_plugin_git_commit`；仅当 op-plugin 仓 + NPU 同在远端时改记 `remote_torch_npu_dir` / `remote_torch_npu_version`（来自 [manual-connect](../../tools/manual-connect-npu-server/SKILL.md)）。

## ATK 模板分节（分支 A）

### 分析结论.md — ATK 模板

```markdown
# 分析结论 — {opName} — {日期}（分支 A · ATK）

## 摘要
- 分支：A · ATK
- op-atk 适配路径：{opName}_pta_atk/（op-plugin 仓根下）
- 分类：单算子 / 功能 / 精度 / 环境 / 混合
- Early-stop 点：例如"3.1 单跑锁定为单算子 aclnnXxx"
- 结论：一句话

## 证据
- xlsx：<真实路径>；命中行（示例）
  | case_id | 运行结果 | 关键列 | 摘要 |
- 关键日志：<tee 日志路径> + 锚点片段（`save result excel file:` / `acc_pass_result:`）
- aclnn 统计：<条数 + 列表>（若涉及 3.1.1）
- 单跑汇总：<single-run/单跑汇总表.csv>

## 外部参考（正文含摘要）
- Dump 指南要点：…（URL）
- 内存踩踏要点（ATK#655）：…（URL）

## 待办
- A2/A3：请用户自行在已知问题表用关键词检索；建议检索词：…
```

### 运行记录.md — ATK 模板

````markdown
# 运行记录 — {opName} — {日期}（分支 A · ATK）

## op-atk 输入
- CANN ATK 用例目录：
- op-plugin 仓根：
- opName / api_type：

## op-atk 4 步执行记录
① auto_adapt.py 输出：opName = ` `；接口原型 = ` `；@register api_type = ` `
② atk case 生成：`{opName}_pta_atk/result/<atk>.json`（模式 B/3file：跳过——预生成 case.json，记录其路径）
③ executor.py npu 分支改写摘要：
④ atk task 命令（`-c` 按模式：A=`result/<atk>.json`；B=传入的 case.json）+ tee 路径 + xlsx 路径：

## 在目标机执行

```bash
# 进入容器 / source 环境 / cd 工作目录 / atk task 全量 或 -wl 单跑
```

## 生成文件与目录
- atk_output/…/report/*.xlsx
- 运行日志/atk_full_*.log
- single-run/单跑汇总表.csv
````

## 分支 C 模板分节（修复子流程 · flow-fix）

> 当**适配层判定通过**（根因落在 op-plugin 适配层）、走过 [flow-fix](../../skills/op-delimitation-flow-fix.md) 后适用。产物落在 `{opName}_pta_atk/`：
>
> **不进分支 C 的分流**（flow-fix 入口闸判定后）：根因为 **aclnn/CANN 底层** → 结单标「转 CANN 团队」；根因为 **精度** → 结单标「精度，超出 pta 适配修复范围」。二者走常规结单（标注结论类型），不开 handoff/回归报告。
>
> - `handoff/根因分析与修复交接.md`（字段见 [flow-fix §阶段 3](../../skills/op-delimitation-flow-fix.md)；op-delimitation 写定界 + op-adaptation 回填修复）
> - `回归报告.md`（格式见 [flow-fix §阶段 5](../../skills/op-delimitation-flow-fix.md)；原始 atk task 命令重跑 + 近邻 + 修复红线自检）
> - `运行日志/regression_full_*.log`

### 分析结论.md — 分支 C 追加段

```markdown
## 修复子流程（flow-fix · 分支 C）
- 定界标签（可修复行）+ 进 flow-fix 判据
- 边界切换：boundary=READONLY_RELEASED / env=LOCAL_DEV_NPU / 切换时刻
- 修复：op-adaptation(MODIFY) 改动算子 + files_modified + CHECKLIST/summary 路径
- 回归结论：PASS（已修复）/ FAIL-熔断（未完全修复，≤3 轮）+ 回归报告路径
- 修复红线自检：5 项全过（见 [flow-fix §阶段 5](../../skills/op-delimitation-flow-fix.md)）
```

## 结单前自检清单（按分支勾齐；缺一项不宣布结单）

**通用**：

- [ ] 路径均为**用户已确认的真实路径**；无虚构盘符 / 占位。
- [ ] 文件名含 **opName + 日期**。
- [ ] 未写入**真实密码 / 私钥内容 / heaven token**；远端接入方式见 [manual-connect](../../tools/manual-connect-npu-server/SKILL.md)。
- [ ] **环境版本可追溯**：默认本地时 `local_torch_npu_dir` / `local_torch_npu_version` + `op_plugin_repo_root` + `op_plugin_git_commit` 已记录；仅当 op-plugin 仓 + NPU 同在远端时改记 `remote_torch_npu_dir` / `remote_torch_npu_version`（若应记项都未拿到，根本没到结单条件）。
- [ ] 外部 wiki / issue / DTS / heaven / CodeHub 等 URL：**正文摘要均来自用户粘贴**；URL 完整保留于「外部参考」段作存档。**无任何「凭 URL 自动归纳」的内容**（见 [hard-constraints · 统一口径](../../knowledge/hard-constraints.md#统一口径--外部资源获取方式)）。
- [ ] A2/A3：只写"请用户自查"与**建议检索词**，不写命中结论。
- [ ] **`analyze/`**（与任务同级的 `{opName}_{日期}/analyze/`）：已创建目录；本轮产生的 grep/草稿/未定稿摘录等**中间分析**均已落盘在该目录下，且在 **`运行记录.md`** 或 **`分析结论.md`** 有路径引用。**若本轮无此类落盘产物**，在 **`运行记录.md`** 写明「analyze/ 无中间落盘」。终稿不得在聊天里替代 `analyze/` 的可追溯存档。
- [ ] **设备环境快照**：`分析结论.md` 或 `运行记录.md` 含 [`templates/device-env-snapshot.md`](../../templates/device-env-snapshot.md) 块（CANN/torch/torch_npu/驱动/NPU + `op_plugin_repo_root`/`op_plugin_git_commit`；缺项标 `[未测]`）。
- [ ] **数值证据**（若走过 UT / 确定性 / Dump）：含 **max diff / allclose / 50 次确定性统计** 具体数值，见 [`templates/numeric-evidence.md`](../../templates/numeric-evidence.md)；禁止仅写「差异大/小」。

**分支 A · ATK 追加**：

- [ ] **op-atk 4 步执行记录已落 `运行记录.md`**：① auto_adapt.py 输出（opName / 接口原型 / @register api_type）② atk case 生成 result 路径（模式 B/3file：标「N/A — 预生成模式跳过 atk case」，并校验 case.json 路径已记录）③ executor.py npu 分支改写摘要 ④ atk task 命令 + tee 路径 + xlsx 路径（按 `atk_output/all_<api_type>_<ts>/` 拼）。
- [ ] xlsx 分类未混淆 `FAILED` 与 `SUCCESS+*精度通过=false`。
- [ ] 若涉及 3.1：`single-run/单跑汇总表` 已存在，列至少含 `case_id / atk task 命令 / 单跑结果`。
- [ ] 若涉及 4.1：已说明走到哪一步（复用 3.1 / dump / UT / 内存踩踏），并附素材路径。

**分支 C · 修复子流程追加**（仅当**适配层判定通过**、走过 [flow-fix](../../skills/op-delimitation-flow-fix.md)）：

- [ ] `handoff/根因分析与修复交接.md` 已产出且 op-adaptation 回填齐全（`files_modified` / `build_status` / `test_status`）。
- [ ] `回归报告.md` 已产出：**原始 atk task 命令**（来自 op-atk-adapt 第 4 步）重跑 + 数值对比 + 近邻（若触发）+ [flow-fix §阶段 5](../../skills/op-delimitation-flow-fix.md) 5 项红线全过。
- [ ] 边界切换已记录（`boundary` + `env` + 时刻）；远端全程保持只读。
- [ ] 回归结论与循环轮次如实记录（PASS / 熔断）。

## 证据矩阵（分析结论 md 中建议插入；按分支适配）

**分支 A · ATK**：

| 分类命中 | xlsx 行证据 | 日志证据 | 单跑证据 | 外部参考 |
|----------|-------------|----------|----------|----------|
| 单算子 | `运行结果` / `失败原因` / `编号` | `TORCH_NPU_LOGS=op_plugin` 中 aclnn 列表 | 单跑汇总表（aclnn 条数列） | GitCode aclnn 映射（参考） |
| 功能/环境 | `运行结果=FAILED` + `失败原因` 分类 | tee 日志中报错帧 | 可选 `-wl` 单跑 | A2/A3 用户自查建议词 |
| 真精度 | `运行结果=SUCCESS` + `*精度通过=false` + `精度详情` | `acc_pass_result:Failed` / csv_report 表 | UT 脚本输出 / `output_grad_0.pt` 等 | dump wiki 要点 + URL |
| 内存踩踏 | 批跑多行 FAIL | 批跑 vs 单跑差异 | 单跑 PASS | ATK#655 要点 + URL |

## 原则

- **真实路径**优先，**禁止**虚构盘符或占位路径冒充已执行。
- **不在 md / skill 中保存密码或密钥内容**。
- **不代写** A2/A3 命中结论；只给检索词。
- **A5 表只筛本次任务对应行**；等价表代替时显式写清文件名来源。
