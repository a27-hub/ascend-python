---
name: op-delimitation-analyze-single-op
description: >-
  Use for ATK-only single-op defining after op-atk-adapt or flow-atk routing,
  including failed case_id enumeration from xlsx, targeted atk -wl reruns,
  aclnn forward/backward counting, memory-stomp checks, and summary tables.
---

# 3.1 / 3.1.1 — 单算子确认（ATK-only）

> **分层阅读**：[`skills/op-delimitation-flow-atk.md`](../../skills/op-delimitation-flow-atk.md) 已定 **两遍** 后才进入本节全文；本节负责 `-wl`/aclnn/内存踩踏细分。  
>
> **前提**：orchestrator Gate 0 已通过；有 ATK 全量跑产出的 `atk_output/.../report/*.xlsx`。
>
> 目的：在**已有失败或待单跑**的 `case_id` 上，用 `atk task -wl` + `TORCH_NPU_LOGS=op_plugin` 日志定界 **aclnn 条数 / 是否单算子 / 是否内存踩踏 / 正反向**。**不替代** 3.2（功能/环境）与 4（真精度）的行级判定。

## 给模型的两条前置备注（开干前先读）

1. **读 xlsx → 走 python 解析**：ATK `report/*.xlsx` 的**表头与关键单元格是中文**（`运行结果` / `失败原因` / `编号` / `*精度通过` 等）；`cat` / `head` / `tail` / `less` / `grep` / `xlsx2csv` 这类纯文本/字节流工具读 xlsx 会得到二进制乱码或丢字（**物理上不可用**，不是禁令）。**改用 python**（`openpyxl` 直读或 `pandas + openpyxl`），最简一行式与完整模板 A/B 见 [op-atk-adapt「读 xlsx 的解析模板」](../op-atk-adapt/SKILL.md#读-xlsx中文表头--中文单元格的解析模板权威单一展开点)。本 skill 不重复展开。
2. **用户粘贴的 ATK 命令是骨架，按当下证据可微调**：用户从 DTS / 终端复制过来的 `atk task` 命令**不必原样照搬**——`-wl '[case_id,...]'` 必须按**当下 xlsx 解析出的真实失败 `case_id`** 重写、`-bl` / `-mt` / `-to` / `--save_data` / `-sd` / tee 文件名按本次目标按需叠加；但 `-c <case_json>` / `-n <yaml>` / `-p <python_script>` 路径、容器入口、`source <set_env.sh>`、工作目录、API 名**必须保持原文不动**。完整对照见 [op-atk-adapt「用户粘贴 ATK 命令的可调整范围」](../op-atk-adapt/SKILL.md#atk-参数可调整范围)。

## 何时进入本步

任一条件满足即可进入：

- 编排器或 3.2/4 需要**单跑定界**；
- xlsx 多行同时命中、需要**甄别内存踩踏**（批跑 FAIL / 单跑 PASS）；
- 需要**统计 aclnn 条数**来判断单 vs 多算子。

## 流程（严格按序）

### 1) 取 `case_id`（来源优先级）

1. **优先**：`atk_output/<CaseSet>_<ts>/report/*.xlsx` 的 `运行结果` / 任一 `*精度通过` 列筛出待分析行的 `编号` / `case_id`。**读 xlsx 走 python 解析**（中文表头/单元格 → 走 [op-atk-adapt 的解析模板](../op-atk-adapt/SKILL.md#读-xlsx中文表头--中文单元格的解析模板权威单一展开点)，最简形式即 `python -c "from openpyxl import load_workbook; ..."` 一行式）；`grep`/`cat` 直读 xlsx 会乱码或丢字（物理事实）。
2. **辅助**：`运行日志/atk_full_*.log` 里的 `FAIL`/`case id` 锚点。
3. **冲突**：以 **xlsx + 用户确认** 为准；列名不同时请用户映射一次后锁定。

### 2) 打开 aclnn 日志

```bash
export TORCH_NPU_LOGS=op_plugin   # 以用户实际 shell 为准
```

### 3) 单跑

> **改命令前先看「可调整范围」**：`-wl '[<case_id>]'` 的 `<case_id>` 必须来自上一步 python 解析的 xlsx 真实失败行；`-mt` / `-to` / `--save_data` 按本次目标按需叠；`-c <case_json>` / `-n <yaml>` / `-p <python_script>` **保持用户粘贴原文不动**。详见 [op-atk-adapt「用户粘贴 ATK 命令的可调整范围」](../op-atk-adapt/SKILL.md#atk-参数可调整范围)。

<a id="单跑取样铁律"></a>

> **单跑对象铁律（防取样偏差）**：`-wl` 里**必须含每个待定界的 FAILED / 确定性 False case_id 本身**，**禁止只单跑旁证 case 就外推结论**。具体：
> 1. **FAILED 行的 case_id 必跑**——`运行结果=FAILED` 的行即使 xlsx `失败原因` 看似"input.bin not exists"等末端症状，也必须单独 `-wl '[<该 id>]'` 跑一次并读其 atk.log（见 [3.5 error 链回溯](#35-读-atklog-完整-error-链)），**不得**用其它 PASS case 的结果代替。
> 2. **按 self dtype 分层取样**——确定性 False 跨 dtype 时，bool / fp16 / bf16 / fp32 / int64（用例实际出现的 dtype）**各取 ≥1 个 False case 单跑**，确认是否与 dtype 系统性相关（如算子不支持某 dtype）；**禁止**只跑支持型 dtype（fp16/bf16/int64）就拿"全 PASS"外推"与 dtype 无关"。
> 3. **证伪优先**——单跑目的不是"凑 PASS"，而是"能否复现批跑失败"。单跑 PASS 仅是**内存踩踏的必要非充分条件**，必须再过 [内存踩踏证伪门](#内存踩踏证伪门)。

```bash
unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2 && \
atk task -c <case_json> -n <yaml> -wl '[<case_id>]'
```
（同一行按需继续拼接，**不要**写成 `--save_data [input]` 等带方括号形式。例：`… -wl '[12]' --save_data input --save_data output`；仅在有 OOM/timeout 证据时再加 `-mt 5 -to 180` 等。）

产物与路径规则：见 [op-atk-adapt](../op-atk-adapt/SKILL.md) 的「产物定位表」。  
环境基线与 `-mt/-to` 调优判定规则：统一以 [op-atk-adapt 的「ATK 跑前环境基线」与「调优参数」](../op-atk-adapt/SKILL.md#atk-跑前环境基线权威本仓库唯一展开点) 为准。

<a id="35-读-atklog-完整-error-链"></a>

### 3.5) 读 atk.log 完整 error 链（FAILED case 必做）

> **xlsx 的 `失败原因` 列常只记末端症状（如 `FileNotFoundError: input.bin not exists` / `accuracy dc failed`），根因级错误（`Unsupported_Operator(EZ3002)` / `build op model failed`）只在 `atk_output/.../log/atk.log` 的 `[ERROR]` traceback 里**。**禁止**把末端症状当根因下结论。

对每个 `运行结果=FAILED`（或单跑 FAIL）的 case，**必须**回溯 atk.log 完整 error 链，取**时间戳最早的根因级 ERROR**：

```bash
# 按失败 case_id 在 atk.log 里找根因级错误（取最早的）
grep -nE 'EZ[0-9]{4}|Unsupported|not supported|Inner error|working operator name|build op model failed|Operator execute failed' \
  atk_output/<CaseSet>_<ts>/log/atk.log
```

**error 链因果方向（上游 → 下游，根因在最早）**：

```
EZ*/Unsupported_Operator / data type DT_* is not supported   ← 根因（算子/dtype 不支持）
  → RuntimeError: ... working operator name is <Op>            ← 算子崩溃
  → case <id> run opp failed                                    ← ATK 标记失败
  → cleanup case artifacts, reason=run_opp_task_failed         ← 正常失败清理（删 input.bin 等）
  → FileNotFoundError: input.bin not exists                     ← 派生症状（不得当根因）
```

- `input.bin not exists` / `accuracy dc failed on N loop` / `run opp task failed` 是**派生症状**，出现在 error 链**末端**，**不得**作为根因写入分析结论。
- 若 grep 到更早的 `EZ*` / `Unsupported` / `not supported` → 根因是算子/dtype 不支持，归 [3.2 functional](../op-delimitation-analyze-functional/SKILL.md) 或 flow-fix，**不是内存踩踏**。

### 4) 判读 & 记录

对每个 `case_id` 记录**四种状态**：

| 状态 | 含义 | 下一步 |
|------|------|--------|
| 单跑 PASS（批跑 FAIL） | **内存踩踏嫌疑** | 见下方 [证伪门](#内存踩踏证伪门)；通过后参考 [ATK#655](https://open.codehub.huawei.com/innersource/atk_G/ATK/issues/655)；**URL 仅存档**，要点由用户粘贴（见 [hard-constraints · 统一口径](../../knowledge/hard-constraints.md#统一口径--外部资源获取方式)） |
| 单跑 FAIL（aclnn 链路明确） | 单算子候选 | 进 3.1.1 计数 |
| 单跑 FAIL（日志含环境/配置错误） | 环境/功能 | 交回 [3.2 functional](../op-delimitation-analyze-functional/SKILL.md) |
| 单跑 SUCCESS + `*精度通过`=false | 真精度 | 交回 [4 precision](../op-delimitation-analyze-precision/SKILL.md)（可复用本步 `--save_data` 产物） |

<a id="内存踩踏证伪门"></a>

> **内存踩踏证伪门（"单跑 PASS / 批跑 FAIL → 内存踩踏"不是单向套用，必须三条件全满足）**：
> 1. **单跑的是失败 case 本身**——`-wl` 含该 FAILED / False `case_id`，不是旁证 case（见 [单跑取样铁律](#单跑取样铁律)）。
> 2. **单跑 atk.log 无更早的真实算子 error**——grep 该 case 的 `atk_output/.../log/atk.log`，无 `EZ[0-9]{4}` / `Unsupported` / `not supported` / `working operator name` / `build op model failed` 等根因级错误（见 [3.5 error 链回溯](#35-读-atklog-完整-error-链)）。`input.bin not exists` / `accuracy dc failed` 是**派生症状**，不算。
> 3. **换批组成能复现漂移**——小批重跑，False 集合随批内 case 组成变化（非某 case 固定属性）。
>
> **任一不满足 → 不得下"内存踩踏"结论**，回 [3.2 functional](../op-delimitation-analyze-functional/SKILL.md) 按 atk.log 首个 error 重新定性。

### 5) aclnn 统计 → 3.1.1

在 `TORCH_NPU_LOGS=op_plugin` 打开后的日志中，grep **`aclnn`** 关键字 → 收集算子名 → **计数**（不得杜撰名/数）。

**可选脚本**：若手头已有 **纯文本** 日志文件，可先跑 `bash tools/op-delimitation-analyze-single-op/scripts/extract_aclnn.sh <log>` 做一次 **token 频次表**（`grep -Eo`，无 `-P`，macOS/Linux 皆可）。脚本末尾说明仅为启发式：**单算子 / 组合类**仍以本节 **计数规则（含反向去重）** 与 xlsx 为准。**正反向分别分析 / 结果表**：[`references/operator-aclnn-analysis.md`](../../references/operator-aclnn-analysis.md)。

**可选脚本（落盘 save_data）**：若需一键对单 `case_id` 执行 `atk task ... --save_data input --save_data output` 并 tee，可用 `bash ${SKILL_ROOT}/tools/op-atk-adapt/scripts/save_atk_data.sh <json> <yaml> <case_id> <输出目录>`；脚本已内含 `unset http_proxy https_proxy` 与 `ATK_BIND_CPU_TYPE=2`。若与你司 ATK `atk_output` 目录层级不一致，以当次产物路径手动拷贝为准（参见 [op-atk-adapt「产物定位表」](../op-atk-adapt/SKILL.md#产物定位表全量跑)）。

<a id="311-计数规则"></a>

**3.1.1 计数规则**（aclnn 计数的唯一权威正文）：

> **共识口径**：`aclnn 个数` ≡ `aclnn 种类数` ≡ **「unique kind 去重 + 剥离辅助算子黑名单 + 正反向同名再去重」** 后剩下的 **业务算子 unique kind 数**。**不是** `grep aclnn` 的命中行数，也**不是** `extract_aclnn.sh` 顶部"顺序列表"的总行数（那只是 token 命中辅助，未做后续两步）。

按下面三步算（顺序固定）：

1. **按算子名（kind）unique 去重**：同一 aclnn 名字（含完整 CamelCase 函数名，如 `aclnnAddmm`）在日志里出现 N 次，**只计 1 种**。同名的常见来源：循环/重复调度、forward 与 backward 各出现一次、`InplaceCopy` / `Cast` 在多处反复被框架插入。
2. **剥离辅助算子黑名单**（**框架/内存/同步/类型转换/排布等非业务算子**——不参与"算子数"计数；但仍可留作证据放进 `aclnn 列表`，备注"辅助类，已剥离"）。当前已知典型项（**非穷举**，可按工程经验扩展，扩展时同步更新本表）：
   - 拷贝/内存：`aclnnInplaceCopy*` / `aclnnCopy*` / `aclnnMemcpy*` / `aclnnMemset*`
   - 排布/形状：`aclnnContiguous*` / `aclnnReshape*` / `aclnnViewCopy*` / `aclnnTransData*` / `aclnnExpand*`(纯广播)
   - 类型转换：`aclnnCast*`（**仅**纯 dtype 转换路径；若本次分析本身就是 cast 算子精度问题，**不**剥离）
   - 填充/初始化：`aclnnFill*` / `aclnnZeros*` / `aclnnOnes*`（仅做初值的辅助路径）
   - 同步/流：`aclnnStreamSynchronize*` / 任何只做调度/同步无数值计算的 op
3. **正反向同名再去重**：
   - **正向**（用例名 / `case_json` 文件名**不含 `backward`**）：第 1、2 步后直接当结果。
   - **反向**（用例名**含 `backward`**）：在第 1、2 步基础上，再去掉**与同次正向用例同名**的 aclnn（反向常会重复出现正向算子，这部分不该再被计入反向独立种类）。

**md 必填证据**（结单门控与 `flow-atk` 决策表都依赖此处）：

- `aclnn 总命中次数`（即 `extract_aclnn.sh` 的 `TOTAL`，仅作过程记录）；
- `aclnn unique kind 列表`（去重后、未剥离黑名单前的全集）；
- `已剥离的黑名单项`（按上面五类标注理由，如"`aclnnContiguous` → 排布辅助"）；
- `反向同名去重项`（仅反向场景填，否则写"N/A —— 正向"）；
- `最终 aclnn 种类数 = ___`（= 1 单算子 / >1 多算子，直接喂给 [`flow-atk` 决策表](../../skills/op-delimitation-flow-atk.md#atk-onescreen-decision)）。

结果：

- 若**最终 aclnn 种类数 = 1**（单算子），与编排器 **5.end** 一致：
  - 来源 = `SUCCESS+精度=false` 行 → 回 [`flow-atk` 决策表](../../skills/op-delimitation-flow-atk.md#atk-onescreen-decision) 走"询问用户是否补做 A2/A3 / dump / UT"的 early-stop 分支；
  - 来源 = `FAILED` 行 → 仍**不可** early-stop，回 [3.2 functional](../op-delimitation-analyze-functional/SKILL.md) 在功能/环境视角下定性。
- 否则 **单算子 = false**，交回编排器按 xlsx 再走 3.2 或 4（精度路径多算子）。

## 单跑汇总表（固定表头；放在pta 目录/`single-run/` 下）

放置规则与 [report](../../report/op-delimitation-report/SKILL.md) 的目录层级一致（**pta 目录** 下的子目录；命名含 opName+日期）。推荐列（CSV 或 xlsx）：

| case_id | atk task 命令 | 单跑结果 | aclnn 种类数 | aclnn unique kind 列表 | 已剥离黑名单项 | save_data 路径 | 备注 |
|---------|----------|----------|------------|-----------------------|---------------|----------------|------|
| 12 | `atk task -c ... -wl '[12]' --save_data ...` | PASS / FAIL | 3 | `aclnnFoo; aclnnBar; aclnnBaz` | `aclnnInplaceCopy(拷贝辅助); aclnnContiguous(排布辅助)` | `atk_output/.../input,output` | 批跑 FAIL 但单跑 PASS → 疑似内存踩踏 |

> **最小必填**：`case_id` / `atk task 命令` / `单跑结果` / `aclnn 种类数`（含 0 / 1 / >1）。`aclnn unique kind 列表` 与 `已剥离黑名单项` 在单算子 / 多算子定性的行**必填**（直接作为 [`flow-atk` 决策表](../../skills/op-delimitation-flow-atk.md#atk-onescreen-decision) 的计数证据），其余列按证据补全。`aclnn 种类数` 必须 = `unique kind 列表` 长度 − `已剥离黑名单项` 长度 − `反向同名去重项`数（反向场景）。

## 参考资料（非强约束）

- GitCode aclnn 映射表（若团队维护）：只作参考，**以当前环境日志/xlsx 为准**。
- 外链获取方式遵循 [hard-constraints · 统一口径](../../knowledge/hard-constraints.md#统一口径--外部资源获取方式)：助手**不主动打开**外链；用户提供行原文则采纳，仅 URL 则作存档。

## 原则

- 不伪造 aclnn 名 / 条数 / `-wl` 下标。
- 单跑前已与用户确认过 xlsx 表头（`运行结果`、`编号`、`失败原因`、`*精度通过` 等）。
- **读 xlsx 一律走 python 解析**（[op-atk-adapt 的解析模板](../op-atk-adapt/SKILL.md#读-xlsx中文表头--中文单元格的解析模板权威单一展开点)）；中文表头/单元格用 `cat`/`grep` 等文本工具直读会乱码丢字，是物理事实，不是禁令；不重复展开模板。
- **用户粘贴的 ATK 命令按需微调**：`-wl` / `-bl` / `-mt` / `-to` / `--save_data` 可改且**应**按当下 xlsx 失败行改写；`-c <case_json>` / `-n <yaml>` / `-p <python_script>` / 容器入口 / `set_env` / 工作目录 / API 名**保持原文不动**（[op-atk-adapt 可调整范围表](../op-atk-adapt/SKILL.md#atk-参数可调整范围)）。
- 单跑命令沿用 ATK 环境基线：跑前 `unset http_proxy https_proxy`，并 `export ATK_BIND_CPU_TYPE=2`。
- `-mt` / `-to` 仅在 OOM / timeout 有证据时调整；默认不改。
- 单跑 PASS 不等于"问题不存在"；**批跑 FAIL 单跑 PASS → 内存踩踏** 不是单向套用，必须先过 [证伪门](#内存踩踏证伪门) 三条件——尤其单跑对象必须是失败 case 本身、且该 case atk.log 无更早根因级 error（[3.5 error 链回溯](#35-读-atklog-完整-error-链)）。
- **禁止只单跑旁证 case 外推结论**：`-wl` 必含待定界 FAILED/False case 本身，且按 self dtype 分层取样（见 [单跑取样铁律](#单跑取样铁律)）。
- **确定性 False ≥2 个时必按 self dtype 做交叉表**（False/True 各 dtype 计数），按 dtype 分两路定性：
  - **False 聚集某 dtype（如 bool 全 False）且该 dtype 在 atk.log 有 `EZ*`/`Unsupported` → dtype 系统性不支持**，非内存踩踏，归 [3.2 functional](../op-delimitation-analyze-functional/SKILL.md) 或 flow-fix。
  - **支持型 dtype 仍 False（如 fp16/bf16/fp32/int64 各有 False）→ 混合信号残余**，须二次判定：① 用 `--save_data` 固定输入比对 indices，区分 `max(dim)` 并列最大值 ties-indices 规则差异（NPU/CPU 选取不同）vs ② 真内存踩踏（回 [证伪门](#内存踩踏证伪门) 三条件）。**不得**把"支持型 dtype False"与"dtype 不支持"一锅端。
  - "与 dtype 无关"结论必须先有分桶表反证（支持型 dtype False 残余已排除 ties-indices）才能下。
