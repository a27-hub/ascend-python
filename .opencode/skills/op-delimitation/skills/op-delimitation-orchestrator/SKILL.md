---
name: op-delimitation-orchestrator
description: >-
  Use as the op-delimitation routing workflow:
  collect the opening contract, classify Gate 0 into ATK (CANN ATK → PTA ATK
  adaptation) or narrow exit, and hand off to the matching flow.
---

# API 一致性 — 编排器（ATK 单主路径）

> **适用范围**：API 一致性类任务。Gate 0 把任务分到 **单主路径 A · ATK**，外加 **极小出口**（与本流程无关或无法对齐输入前经一次确认仍无法补齐）。
> **(A) ATK**：输入含 CANN ATK 用例目录 + op-plugin 仓根 → **[op-delimitation-flow-atk](../op-delimitation-flow-atk.md)**（CANN→PTA ATK 适配 + 跑 → xlsx-first、两遍再下沉 tools）→ report。  
> **（极小出口）**：与 API 一致性无关；或用户无法补齐 CANN ATK 用例目录 + op-plugin 仓根，经 **一次**对齐确认后仍不能收敛 —— **礼貌退出**。  
>
> **一句话原则**：先判 ATK → 再适配 + 跑（产出 xlsx）→ 再行级判读 → 最后结单 md。**不对着打屏下结论；不造表；A2/A3 只提示用户自搜。**

## 全局硬约束（权威外置 · 必读链）

> **全文单点权威**：[`knowledge/hard-constraints.md`](../../knowledge/hard-constraints.md)（[`§统一口径`](../../knowledge/hard-constraints.md#统一口径--外部资源获取方式) · [`§远端只读边界`](../../knowledge/hard-constraints.md#远端只读边界)）。
>
> 含：**统一口径**（wiki/CodeHub 不抓取；xlsx python 解析例外；URL 仅存档）+ **远端只读边界**（禁止 pip/conda/apt/rm 越界；允许 pta 目录写产物 + 当前 shell 临时 env + **切换已存在 venv/conda**；灰区写 `运行记录.md`）+ **op-atk 适配态** + **失误恢复**。
>
> 编排器与其它 skill **短句引用**上文，不在此重复展开。命中禁止项 → 操作失败，回到 hard-constraints 重判。

## 何时使用 / 何时不使用

**使用**（全部满足）：

1. 话题涉及 `api一致性 / 精度 / 功能 / 单算子 / NPU torch_npu` 等；
2. **输入**：CANN ATK 用例目录（4 件套 `atk.yaml`/`generate.py`/`executor.py`/`node.yaml`，或 3 件套 `case.json`/`executor.py`/`node.yaml`——用户已生成用例）+ op-plugin 仓根（含 `CHECKLIST.md` + `codegen/templates/_op_plugin_docs.py`）；
3. 预期产物：`atk_output/.../report/*.xlsx` 等 ATK 产物。

**不使用**（极小出口 · 满足任一可先礼貌退出或在一次确认未果后退出）：

- 话题与本仓库 **API 一致性 / torch_npu 对标**无关；
- 用户无法补齐 CANN ATK 用例目录 + op-plugin 仓根，经单次澄清仍不能收敛。

## Gate 0（硬门槛，先过再继续）

**在执行任何其它步骤前**，先做**输入分类**：

```text
取：用户提供的 CANN ATK 用例目录路径 + op-plugin 仓根路径
判：
  1) 两者都存在且（4 件套齐全，或 3 件套 case.json/executor.py/node.yaml 齐全）→ 分支 A (ATK)，记录 mode（4file/3file）
  2) 明确不属于 API 一致性，或无法在澄清后对齐输入 → 极小出口 · 礼貌退出
```

| 判定 | 分支 | 动作 |
|------|------|------|
| CANN ATK 用例目录 + op-plugin 仓根 齐全 | **A · ATK** | 「开场契约」→ [flow-atk](../op-delimitation-flow-atk.md)。 |
| 不属于本流程或无法对齐输入 | **极小出口 · 退出** | 模板：`当前输入无法归入 ATK 路径；若属 API 一致性请补充 CANN ATK 用例目录 + op-plugin 仓根，若否则本 skill 不展开。` |

> **不要做的事**：不得在无用户确认时**捏造**用例目录路径或 opName。

## 开场契约（Gate 0 通过后，阻塞后续）

按序向用户索取，缺一项前**不得**开始执行/结论。

### 字段清单 + stamp（权威外置）

> **全文**：[`knowledge/intake-contract.md`](../../knowledge/intake-contract.md)（字段清单 / 跨字段校验 / stamp 规则）  
> **stamp 模板**：[`templates/stamp-table.md`](../../templates/stamp-table.md)
>
> **编排器仍执行的摘要**：
> - 首轮回复必带 stamp（4 标记：✓/✗/◯/⚠）；缺阻塞项不得进下一阶段。

### 通用项

1. **CANN ATK 用例目录路径**（必填 · 阻塞）：目录下应有 `atk.yaml`/`generate.py`/`executor.py`/`node.yaml` 4 件套，或 `case.json`/`executor.py`/`node.yaml` 3 件套（预生成用例）。
2. **op-plugin 仓根路径**（必填 · 阻塞）：仓根下应有 `CHECKLIST.md` + `codegen/templates/_op_plugin_docs.py`。
3. **目标 opName**（可选）：省略则 `auto_adapt.py` 从 `CHECKLIST.md` 自动提取。vs `CHECKLIST.md` 中 `torch_npu function name:` 行。
4. **NPU 设备 id**（可选，默认 0；远端时为远端 NPU id）。
5. **pta 目录（{opName}_pta_atk/）**（必填 · 阻塞）：确认结单/日志将写入的**pta 目录（{opName}_pta_atk/）**（避免写入 `home/`、`data/`、`root/` 广根目录）。在最终落地的pta 目录子目录（建议形如 **`pta 目录/{opName}_{日期}/`**）执行 **`mkdir -p analyze`**。
6. **远端 SSH 字段**（窄条件必填）：仅当「op-plugin 仓 + NPU 同在远端」时阻塞；**默认本地**（见 [hard-constraints · 本地同机约束](../../knowledge/hard-constraints.md)）。**不支持**「本地建 pta + 远端跑 NPU」。需登录远端 → 引导填 `tools/manual-connect-npu-server/config/remote-config.yaml`。
7. **环境变量基线**：跑 `atk task` 前必须 `unset http_proxy https_proxy` 并 `export ATK_BIND_CPU_TYPE=2`。统一以 [op-atk-adapt · ATK 跑前环境基线](../../tools/op-atk-adapt/SKILL.md#atk-跑前环境基线权威本仓库唯一展开点) 为权威源。

## 下一阶段路由（契约达成后）

### 分支 A · ATK

| 次序 | 说明 |
|------|------|
| （窄可选）远端 | [manual-connect-npu-server](../../tools/manual-connect-npu-server/SKILL.md) — 仅当仓+NPU同在远端时 |
| **主编排（必读）** | **[op-delimitation-flow-atk](../op-delimitation-flow-atk.md)** — ATK xlsx 一屏决策表、两遍规则、与各 `tools/` 的映射 |
| **适配+执行** | **[op-atk-adapt](../../tools/op-atk-adapt/SKILL.md)** — CANN→PTA ATK 适配 + 跑（4 步：auto_adapt → atk case → 改 executor → atk task）|
| 5.end | [op-delimitation-report](../../report/op-delimitation-report/SKILL.md) |

### 次级路由 · 定界 → flow-fix（修复子流程）

> **不是 Gate 0 出口**。分支 A 主编排走到 **阶段 4.1 定界** 后，由 **定界标签** 决定是否进 **[flow-fix](../op-delimitation-flow-fix.md)**（分支 C · 修复子流程）。

| 定界标签 | 进 flow-fix？ |
|---|---|
| 单算子（3.1）/ 功能·环境→aclnn类 | 是（须 flow-fix **阶段 1 诊断**） |
| **aclnn 算子未实现/不存在** | **否**（直接结单标「转 CANN 团队」）|
| 组合类精度 | 是（阶段 1 诊断定首个分叉） |
| **真精度** | 默认否；疑似适配层 dtype/cast 缺陷 → 进诊断复核 |
| 其余（环境/驱动、用例脚本、超时/OOM、调度、A2/A3 已知） | 否（原 report 结单） |

> **修复范围 = pta 适配层**（op-adaptation 生成范围）。**Ascend C 算子 / aclnn 底层** → 转 CANN 团队；**精度** → 超范围。

**前后关系铁律**：flow-fix 由 op-delimitation **主控**；`op-adaptation(MODIFY)` 是它在 **阶段 1 诊断确诊** 后 **下沉调用** 的下游（地位同 `op-atk-adapt`/`analyze-*`），**永不自行启动**。详见 [flow-fix](../op-delimitation-flow-fix.md) 与 [hard-constraints · 边界切换](../../knowledge/hard-constraints.md#边界切换--只读分析态--本地可写修复态)。

**术语**  
- **5.end**：已达到可结单条件。**≠** 「终端打出 `atk task success!`」。  
- **early-stop**：ATK 下若 3.2 已定功能结论，可跳 4，依据写在 md。

## 结单门控（按分支勾齐才能宣布「已结单」）

**通用**：

- [ ] **契约**：CANN ATK 用例目录 + op-plugin 仓根 + pta 目录（{opName}_pta_atk/） 已记录。
- [ ] **环境**：默认本地时 `local_torch_npu_dir`/`version` + `op_plugin_repo_root` + `op_plugin_git_commit` 已记入会话上下文；远端跑时 `remote_torch_npu_dir`/`version` 已记入（窄条件性）。
- [ ] **5.end**：产物 = 分析结论 md + 运行记录 md；命名含 opName + 日期。

**分支 A · ATK**：在通用项之上，另完成 [op-delimitation-flow-atk](../op-delimitation-flow-atk.md) 中 **「ATK · 结单门控核对单」**。

**分支 C · 修复子流程追加**（仅当走过 [flow-fix](../op-delimitation-flow-fix.md)）：

- [ ] **handoff 包**：`{opName}_pta_atk/handoff/根因分析与修复交接.md` 已产出且 op-adaptation 回填字段齐全。
- [ ] **边界切换**：`运行记录.md` 已记录 `boundary=READONLY_RELEASED` + `env=LOCAL_DEV_NPU` + 切换时刻。
- [ ] **修复**：op-adaptation(MODIFY) 已调用；`files_modified` 清单已落 md。
- [ ] **回归**：用 **原始 atk task 命令**（来自 op-atk-adapt 第 4 步）本地重跑，`回归报告.md` 已产出；结论 PASS 或熔断（≤3 轮）如实记录。
- [ ] **修复红线**：[flow-fix §阶段 5](../op-delimitation-flow-fix.md) 5 项红线自检全部通过。

## A2 / A3 / A5 已知问题表的处理纪律

> 全文：[`knowledge/a2-a3-a5-discipline.md`](../../knowledge/a2-a3-a5-discipline.md)

## 反模式（必须避免）

> 全文：[`knowledge/anti-patterns.md`](../../knowledge/anti-patterns.md)（全局 #1–#8）。
>
> **编排器必记两条**：① 不绕过 Gate 0；② 不凭 URL 自造摘要。

## 信息不足时

列出**缺失项清单**（如：CANN ATK 用例目录路径、op-plugin 仓根、opName、pta 目录（{opName}_pta_atk/）），请用户补齐。**禁用**占位假路径冒充已执行。
