---
name: op-delimitation
description: >-
  Use for PyTorch Ascend (torch_npu) API 一致性分析 across A2/A3/A5 servers, including
  Gate 0 routing, evidence alignment, tool
  handoff, and final report navigation across this bundle.
---
# API 一致性 · 技能套件总入口

> **中心短、辐射厚**：本文件只做 **路由**；硬约束 / 契约字段 / 反模式见 [`knowledge/`](knowledge/INDEX.md)；主编排见 [`skills/`](skills/)；可操作细节见 [`tools/`](tools/)。
>
> **统一入口**：本套件已合并原 `op-delimitation-a2a3` 与 `op-delimitation-a5`，覆盖 **A2 / A3 / A5** 全服务器。a2a3 的历史案例库（[`examples/case-library.md`](examples/case-library.md)）、精度/内存对比脚本、以及 Step 1–4 详细操作（[`op-atk-adapt`](tools/op-atk-adapt/SKILL.md)）均已并入。a2a3 原始文档已删除。

## 概述

面向 **PyTorch Ascend（torch_npu）API 一致性** 任务：**Gate 0 分流 → 开场契约 → 分支主编排 → 按需下沉 tools → 结单 report**。证据以用户粘贴文本 + ATK xlsx 为准（见 [hard-constraints · 统一口径](knowledge/hard-constraints.md)）。

## Gate 0（二出口）

| 判定                                                                               | 分支                           | 下一步                                                                                                                        |
| ---------------------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| CANN ATK 用例目录 + op-plugin 仓根 齐全                                            | **A · ATK**             | [skills/op-delimitation-flow-atk.md](skills/op-delimitation-flow-atk.md)                                                                                       |
| 与 API 一致性无关或无法对齐输入                                                    | **极小出口**             | 礼貌退出                                                                                                                      |

**主编排器（契约 + 路由）**：[skills/op-delimitation-orchestrator/SKILL.md](skills/op-delimitation-orchestrator/SKILL.md)

**定界次级路由 → flow-fix（修复子流程）**：分支 A 主编排走到阶段 4.1 定界后，经**适配层判定**决定是否进 [flow-fix](skills/op-delimitation-flow-fix.md)——**仅修 op-plugin 适配层**（op-adaptation 生成范围）；Ascend C 算子/aclnn 底层转 CANN，精度超范围，二者不进。`op-adaptation(MODIFY)` 是其下游，由 op-delimitation 主控调用，**不自行启动**。

## 子 skill 速查

| 阶段          | Skill                                                                                                                                                                                                                |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gate 0 + 契约 | [orchestrator](skills/op-delimitation-orchestrator/SKILL.md)                                                                                                                                                                          |
| A 主编排      | [flow-atk](skills/op-delimitation-flow-atk.md)                                                                                                                                                                                        |
| C 修复子流程  | [flow-fix](skills/op-delimitation-flow-fix.md) — 定界可修复后调 op-adaptation(MODIFY) + 回归 |
| 远端 SSH（窄可选） | [manual-connect-npu-server](tools/manual-connect-npu-server/SKILL.md) — 仅仓+NPU同在远端时                                                                                                                                                 |
| A 适配+执行   | [op-atk-adapt](tools/op-atk-adapt/SKILL.md)                                                                                                                                                                 |
| A 分析        | [analyze-single-op](tools/op-delimitation-analyze-single-op/SKILL.md) / [analyze-functional](tools/op-delimitation-analyze-functional/SKILL.md) / [analyze-precision](tools/op-delimitation-analyze-precision/SKILL.md) |
| 结单          | [report](report/op-delimitation-report/SKILL.md)                                                                                                                                                                      |

## 硬约束（必读链）

- [knowledge/hard-constraints.md](knowledge/hard-constraints.md) — 统一口径 + 远端只读边界（含 venv **切换**允许 / **造改删**禁止）+ op-atk 适配态 + 本地同机约束
- [knowledge/intake-contract.md](knowledge/intake-contract.md) — 字段清单 + stamp
- [knowledge/anti-patterns.md](knowledge/anti-patterns.md) — 流程反模式
- [knowledge/gotchas.md](knowledge/gotchas.md) — 分析高频陷阱
- [knowledge/history.md](knowledge/history.md) — 案例索引
- [templates/stamp-table.md](templates/stamp-table.md) — 首轮 stamp 模板
- [templates/device-env-snapshot.md](templates/device-env-snapshot.md) — 结单环境快照
- [templates/numeric-evidence.md](templates/numeric-evidence.md) — UT/确定性/Dump 数值块

## references/（按需 Read）

- [atk-xlsx-parsing.md](references/atk-xlsx-parsing.md) — xlsx 解析
- [atk-tuning-cases.md](references/atk-tuning-cases.md) — `-mt`/`-to` 触发条件与阶梯（瘦身后）
- [ssh-auth-policy.md](references/ssh-auth-policy.md) — SSH 三档 + 硬禁（窄可选）
- [ssh-windows-setup.md](references/ssh-windows-setup.md) — Windows 装机（窄可选）
- [operator-aclnn-analysis.md](references/operator-aclnn-analysis.md) — aclnn 正反向 / 单算子 vs 组合
- [dump-analysis.md](references/dump-analysis.md) — 组合类 Dump 定界
- [gpu-nodes-config.md](references/gpu-nodes-config.md) — 双标杆 nodes.yaml / output_path
- **合并自 a2a3**：现行权威——Step 1–4 操作见 [`op-atk-adapt`](tools/op-atk-adapt/SKILL.md) · ATK 用例定位见 [`flow-atk` 一屏决策表](skills/op-delimitation-flow-atk.md) · 精度标杆选型见 [`atk-tuning-cases`](references/atk-tuning-cases.md) · ACLNN 分析见 [`operator-aclnn-analysis`](references/operator-aclnn-analysis.md)

## 人类文档

- [examples/case-library.md](examples/case-library.md) — 历史 ~130 例 API 根因/结论详情库（合并自 a2a3）
- [README.md](README.md) — 快速开始
- [AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md) — 平台加载说明
