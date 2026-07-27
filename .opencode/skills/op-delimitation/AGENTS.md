# API 一致性分析 Agent（ATK · 多 Skill）

> 适用范围：API 一致性分析任务的"适配+执行 → 分析 → 结单"多 skill 流程。Gate 0 路由与硬约束链见 [`SKILL.md`](SKILL.md)。

上传/迁移时带上 `AGENTS.md`、`CLAUDE.md`，以及 `tools/`、`skills/`、`knowledge/`、`references/`、`templates/`、`report/`、`examples/` 等正文路径。

## Skill 一览（Codex 视角）

| Skill | 路径 | 说明 |
|-------|------|------|
| op-delimitation | [SKILL.md](SKILL.md) | 套件总入口（Read） |
| op-delimitation-orchestrator | [skills/op-delimitation-orchestrator/SKILL.md](skills/op-delimitation-orchestrator/SKILL.md) | Gate 0 + 总编排 |
| op-delimitation-flow-atk | [skills/op-delimitation-flow-atk.md](skills/op-delimitation-flow-atk.md) | 分支 A · ATK 主编排 |
| manual-connect-npu-server | [tools/manual-connect-npu-server/SKILL.md](tools/manual-connect-npu-server/SKILL.md) | 远端 SSH 接入（窄可选，默认本地） |
| op-delimitation-op-atk-adapt | [tools/op-atk-adapt/SKILL.md](tools/op-atk-adapt/SKILL.md) | CANN→PTA ATK 适配+执行 |
| op-delimitation-analyze-single-op | [tools/op-delimitation-analyze-single-op/SKILL.md](tools/op-delimitation-analyze-single-op/SKILL.md) | 3.1 单算子 |
| op-delimitation-analyze-functional | [tools/op-delimitation-analyze-functional/SKILL.md](tools/op-delimitation-analyze-functional/SKILL.md) | 3.2 功能/环境 |
| op-delimitation-analyze-precision | [tools/op-delimitation-analyze-precision/SKILL.md](tools/op-delimitation-analyze-precision/SKILL.md) | §4 精度 |
| op-delimitation-flow-fix | [skills/op-delimitation-flow-fix.md](skills/op-delimitation-flow-fix.md) | 分支 C · 修复子流程 |
| op-delimitation-report | [report/op-delimitation-report/SKILL.md](report/op-delimitation-report/SKILL.md) | 5 结单 |

## A2 / A3 已知问题表

- **当前不接入**：Agent 不查 A2、A3；遇到需对照情形时提示用户自行用关键词搜索（见 `knowledge/a2-a3-a5-discipline.md`）。
- **A5 表**：当前不自动解析 xlsx，由用户先筛行再把命中行原文粘给助手。
