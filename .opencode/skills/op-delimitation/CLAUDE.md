# API 一致性分析套件（ATK · Claude Code / CLI）

> 套件总入口与 Gate 0 路由见 [`SKILL.md`](SKILL.md)。本文件只管 Claude Code 平台加载说明。

## 技能发现与调用

各技能正本在 `skills/`、`tools/`、`report/` 下。Claude Code（终端 `claude`）有两种用法：

### 方式一：subagent 一键托管（推荐）

本仓库在 **[`.claude/agents/op-delimitation.md`](../../.claude/agents/op-delimitation.md)** 提供专用 subagent。它按 `SKILL.md` 的「子 skill 速查表」逐段 Read 正本、走 Gate 0 → 契约 → flow-atk → 下沉 tools → 结单；适合端到端，也支持「指定从某阶段切入」（如已持 xlsx 直接做 3.1 单算子、或只写结单）。

```
> 用 op-delimitation agent 处理这个算子：CANN ATK 用例目录 <路径>，op-plugin 仓根 <路径>。
> （或：只做 3.1 单算子定界 / 只写结单 —— 告诉 agent 从哪个阶段切入）
```

### 方式二：直接 Read 正本（跨平台、单点深入）

不用 subagent 时，直接让助手 Read 对应 `SKILL.md`：

| 目的 | 请先 Read |
|------|-----------|
| Gate 0 + 总编排 | [`skills/op-delimitation-orchestrator/SKILL.md`](skills/op-delimitation-orchestrator/SKILL.md) |
| ATK 主编排 | [`skills/op-delimitation-flow-atk.md`](skills/op-delimitation-flow-atk.md) |
| 修复子流程 | [`skills/op-delimitation-flow-fix.md`](skills/op-delimitation-flow-fix.md) |
| 远端 SSH（窄可选） | [`tools/manual-connect-npu-server/SKILL.md`](tools/manual-connect-npu-server/SKILL.md) |
| ATK 适配+执行 | [`tools/op-atk-adapt/SKILL.md`](tools/op-atk-adapt/SKILL.md) |
| 单算子 / 功能 / 精度 | `tools/op-delimitation-analyze-*/SKILL.md` |
| 结单 | [`report/op-delimitation-report/SKILL.md`](report/op-delimitation-report/SKILL.md) |

> 历史上曾用 `.claude/skills/<名>/SKILL.md` 软链 + `/<技能名>` slash 调用；该软链入口当前未在本仓库提供，请改用上述 subagent 或直接 Read 正本。

## 远端配置

编辑 `tools/manual-connect-npu-server/config/remote-config.yaml`；勿将含密码/私钥的副本提交到公开 Git。

## 与 Cursor 共用

正本在 `SKILL.md`、`skills/`、`tools/`、`knowledge/`、`report/`。编辑器侧按各自平台规则加载。
