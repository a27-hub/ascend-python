# API 一致性分析

**PyTorch Ascend（`torch_npu`）API 一致性分析技能集（A2 / A3 / A5）**

## 项目简介

本仓库是一套面向 **PyTorch Ascend API 一致性** 任务的可复用技能集合，供 **Cursor、Claude Code、Codex** 等助手按 **`SKILL.md`**、**`skills/`**、**`tools/`** 加载使用。流程入口与门禁规则详见 **`AGENTS.md`**、**`CLAUDE.md`**，并与各 `SKILL.md` 正文保持一致。

**第一性原理**：任务上的「一致性」必须以 **可追溯的 ATK 执行与证据**（ATK：`report/*.xlsx` + `运行日志/`）为出发点，再在 **功能 / 精度 / 单算子** 等路径上逐项归因，最后用结单模版收口——**不靠打屏快照下结论**，**不凭空编造 case id 或路径**。

**分析原则**：在 ATK 脚本执行场景下，默认 **设备语义与后端名**（如 CUDA↔NPU、NCCL↔HCCL）已由运行环境或用户粘贴命令承载。助手在各阶段的判断应聚焦于：**测试结果与断言所依赖的算子语义、分布式原语、 dtype/layout、梯度与确定性** 在 NPU 路径上是否与 CPU/GPU 标杆预期一致，并区分 **环境问题、单算子缺陷、组合算子发散、精度阈值** 等不同出口。

---

## 工作流程概览

```
开场契约 ──► Gate 0（按 CANN ATK 用例目录 + op-plugin 仓根 二出口）
                 │
          ┌──────┴──────┐
          ▼             ▼
   分支 A·ATK        极小出口
   flow-atk
          ├─ CANN→PTA ATK 适配/xlsx
          ├─ 行级归因
          └─ report
```

| 意图（阶段感）       | 做什么（摘要）                                                                                                                                                                                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **套件总入口** | 读**[`SKILL.md`](SKILL.md)** — Gate 0 路由、硬约束链、子 skill 速查。                                                                                                                                                                                           |
| **分流与契约** | **[`skills/op-delimitation-orchestrator/SKILL.md`](skills/op-delimitation-orchestrator/SKILL.md)** — Gate 0（A · ATK / 极小出口）+ 开场契约；字段清单见 [`knowledge/intake-contract.md`](knowledge/intake-contract.md)。               |
| **主编排**     | [`skills/op-delimitation-flow-atk.md`](skills/op-delimitation-flow-atk.md)。**两遍规则**仍为硬约束。 |
| **证据与细化** | ATK：**`tools/op-atk-adapt/`**；远端（窄可选）：**`tools/manual-connect-npu-server/`**；按需 **`tools/op-delimitation-analyze-*`**。细节见 [`references/`](references/) 与 [`knowledge/`](knowledge/INDEX.md)。                            |
| **结单**       | **`report/op-delimitation-report/`**。                                                                                                                                                                                                                          |

**路径约定：** 各工具的 **权威正文** 为 **`tools/.../SKILL.md`** 或 **`skills/...`**；Claude Code 侧另提供 **`.claude/agents/op-delimitation.md`** subagent 作为一键托管入口（详见 [`CLAUDE.md`](CLAUDE.md)）。

---

## 仓库布局

目录结构与子 skill 速查详见 [`SKILL.md`](SKILL.md)。核心目录：`knowledge/`（必读约束）· `references/`（按需操作）· `skills/`（编排器 / flow-*）· `tools/`（op-atk-adapt / analyze-* / manual-connect）· `report/`（结单）· `examples/`（案例归档）· `templates/`（模板）。

---

## 快速开始

推荐使用「**先 [`SKILL.md`](SKILL.md) / orchestrator → 再主编排 → 按证据下沉 tools**」的顺序。将本仓库作为 Cursor / Claude Code 工作区根（或把 **`skills/`、`tools/`、`.cursor/`** 一并纳入上下文），再任选下面的一种方式开聊。

### 方式一：从编排器逐段推进（最稳妥）

让助手 **Read** **[`skills/op-delimitation-orchestrator/SKILL.md`](skills/op-delimitation-orchestrator/SKILL.md)**，先完成 **Gate 0** 与 **开场契约**，再按分流进入 flow-atk。适合「首次进入某算子分析、希望别跳步」的场景。

```
> 请读取本仓库 skills/op-delimitation-orchestrator/SKILL.md，按 Gate 0 判断分支。
> CANN ATK 用例目录 + op-plugin 仓根路径请整段粘贴。
> 缺什么字段请列出清单，齐了再往下走。
```

### 方式二：端到端托管一句话（仍受 skill 纪律约束）

```
> 按本仓库 api 一致性流程处理：从编排器 Gate 0 开始，
> 分支 A 跟 flow-atk，直到能写结单所需字段。
> CANN ATK 用例目录：<路径>；op-plugin 仓根：<路径>；产物根目录：<路径>。
```

### 方式三：单点深入（只抠某一步）

| 目的                                              | 请先 Read / 执行                                                                                   |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 只接 SSH / 对齐`torch_npu`（窄可选，默认本地）   | **`tools/manual-connect-npu-server/SKILL.md`**                                             |
| 只抠 ATK 适配+执行、xlsx、`-mt/-to`、`--save_data` | **`tools/op-atk-adapt/SKILL.md`**                                             |
| 只用 UT 壳比对 CPU/NPU（已有 bin/pt）             | **`tools/op-delimitation-analyze-precision/scripts/ut_forward.py`** / **`ut_backward.py`** |
| 只写结单                                          | **`report/op-delimitation-report/SKILL.md`**                                               |

**Claude Code**：本仓库在 **`.claude/agents/op-delimitation.md`** 提供专用 subagent（一键托管或指定阶段切入）；单点深入也可直接 Read 对应 `tools/.../SKILL.md`。详见 **`CLAUDE.md`**。

---

## 延伸阅读

- **[`examples/case-library.md`](examples/case-library.md)** — 历史 ~130 例 API 根因/结论详情库（合并自 a2a3，按功能/精度/确定性/内存/分布式/边界值分类）。
- **精度/内存对比脚本**（合并自 a2a3）：`single_compare` / `double_benchmark_compare` / `new_benchmark_compare` / `cv_fused_double_benchmark_compare` / `memory_compare` / `utils` — 见 [`tools/op-delimitation-analyze-precision/scripts/`](tools/op-delimitation-analyze-precision/scripts/) 与选型表 [`references/atk-tuning-cases.md`](references/atk-tuning-cases.md)；Step 1–4 详细操作见 [`op-atk-adapt`](tools/op-atk-adapt/SKILL.md)。
- **UT 归档**：UT 脚本在 **`tools/op-delimitation-analyze-precision/scripts/`**（`ut_forward.py` / `ut_backward.py` / `determinism_check.py`）。
- **[`knowledge/INDEX.md`](knowledge/INDEX.md)** — 重构迁移对照与权威路径索引。

---

## 贡献与同步约定

- 更新 **`AGENTS.md`**、**`CLAUDE.md`**、**`.claude/agents/op-delimitation.md`** 与 **`skills/`、`tools/`、`knowledge/`** 时，避免「正本路径」口径漂移；文档内链优先写 **`tools/.../SKILL.md`** 或 **`knowledge/`**，并注明 subagent 等价入口。
- **硬约束 / 契约 / 反模式** 以 [`knowledge/`](knowledge/INDEX.md) 为唯一展开点；orchestrator 只保留 Gate 0 与路由高信号段落。
