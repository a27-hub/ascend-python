# op-delimitation 知识库索引

> 各 knowledge 文件管什么、何时读。读完本索引即可判定是否需继续 Read 具体文件——**「两遍规则」的配套增强**。

## knowledge/（必读链 · 按需 Read）

| 文件 | 一句话职责 | 触发条件（何时 Read） |
|------|-----------|----------------------|
| [`hard-constraints.md`](hard-constraints.md) | 远端只读边界 + xlsx 解析例外 + ATK 命令约束 + op-atk 适配态 + 失误恢复 | 任何涉及远端操作 / 读表 / 命令调整时必读 |
| [`intake-contract.md`](intake-contract.md) | 开场契约字段清单 + 跨字段校验 + stamp 规则 | Gate 0 通过后首轮必读 |
| [`anti-patterns.md`](anti-patterns.md) | 全局 #1-#8 反模式 + SSH 特化 | 任何阶段决策前扫 |
| [`gotchas.md`](gotchas.md) | 9 条分析陷阱（现象/陷阱/正确做法三段式） | 分析结论与预期不符时 Read |
| [`history.md`](history.md) | 结论类型速查 + 高频根因模式 + 案例索引 | Step 3 定位时提示用户检索 |
| [`a2-a3-a5-discipline.md`](a2-a3-a5-discipline.md) | A2/A3/A5 已知问题表处理纪律 | 遇 A2/A3/A5 对照时 |

## templates/（模板 · 按阶段触发）

| 文件 | 一句话职责 | 触发条件（何时用） |
|------|-----------|------------------|
| [`../templates/stamp-table.md`](../templates/stamp-table.md) | 开场契约状态 stamp 表 | 契约首轮回复必带 |
| [`../templates/device-env-snapshot.md`](../templates/device-env-snapshot.md) | 设备环境快照（CANN/torch/torch_npu/驱动/NPU） | 结单时 |
| [`../templates/numeric-evidence.md`](../templates/numeric-evidence.md) | UT/确定性/Dump 数值证据块 | UT / 确定性 / Dump 产出时 |
| [`../templates/conclusion.md`](../templates/conclusion.md) | Step5 结单报告 7 项骨架 | Step5 结单时 |

## 重构迁移史

MR-1~MR-6 迁移对照表已移 [`MIGRATION.md`](MIGRATION.md)（仅供重构追溯，解单不读）。
