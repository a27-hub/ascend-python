# A2 / A3 / A5 已知问题表的处理纪律

> **编排器入口**：[`skills/op-delimitation-orchestrator/SKILL.md`](../skills/op-delimitation-orchestrator/SKILL.md)

- **A2 / A3**（已知问题表，通常不在当前任务附件中）：**不接入、不代查、不编造命中行**；需要对照时仅**提示用户**自行用关键词检索（case 名、ATK json 名、API 名、失败摘要片段），助手可给**建议检索词**。
- **A5（`A5继承框架自动化问题分析.xlsx`）**（历史已知问题表，用户可提供）：**允许助手读**，但只按本次任务的 API 名 / 命令 / 失败用例编号**筛行**，不 dump 整表；用户手上若为等价表（非 A5 文件名）允许代替。
