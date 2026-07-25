# Case 库导航

> op-adaptation 经验案例库。按接入模式分区，每区一个子索引。C-confirm（Create）/ M-discover（Modify）后按算子类型/场景/改动类型匹配，命中则读对应 case 全文作决策与避坑参考。

| 分区 | 适用 | 子索引 |
|------|------|--------|
| **Create 模式** | 全新算子接入（`.md` + `_def.cpp` → torch_npu） | [create-mode/README.md](create-mode/README.md) |
| **Modify 模式** | 已有算子的修改/重构 | [modify-mode/README.md](modify-mode/README.md) |
| **M-adapt 子模式** | A5 平台适配（Ascend950 / Atlas A5），含 4 维选型指南 | [m-adapt/README.md](m-adapt/README.md) |

> **通用陷阱（跨模式必读）**：任意带「可选/条件输出」(out_flag 类) 的算子接入 → [create-mode/npu_moe_gating_top_k-create.md](create-mode/npu_moe_gating_top_k-create.md)。

> 新增 case：从 `../templates/case-template.md` 起，放入对应模式子目录，并在该子目录 README 索引表加一行。
