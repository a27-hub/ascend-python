# 接缝论证（技术根基 · 下游零改动消费的前提）

> 以下论证链证明：op-atk-adapt 产出的 xlsx 可被 flow-atk 一屏决策表 / analyze-single-op / analyze-functional / analyze-precision **零改动**消费。

**论证链（6 步）**：

1. **op-atk 适配后 `node.yaml` 含双 backend**：`auto_adapt.py` 把 CANN 的 `backend: pyaclnn` 改为 `backend: npu`，**保留** `backend: cpu` 节点不动。ATK 对同一 case 在 npu 与 cpu 各跑一遍。

2. **ATK 双 backend 跑出对比 xlsx**：`report/*.xlsx` 列含 `编号`（= case_id）/ `运行结果`（SUCCESS / FAILED）/ `失败原因` / `cpu_0_精度通过`（bool）/ `npu_*_精度通过`（bool，列名后缀恒为「精度通过」）/ `精度详情`。

3. **与 flow-atk 一屏决策表输入对齐**：决策表判读条件（`运行结果 = FAILED` → 3.2 功能/环境；`运行结果 = SUCCESS 且某列后缀「精度通过」=false` → 4 真精度）**完全匹配**。

4. **与 analyze-precision 列名识别规则对齐**：「凡列名以「精度通过」四字结尾即视为精度是否达标语义列」——`cpu_0_精度通过` / `npu_*_精度通过` 列名后缀 = 「精度通过」，**完全匹配**。

5. **与原 reproduce 产物定位表对齐**：原 reproduce 产出的 xlsx 列含 `运行结果` / `失败原因` / `编号` / `*精度通过` / `精度详情`——op-atk 产出 xlsx 列结构**完全一致**（ATK 工具链相同，只是 backend 从 pyaclnn 换 npu）。

6. **3.1 单算子 gate 的 `TORCH_NPU_LOGS=op_plugin` + `-wl` 单跑机制在 PTA executor 下仍成立**：flow-atk 决策表的 3.1 gate 是前置 gate（每个 `FAILED` / `SUCCESS+精度=false` 行都先过 3.1），它不消费 xlsx 列，而是另跑一次 `TORCH_NPU_LOGS=op_plugin` + `-wl '[case_id]'` 单跑 → 统计 aclnn 种类数。**pta executor 下这条机制仍成立**：pta executor 的 npu 分支调 `torch_npu.<opName>(...)` → 走 op_plugin 适配层 → 底层 aclnn 调用 → `TORCH_NPU_LOGS=op_plugin` 照样抓得到 aclnn 调用与计数。aclnn 计数口径（unique kind 去重 + 剥离辅助算子黑名单 + 正反向同名再去重，[single-op §3.1.1](../../op-delimitation-analyze-single-op/SKILL.md#311-计数规则) 为唯一权威）**不变**。

**结论**：两个前提共同保证下游零改动消费——① xlsx 列结构完全兼容（§1-5）；② 3.1 gate 的 `TORCH_NPU_LOGS=op_plugin` 单跑机制在 pta executor 下仍成立（§6）。
