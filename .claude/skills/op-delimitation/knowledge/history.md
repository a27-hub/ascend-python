# 历史案例索引（API 一致性工作流）

> **用途**：Step 3 定位时，助手 **提示用户**用 API 名 / 症状 / 结论类型在本索引与 [`examples/case-library.md`](../examples/case-library.md) 中 **自行对照**——**不**自动抓取内网表、**不**编造命中行（与 [A2/A3/A5 纪律](a2-a3-a5-discipline.md) 一致）。
> **结构**：本文件是 **精简索引 + 高频根因速查**；逐 API 的根因/结论 **详情** 见 [`examples/case-library.md`](../examples/case-library.md)（原 a2a3 712 行案例库整编）。

---

## 一、结论类型速查（结单对齐）

| 结论类型 | 典型特征 | op-delimitation 出口 |
|---------|---------|---------------------|
| 环境 / 链路 / 调度 | xlsx `FAILED` + 连接/timeout/OOM；`-mt/-to` 后 SUCCESS | functional / op-atk-adapt |
| 功能 / 拦截 / 逻辑 | `运行结果=FAILED` + aclnn/断言/NotImplemented | functional → 单算子 or 组合 |
| 真精度 | `SUCCESS` + `*精度通过=false` | analyze-precision → Dump / UT |
| 单算子（转第三方） | 去重后 aclnn **= 1**（正或反独立） | single-op early-stop → report |
| 组合类精度 | 去重后 aclnn **> 1** | precision Dump |
| 确定性 | 50 次循环不一致 / 任务 `-p` 确定性脚本 Fail | determinism_check + 结单专段 |
| 无法复现 | ≤3 轮熔断或用户确认环境不可对齐 | report 待办 + 环境侧条目 |

**历史结论分布**（来自 [case-library §一](../examples/case-library.md#一结论分类总览)，约 130 例）：非问题 ~25、用例/测试问题 ~30、算子问题（转第三方）~18、PTA 代码修复 ~25、精度标准/配置问题 ~12、CANN 修复 ~8、ATK 工具问题 ~6、环境/版本问题 ~5。

---

## 二、高频根因模式（定位时优先对照）

> 摘自 [case-library §三](../examples/case-library.md)；命中即按对应结论收口，避免重复定位。

| # | 模式 | 特征 | 解决 | 影响面 |
|---|------|------|------|--------|
| 3.1 | **测试用例输入不一致**（最常见，~25 例） | GPU 与 NPU 独立生成输入，md5/二进制不同 | 加 `-sd` 同步输入，或改用统一输入生成逻辑 | 精度 / 确定性 / 分布式 |
| 3.2 | **ReduceSum / 归约精度**（~15 例） | 反向 broadcast 触发 reduce_sum，CPU/NPU 浮点归约实现差异致二进制不一致 | 改双标杆或 `cv_fused_double_benchmark` | Tensor.add/div/fill_、where 等反向 |
| 3.3 | **小算子拼接累计误差**（~10 例） | API 由多个小算子拼接，fp16/bf16 误差逐步累积 | 已有结论，按 Float16/Bfloat 精度泛化处理 | sin / rsqrt / pow 等反向 |
| 3.4 | **Inplace + 确定性冲突**（~6 例） | `inplace=True` 修改输入，多轮执行结果不一致 | 用例避免原地修改，或 ATK clone 输入 | leaky_relu / logit_ / embedding / ctc_loss |
| 3.5 | **ACLNN 算子版本/缺失**（~8 例） | CANN 包未含对应 aclnn 算子或版本不匹配走错路径 | 更新 CANN 版本或等待算子合入 | chunk_cat / TripletMarginLoss / cdist |
| 3.6 | **PTA 适配代码 Bug**（~10 例） | meta 注册 / 参数透传 / 校验逻辑 / shape 推导有误 | 改 op-plugin 或 pytorch 仓代码 | grouped_matmul / fusion_attention / LSTM / histc / linear |
| 3.7 | **ATK 工具 Bug**（~6 例） | ATK 在确定性测试中处理 inplace / 梯度累计 / 输入读取有缺陷 | 更新 ATK 或 workaround | logit_ / stack 反向 / amp_update_scale |

---

## 三、本仓库已收录案例（按 DTS）

| DTS 单号 | API / 场景 | Gate 0 | 结论大类 | 详解 |
|---------|-----------|--------|---------|------|
| —（issue-5 端到端范例） | `binary_cross_entropy_with_logits` | A · ATK | 内存一致性 + 算子 | （step1 内存 / step2 定位 / step5 结论全产物，历史范例） |

> 约 130 条历史 API 根因/结论详情：[`examples/case-library.md`](../examples/case-library.md)（功能 / 精度 / 确定性 / 内存 / 分布式 / 边界值 分类）。

---

## 四、按症状检索（给用户的关键词）

| 症状关键词 | 优先 Skill / reference | 已收录案例 |
|-----------|------------------------|-----------|
| 全量 FAIL + 单跑 PASS | single-op 内存踩踏 · ATK#655（用户粘贴要点） | — |
| `SUCCESS` + 精度列 false | analyze-precision · [dump-analysis](../references/dump-analysis.md) | case-library §2.2 |
| ReduceSum / 反向归约二进制不一致 | 双标杆 · [atk-tuning-cases](../references/atk-tuning-cases.md) | case-library §2.2（add/div/fill_/where 反向） |
| 小算子拼接 fp16/bf16 误差 | 精度泛化（不解决） | case-library §2.2（sin/rsqrt/pow 反向） |
| inplace 确定性飘 | determinism_check · [gotchas §3](gotchas.md#3-inplace-操作破坏确定性验证) | case-library §2.3 |
| 内存一致性 >5% | [gotchas §1](gotchas.md) | case-library §2.4 |

---

## 五、新增案例约定（团队维护）

在 **`examples/`** 追加条目时，每条至少包含：

1. **DTS 单号** + **Gate 0 分支** + **API 名**
2. **表象**（用户一句话描述）与 **证据签名**（xlsx 行 / tee 原文片段）
3. **根因**（1–3 句，含关键路径/参数）
4. **结论类型**（§一 表之一）
5. **关键命令**（用户粘贴骨架，含 `-mt/-to/-sd` 若适用）
6. **产物路径**（真实 atk_output / `analyze/`）

追加后在本文件 **§三** 表格增一行链接即可；**勿**在 `knowledge/` 重复粘贴长命令块。逐 API 详情统一沉淀到 [`examples/case-library.md`](../examples/case-library.md)。

---

## 六、与 flow 的衔接

| 阶段 | 如何使用本索引 |
|------|---------------|
| flow-atk 一屏决策卡壳 | 给用户 **建议检索词**（API + 症状列） |
| analyze-functional 归类犹豫 | 指向 §四 症状 → 已收录案例 |
| analyze-precision / dump 定界 | 先查 §二 ReduceSum / 小算子拼接 / 内存模式是否命中 |
| report 5.end | 「历史参考」段写 **DTS 单号 + 本节链接**，不写「表中第 N 行命中」除非用户粘贴 |
