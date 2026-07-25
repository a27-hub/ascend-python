# 分析 Gotchas（API 一致性 · 高频陷阱）

> **适用**：`flow-atk`、op-atk-adapt / analyze-* / report 全链路。  
> **纪律**：命中任一条 → 暂停当前归类，按本节「正确做法」修正后再继续；完整反模式见 [`anti-patterns.md`](anti-patterns.md)。

---

## 1. GPU 双标杆 `nodes.yaml` 占位值未替换

**现象**：ATK 双标杆任务连不上 GPU、NPU 端拉 GPU 输出 **404**。

**陷阱**：`nodes_*.yaml` 里的 **GPU host / port / output_path** 常为占位或过期；**output_path 必须与 GPU 端 `atk server` 工作目录一致**，否则 NPU 无法下载 GPU 侧产物。

**正确做法**：用户粘贴 yaml 片段后，按 [`references/gpu-nodes-config.md`](../references/gpu-nodes-config.md) 逐项替换；跳板场景配合 。**禁止**助手凭 URL 或旧对话推断 host。

---

## 2. Dump / CSV 的 Mean 不可作最终裁决

**现象**：CSV 里某算子 Mean「看起来接近」，但精度仍失败。

**陷阱**：正负相消、全零附近、统计舍入会让 Mean 误导定界。

**正确做法**：Mean **仅导航**；**最终依据**为对应 `.pt` 的 `torch.allclose` / `max(abs(a-b))`。见 [`references/dump-analysis.md`](../references/dump-analysis.md)。

---

## 3. inplace 操作破坏确定性验证

**现象**：50 次循环前几次一致、后续分叉；或 UT Pass 但 ATK 确定性 Fail。

**陷阱**：`leaky_relu_(True)`、`logit_`、`embedding(max_norm=...)` 等 **原地改输入**；同一份 `input.bin` 被多次 load 后已被污染。

**正确做法**：`determinism_check.py` / 任务 `-p` 脚本中 **每次 run_once 重新 load**；检查用例脚本是否 inplace；结单写清是否排除 inplace。

---

## 4. GPU/NPU 输入不同步导致「假精度问题」

**现象**：离线 UT 或双标杆对比 max diff 很大，但 ATK 加 `-sd` 后通过。

**陷阱**：GPU 与 NPU **各自随机生成**输入，md5 不同。

**正确做法**：ATK 复跑叠加 **`-sd`（`--save_data`）** 同步输入；UT 使用 `--save_data` 产物而非现场 randn。见 [op-atk-adapt · `--save_data` 产物约定](../tools/op-atk-adapt/SKILL.md#save_data--sd-产物约定)。

---

## 5. 反向含 ReduceSum 却用单标杆判精度

**现象**：`Tensor.add/div/fill_`、`torch.repeat_interleave` 等反向涉及 **ReduceSum 归约**，单标杆高精度下误报 FAIL。

**陷阱**：把「单标杆 PASS/FAIL」当成唯一判据。

**正确做法**：反向路径优先 **双标杆** 或团队约定的 CV 融合双标杆流程；结单注明标杆类型，勿写「精度不达标」而无标杆说明。

---

## 6. fp16 / bf16 小算子拼接累计误差

**现象**：组合 API 在 fp16/bf16 下 UT 边缘 FAIL，fp32 通过。

**陷阱**：`sin/rsqrt/pow` 等反向由多个小算子拼接，低精度 **逐步放大**。

**正确做法**：记录 dtype；必要时升精度对照或走 Dump 定界首个分叉算子；勿单用 Mean 一句话结单。

---

## 7. 用例 `range=(-inf, inf)` 导致精度失败

**现象**：AdamW、addmm、bmm 等在大范围输入下精度不达标。

**陷阱**：把框架精度问题与 **用例参数不合理** 混为一谈。

**正确做法**：对照 用例 JSON / `-p` 脚本中的 **input range**；若用户粘贴显示极端 range，结单倾向 **用例/标杆问题** 并建议收窄范围验证。

---

## 8. ATK Fail + UT Pass 时未排除 `-p` 自定义标杆

**现象**：xlsx FAILED 或精度列失败，但 UT 白盒 Pass。

**陷阱**：任务 `-p <python_script>` 引入 **与标准标杆不一致** 的自定义逻辑。

**正确做法**：在 [`references/atk-tuning-cases.md`](../references/atk-tuning-cases.md) 纪律下，尝试 **去掉 `-p` 重跑 ATK**（保留 `-c/-n` 等不可改字段）；仍 Fail 再读源码 / Dump。结单记录是否做过「去 `-p` 对照」。

---
