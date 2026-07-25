---
name: op-delimitation-analyze-precision
description: >-
  Use for ATK-only precision analysis when report xlsx rows are successful but
  precision columns fail, including precision detail review, save_data
  artifacts, dump/UT replay, and output_grad evidence.
---

# 4 / 4.1 — 真精度问题（ATK-only）

> **分层阅读**：先做 [`skills/op-delimitation-flow-atk.md`](../../skills/op-delimitation-flow-atk.md)；**两遍**后仍无法在精度路径落地 → Read **本 SKILL**。  
>
> **前提**：orchestrator Gate 0 已通过；输入为 **ATK 跑**得到的 `report/*.xlsx`。
>
> **入口条件（与 3.2 互斥）**：xlsx 行 `运行结果 == SUCCESS` 且**存在**至少一列、列名**后缀为**「精度通过」的值为 `FALSE` / `Failed`。若该行 `运行结果 == FAILED` → 回 [3.2 functional](../op-delimitation-analyze-functional/SKILL.md)，不要硬套本章。

## 列名识别（统一规则，不要写死单一列名）

报表中精度列常带设备/场景前缀：`cpu_0_精度通过`、`npu_npu_xxx_精度通过` 等。

**统一规则**：凡列名 **以「精度通过」四字结尾** 即视为"精度是否达标"语义列。读其布尔 / `PASS`/`FAIL` / `True`/`False` / `Failed` 等值。

筛选伪代码（参考，pandas）：

```python
import pandas as pd

df = pd.read_excel("<path>/report/*.xlsx")
acc_cols = [c for c in df.columns if str(c).endswith("精度通过")]

def is_failed_val(v):
    if isinstance(v, bool):
        return v is False
    s = str(v).strip().lower()
    return s in {"false", "failed", "fail", "否", "0"}

mask_success = df["运行结果"].astype(str).str.upper().eq("SUCCESS")
mask_acc_fail = df[acc_cols].apply(lambda col: col.map(is_failed_val)).any(axis=1)
precision_rows = df[mask_success & mask_acc_fail]
```

若表头为英文/其它变体，列出实际列名请用户映射一次再固化。

## 步骤 4 — 从 xlsx 定位真精度行

1. 读取 `atk_output/<CaseSet>_<ts>/report/*.xlsx`（路径由用户指定；不得臆造）。
2. 按上节规则筛出 `precision_rows`。
3. 对每个命中行记录：
   - `编号` → 精度失败的 **case_id**；
   - `精度详情`（或等价列）→ **逐行摘要**，保留关键数值差异描述。
4. 进入 **4.1**。

## 步骤 4.1 — 精度定位决策顺序

**按序尝试**，命中即进下一步，不命中再走下一项：

1. **复用 3.1 单跑产物**：若 [3.1 single-op](../op-delimitation-analyze-single-op/SKILL.md) 已对该 case 跑过 → 直接读取 `single-run/` 下**单跑汇总表**。
   - 单跑 PASS（批跑 FAIL） → 结合 [ATK#655](https://open.codehub.huawei.com/innersource/atk_G/ATK/issues/655) 说明内存踩踏嫌疑。
   - 单跑 FAIL → 继续下一步。
2. **A2 / A3 对照**：**不接入**。列出**建议检索词**（case 名 / `case_json` / API 名 / `精度详情` 关键片段）让用户自查；助手不代查、不编造命中。
3. **多 aclnn（组合类精度）**：用本 Skill 自带的 **`scripts/dump_npu_cpu.py`**（`TorchDispatchMode`）在 **NPU 与 CPU 各跑一遍** 目标 API，落 **两份 CSV + 逐算子 `.pt`**，对比首个统计量分叉的算子以定界。完整操作见下文 **「Dump：算子级定界」**。
   - 内网 **dump / ACLNN 映射** wiki（若团队有归档）：`https://wiki.huawei.com/domains/158084/wiki/321817/WIKI2026031710434355` —— **默认助手不可读 URL**（见 [hard-constraints · 统一口径](../../knowledge/hard-constraints.md#统一口径--外部资源获取方式)）；补充步骤须 **用户粘贴** 正文，URL 仅写入「外部参考」存档；不得凭 URL 或旧对话编造命令。
4. **无 aclnn**：走 **UT 复现**。在 3.1 单跑基础上加 `--save_data input --save_data output`：

   ```bash
   unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2 && \
   atk task -c <case_json> -n <yaml> -wl '[<case_id>]' \
     --save_data input --save_data output
   ```

   取 `atk_output/.../input/` 与 `output/`（含 `output_grad_0.pt`），用 [`scripts/ut_forward.py`](scripts/ut_forward.py) / [`ut_backward.py`](scripts/ut_backward.py)；**确定性**用 [`scripts/determinism_check.py`](scripts/determinism_check.py)（50 次循环，结单数值见 [`templates/numeric-evidence.md`](../../templates/numeric-evidence.md)）。  
   - CPU vs NPU **梯度不一致** → 记录 **max diff 数值** → 再转 Dump。  
   - 50 次确定性不一致 → **非确定性问题**；检查 [gotchas · inplace](../../knowledge/gotchas.md#3-inplace-操作破坏确定性验证)。

## Dump：算子级定界（组合类 API · 精度）

> **全文**：[`references/dump-analysis.md`](../../references/dump-analysis.md) — 数据准备、`dump_npu_cpu.py` 定制、CSV/`.pt` 分析、存档与交付。本节只保留入口条件。

**适用**：xlsx 真精度入口 + **多条 aclnn（组合实现）**；单算子（=1）→ 回 [single-op](../op-delimitation-analyze-single-op/SKILL.md) early-stop，**不必 Dump**。

**脚本**：`tools/op-delimitation-analyze-precision/scripts/dump_npu_cpu.py`

**结单**：首个分叉算子名 + **max diff 数值** + CSV/`.pt` 路径（禁止仅写 Mean 定性结论）。

## 精度对比脚本（标杆选择）

> 选型表与参数详见 [`atk-tuning-cases`](../../references/atk-tuning-cases.md)；脚本位于 `tools/op-delimitation-analyze-precision/scripts/`。

| 场景 | 标杆 | 脚本 |
|------|------|------|
| NPU 输出 vs CPU/GPU 升精度标杆 | 单标杆 | `scripts/single_compare.py` |
| NPU vs GPU 双侧对比 + 升精度标杆 | 双标杆 | `scripts/double_benchmark_compare.py` |
| 全面统计（max/avg re ratio、rmse ratio、ULP 分布） | 新精度标准 | `scripts/new_benchmark_compare.py` |
| CV 融合类算子 | 双标杆 | `scripts/cv_fused_double_benchmark_compare.py` |
| NPU vs GPU 内存数据对比 | — | `scripts/memory_compare.py` |
| 共用精度计算工具函数 | — | `scripts/utils.py` |

**ReduceSum/归约反向**：单标杆高精度标准不通过时优先改双标杆或 `cv_fused_double_benchmark`（见 [history §二 · 3.2](../../knowledge/history.md)）。

## 其它参考

- **内存踩踏**（ATK 批跑 FAIL / 单跑 PASS）：[ATK#655](https://open.codehub.huawei.com/innersource/atk_G/ATK/issues/655)。**URL 仅存档**，要点由用户粘贴（同 [hard-constraints · 统一口径](../../knowledge/hard-constraints.md#统一口径--外部资源获取方式)）；链接写入 md 的「外部参考」段。

## 结单最小字段（移交 [report](../../report/op-delimitation-report/SKILL.md)）

- 命中行列表：`case_id` / 命中的**精度列名** / `精度详情` 摘要。
- 定位路径走到了 4.1 哪一步、结论是什么（单跑复用 / dump / UT / 内存踩踏）。
- 复现脚本、`input.bin` / `output_grad_0.pt` 等素材的真实路径。
- 外部 wiki / issue 的**摘要 + URL**。

## 原则

- **不写死** `cpu_0_精度通过`；**以列名后缀匹配**。
- **不臆造** sheet 名、阈值、`精度详情` 内容。
- 表头不一致时：列出实际表头请用户映射一次，并把映射结果更新到会话上下文（不改本 skill 正文）。
