# 开场契约 · 字段清单与 stamp（单点权威）

> **编排器入口**：[`skills/op-delimitation-orchestrator/SKILL.md`](../skills/op-delimitation-orchestrator/SKILL.md) §开场契约  
> **stamp 模板**：[`templates/stamp-table.md`](../templates/stamp-table.md)

---

## 字段清单（用户可一次粘齐或多轮散落给）

下面这张对照表列出**单次任务开场契约的全部字段**——用户可以一次性把全部值粘过来，也可以多轮散落给（助手负责把散落输入归位到这张 schema，并在 stamp 表里可视化已齐 / 缺失 / 冲突状态）。**不要求**用户必须按 YAML 形式贴；**纯自然语言粘贴**也接受。

**两条纪律**：① 字段值仍以**用户粘贴的纯文本**为唯一来源（不接受 URL 抓取 / OCR / 智能推断，遵循 [hard-constraints · 统一口径](hard-constraints.md#统一口径--外部资源获取方式)）；② 助手做"语法对齐 + 缺失检测 + 跨字段值相等校验"，**不**做语义推断。

| 字段 | 必填 / 可选 | 跨字段校验对照（值相等）|
|------|-------------------|----------------------|
| CANN ATK 用例目录路径 | 必填 · 阻塞 | vs 目录下应有 `atk.yaml`/`generate.py`/`executor.py`/`node.yaml` 4 件套，或 `case.json`/`executor.py`/`node.yaml` 3 件套（预生成用例）|
| op-plugin 仓根路径 | 必填 · 阻塞 | vs 仓根下应有 `CHECKLIST.md` + `codegen/templates/_op_plugin_docs.py` |
| 目标 opName | 可选 | vs `CHECKLIST.md` 中 `torch_npu function name:` 行；vs `_op_plugin_docs.py` 中 `_add_torch_npu_docstr("<opName>", ...)` |
| NPU 设备 id | 可选（默认 0；远端时为远端 NPU id） | — |
| pta 目录（{opName}_pta_atk/） | 必填 · 阻塞 | — |
| **远端 SSH 字段**（host/user/port/key_path、容器 runtime/name/镜像、set_env.sh 路径、remote_torch_npu_dir/version）| **窄条件必填**（仅当 op-plugin 仓 + NPU 同在远端时阻塞；默认本地跳过）| vs `start_train_container.sh -i <image>` / `source <set_env.sh>`；auto_adapt.py 要求仓根可写，atk task 要 NPU 同机（见 [hard-constraints · 本地同机约束](hard-constraints.md)）|

> ⚠ **行号为现状快照**——实施时以**段落名**定位，不依赖行号。
> ⚠ **不支持**「本地建 pta + 远端跑 NPU」——`auto_adapt.py` 在仓根建 `{opName}_pta_atk/`，`atk task` 要在同机跑，二者必须同机。
> ⚠ xlsx 路径 / tee 日志路径 / ATK 跑前基线为**下游产物/环境字段**（跑完后回填或默认 export），非开场阻塞输入，详见 [op-atk-adapt](../tools/op-atk-adapt/SKILL.md)。

---

## 跨字段一致性校验（机制 · 不强制清单）

助手在生成 stamp 表前对**值相等性**做交叉校验：发现任意两个字段在用户输入里**应当相等却不相等**时，stamp 表里该项标 ⚠ 让用户复核，**不擅自改值**。

**示例**：

- `opName` = `npu_swiglu_group_quant_backward`，但 `atk.yaml` 的 `name` = `function_swiglu_group_quant_backward` —— 标 ⚠ 提醒两套命名并存（opName 来自 CHECKLIST，api_type 来自 atk.yaml，两者不同是正常的，但结单引用 xlsx 路径时按 api_type 拼）。3 件套模式无 atk.yaml：api_type 改取自 executor `@register` / case.json，⚠ 逻辑同样适用；
- `op-plugin 仓根` 路径下缺 `CHECKLIST.md` → 标 ✗ 阻塞，让用户提供或直接给 opName。

**限制**（与统一口径联动）：助手只做"语法 + 字段对齐 + 值相等"的纯文本校验，**不**做"语义推断"；冲突时**让用户决定哪个为准**，不擅自改值。

---

## 契约状态 stamp · 首轮回复必带

Gate 0 通过 + 收到（部分）开场契约后，助手**首轮回复**必须输出 stamp 表——把散落输入对齐到字段清单、可视化"已齐 / 缺失·阻塞 / 缺失·可选 / 冲突"四种状态。

**4 标记**：✓ 已收到 ｜ ✗ 缺失·阻塞（不补不进下一阶段） ｜ ◯ 缺失·可选（按下游降级规则继续） ｜ ⚠ 收到但与其它字段冲突（须用户复核）。

**模板正文**：见 [`templates/stamp-table.md`](../templates/stamp-table.md)。

**后续轮次**：不强制每轮都贴 stamp 表；但**任何一项从 ✗ 翻 ✓ / ⚠ 解到 ✓** 时，应在该轮回复**前面**加一行"契约状态变化：<字段> X → Y"，再继续执行。

**第一轮回复自检（契约前后）**：先复述 Gate 0 判定结果（**A · ATK / 极小出口**）→ 列缺失项 → 补齐前不作「已适配 / 已定结论」式表达。
