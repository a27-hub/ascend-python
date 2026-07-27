---
name: op-delimitation-flow-fix
description: >-
  Secondary routing AFTER demarcation (not Gate 0). Bridges op-delimitation
  (日志级定界: 算子名 + 调用点 + 症状) and op-adaptation (执行级修改) with a 代码级诊断
  phase: read op-plugin source to locate the specific defect, derive the fix
  direction, and triage to 适配层 / aclnn底层 / 精度. Then boundary switch →
  handoff (modification_description = 诊断产出) → op-adaptation(MODIFY) 按方案执行 →
  回归验证 with the original repro command on local NPU. Without this diagnosis
  phase op-adaptation would 盲修; op-adaptation never starts on its own.
---

# flow-fix 主流程 — 主编排（分支 C · 修复子流程）

> **不是 Gate 0 出口**。flow-fix 是 **定界之后的次级路由**：`flow-atk` 走到 **阶段 4.1 定界** 后，由 **定界标签初筛** 决定是否进入本流程。Gate 0 二出口（A · ATK / 极小出口）**不变**。
>
> **前后关系（务必遵守）**：**op-delimitation 是唯一主控与决策者，一以贯之**。本流程在 **阶段 1 诊断确诊为 op-plugin 适配层缺陷** 后，才下沉调用 `op-adaptation(MODIFY)`（地位同 `op-atk-adapt` / `analyze-*`），**永不自行启动**。修复完由 **op-delimitation 自己** 接手回归验证。
>
> **本流程的核心价值**：补上 **代码级诊断**。op-delimitation 只定到「算子名 + 调用点」（日志级，看不到 aclnn 内部）；op-adaptation 要「明确的修改方案」才动手（执行级）。中间「读 op-plugin 源码定位具体缺陷、推导修复方向、确诊是不是适配层能修」由本流程 **阶段 1** 承担——**否则 handoff 的修改方向无人填，op-adaptation 只能盲修**。
>
> **环境**：修复 + 回归在 **本地 NPU 开发环境**（本地可写 op-plugin 仓 + 编译环境）。远端只读边界在本流程改码阶段不生效（操作本地仓）；修复红线恒生效——见 [hard-constraints · 边界切换](../knowledge/hard-constraints.md#边界切换--只读分析态--本地可写修复态) 与 §阶段 5 修复红线自检。

<a id="fix-phase-overview"></a>

## 必读顺序与本层边界

| 次序 | Read |
|------|------|
| 0 | 已完成：[orchestrator Gate 0](op-delimitation-orchestrator/SKILL.md#gate-0硬门槛先过再继续) + 开场契约 + 分支 A 主编排 + **阶段 4.1 定界** |
| 1（本文件） | **当前 md**：次级路由 + **六阶段地图（含阶段 1 诊断）** + 边界切换 + 回归循环 |
| 2（下游 skill） | [`op-adaptation/SKILL.md`](../../op-adaptation/SKILL.md)（MODIFY 模式）+ [`references/guides/modify-guide.md`](../../op-adaptation/references/guides/modify-guide.md) |
| 3（诊断方法论） | [`op-adaptation/cpp-guide.md`](../../op-adaptation/references/guides/cpp-guide.md)（标准结构 + 找相似算子）· [`operator-aclnn-analysis.md`](../references/operator-aclnn-analysis.md)（算子语义） |
| 结束 | [`op-delimitation-report`](../report/op-delimitation-report/SKILL.md) **分支 C 模板分节** |

> handoff 字段规范、回归报告格式、5 项修复红线均已**内联**到 §阶段 3 / §阶段 5，不再单列独立文件。

**本文件不写**：op-adaptation M-* 步骤细节（权威在 `modify-guide.md`）、UT 生成（权威在 `op-test`）、近邻用例选取（权威在 issue-pipeline `verify.md`）——避免双源漂移。

---

## 次级路由触发条件（定界 → 进诊断）

定界结论的权威产出点：分支 A 主编排的 **阶段 4.1**，见 [`analyze-single-op`](../tools/op-delimitation-analyze-single-op/SKILL.md) / [`analyze-precision`](../tools/op-delimitation-analyze-precision/SKILL.md)（分支 A）。

> **修复范围 = pta 适配层 = op-adaptation 生成范围**（`op_plugin_functions.yaml` / `derivatives.yaml` / `opapi/*.cpp` / `_meta_registrations.py` / test / schema·allowlist）。**不在范围**：Ascend C 算子 / aclnn 底层实现（转 CANN 团队）、精度问题（超出范围）。详见 [§修复范围（铁律）](#修复范围铁律)。

### 定界初筛（是否进阶段 1 诊断）

| 定界标签 | 进阶段 1 诊断？ |
|---|---|
| 单算子（3.1 语义） | 是 |
| 功能·环境 → aclnn 类失败（`aclnn*` + `failed` / `EZ*` / `Operator execute failed`；**aclnn 存在但调用失败**） | 是 |
| **aclnn 算子未实现 / 不存在**（`not implemented` / `no aclnn implementation` / `未找到` / `NotImplementedError` / `aten::xxx not impl`） | **否**（直接转 CANN，不进诊断——aclnn 底层算子缺失，op-plugin 适配层无 aclnn 可接，非"接入缺陷"） |
| 组合算子·组合类精度 | 是（诊断定首个分叉是否适配层缺陷） |
| 真精度（数值对标失败） | 默认否（精度超范围）；**除非疑似适配层 dtype/cast 缺陷** → 进诊断复核 |
| 功能·环境 → 环境/驱动/依赖、用例脚本/配置、超时/OOM、调度、A2/A3 已知 | 否（原 report 结单） |

> **根因层级（适配层 / aclnn 底层 / 精度）不在入口判**——那要读 op-plugin 源码才能确诊（op-delimitation 日志级给不出，见 [§能力边界](#能力边界)）。初筛通过 → 进 **阶段 1 诊断** → 由诊断定三出口。**op-adaptation 不反向触发、不参与入口判定。**

---

## 「两遍规则」→ 全文 Read 下游

> 与 ATK 对齐：先在本文件 + 已掌握的定界结论上尝试收敛；**同一决策点两轮**仍无法决定（如：诊断定位不到具体缺陷 / modification_description 细不到能派生 change type / 回归失败归因不清）→ **必须 Read** [`op-adaptation/references/guides/modify-guide.md`](../../op-adaptation/references/guides/modify-guide.md) **全文** 再执行。

**典型决策点**：诊断是否真定位到 op-plugin 具体缺陷、修复方向是否对应明确 change type、回归失败是「修复未生效」还是「根因层级判错（实为 aclnn 底层）」、循环是否到 3 次上限。

---

## 输入契约（从定界阶段继承，阻塞诊断）

进入 flow-fix 前，以下必须已就绪（来自分支 A 主编排的产物）：

| 字段 | 来源 | 用途 |
|---|---|---|
| **定界标签**（初筛通过） | `分析结论.md` 定界块 | 进阶段 1 诊断的判据 |
| **主责 aclnn / API 名** | 同上 | 阶段 1 `discover --op-name` + 函数体定位 |
| **调用点锚点**（`opapi/Xxx.cpp:行`） | 阶段 5 源码对照 | 阶段 1 Read 函数体起点 |
| **原始 atk task 命令**（ATK `atk task`） | `运行记录.md` | 阶段 5 回归重跑 |
| **数值证据块**（UT / Dump / 确定性） | [`templates/numeric-evidence.md`](../templates/numeric-evidence.md) | 阶段 1 对比参考 + 阶段 5 回归基线 |
| **op-plugin 仓路径**（阶段 1 只读；阶段 2 起要可写） | 用户确认 | 阶段 1 Read 源码 + 阶段 4 project_root（`find_repo_root` 靠它） |

缺「主责算子名 / 调用点锚点 / 原始 atk task 命令」→ **阻塞**，列清单问用户；不得臆造路径或用替代命令冒充原始 atk task 命令。

---

## 六阶段路线图

```text
阶段1 pta 缺陷诊断（只读分析态） ★核心
      → 三出口：
         a 适配层缺陷明确 → 推导修复方向 → 阶段2
         b aclnn/CANN 底层 → 转 CANN（不切边界，结单）
         c 精度发散 → 标超范围（不切边界，结单）
阶段2 边界切换 + 仓准备（确诊适配层后才切可写）
      → 阶段3 产出 handoff 包（modification_description = 诊断产出）
      → 阶段4 调用 op-adaptation(MODIFY)  [下游]
            └ M-discover → M-scope → M-gate → M-execute → C-build
            └ C-test 按 scope 派生自动拉 op-test（算子级自检）
      → 阶段5 回归验证（op-delimitation 自己跑回原始 atk task 命令，本地 NPU + 新编译 wheel）
            ├ FAIL → 回阶段4 再修（循环 ≤3）
            └ PASS
      → 阶段6 移交 report（含修复记录 + 回归证据）
```

| 阶段 | 主编排一句 | 正文权威 |
|------|------------|-----------|
| **1** | **代码级诊断**：`discover` 定位 + Read 调用点函数体 + 缺陷分析 + 对比 CPU 参考 + 三出口确诊 | 本文件 §阶段 1 + [`cpp-guide`](../../op-adaptation/references/guides/cpp-guide.md) + [`operator-aclnn-analysis`](../references/operator-aclnn-analysis.md) |
| 2 | 确诊适配层后声明 `boundary=READONLY_RELEASED` + `env=LOCAL_DEV_NPU`；确认本地 op-plugin 仓干净；定位 project_root | 本文件 §阶段 2 + [hard-constraints · 边界切换](../knowledge/hard-constraints.md#边界切换--只读分析态--本地可写修复态) |
| 3 | 填 handoff 包（字段见 §阶段 3）：`op_name` / **`modification_description`（诊断产出）** / 源码锚点 / 数值证据 / 根因层级 | 本文件 §阶段 3（handoff 字段内联） |
| 4 | cwd 切到 op-plugin 仓；调 `op-adaptation` MODIFY 全流程；C-test 按 scope 派生 | [`op-adaptation/SKILL.md`](../../op-adaptation/SKILL.md) + [`modify-guide.md`](../../op-adaptation/references/guides/modify-guide.md) |
| 5 | 装新编译 wheel；取 **原始 atk task 命令** 本地重跑；近邻用例条件触发；跑 5 项红线自检（见 §阶段 5） | 本文件 §阶段 5（回归报告 + 红线内联） |
| 6 | 填回归报告（修复记录 + 回归证据 + files_modified）；回 report **分支 C 分节** 结单 | [`report`](../report/op-delimitation-report/SKILL.md) |

---

## 阶段 1 · pta 缺陷诊断（只读分析态）★核心

> 补上 op-delimitation（日志级定界）与 op-adaptation（执行级修改）之间的 **代码级诊断**：读 op-plugin 源码定位**具体缺陷**、推导**修复方向**、三出口**确诊根因层级**。**全程只读，不改码**——在只读分析态完成，确诊适配层可修才进阶段 2 切边界。
>
> **为什么必须这一步**：op-delimitation 日志只打到 op_plugin 的 `EXEC_NPU_CMD` 层，看不到 aclnn 内部 → 判不出「参数传错（适配层）vs aclnn 算错（底层）」；op-adaptation 要明确 `modification_description` 才动手。没有诊断段，handoff 的修改方向无人填 → op-adaptation 盲修。

**输入**：定界结论（算子名 + 调用点 + 症状 + 数值证据）+ op-plugin 仓（本地可读；或远端 `remote_torch_npu_dir` 只读）。

**步骤**：

1. **discover 定位算子全貌**（只读 grep）：`op_adapt.py discover --op-name <算子名>` → 算子在 yaml / cpp / meta / test 的现状表；与 op-delimitation 调用点锚点交叉验证。
2. **Read 调用点函数体**：读 `op_plugin/ops/opapi/Xxx.cpp` 主责函数（+ `_meta_registrations.py` 的 meta + `derivatives.yaml` 的 binding，如涉及）；**函数体读完**，不只看锚点行。
3. **缺陷分析**（对照 [`cpp-guide`](../../op-adaptation/references/guides/cpp-guide.md) 标准结构与 pattern）逐项查：
   - 参数处理（dtype / shape / optional 传递）是否正确
   - dtype 分支是否缺失（某 dtype 走错路径 / 未覆盖）
   - shape 推导（`KernelNpuOutputSize`）是否正确
   - `TORCH_CHECK` 约束是否缺失（脏输入进 aclnn）
   - aclnn 变体选择是否正确（`DO_COMPATIBILITY` / 用错 aclnn）
   - 正反向 binding（`derivatives`）是否接对
   - `_out` / `_` 变体是否覆盖
4. **对比参考**：CPU 参考实现（op-test 的 `cpu_op_exec` / `supported_op_exec`，或 PyTorch 原生算子语义）+ 算子 `.md` / `_def.cpp` 语义 + 数值证据（哪一步开始偏）。

**三出口判定**（写进 handoff 的根因层级 + 诊断结论）：

| 出口 | 判据 | 处理 |
|---|---|---|
| **a 适配层缺陷明确** | 步骤 3 定位到 op-plugin 文件的**具体缺陷**（参数 / dtype / shape / check / binding / 变体） | 推导 **change type + modification_description**（改哪个函数、怎么改）→ 进阶段 2 |
| **b aclnn / CANN 底层** | 步骤 3 确认 op-plugin 包装逻辑 + 参数**均正确**，问题在 aclnn 本身：**算子未实现/不存在**（`not implemented` / `no aclnn implementation` / `未找到` / `NotImplementedError`）**或** 返回错 / 算错（`EZ*` / 结果错） | **不切边界**，结单标「转 CANN 团队」+ 附诊断依据 |
| **c 精度超范围** | 数值精度发散、步骤 3 **无**适配层逻辑缺陷 | **不切边界**，结单标「精度，超范围」+ 附诊断依据 |

**产出**（阶段 3 落盘进 handoff）：

- **根因层级**（确诊，非日志猜）：`op_plugin_adapter`（含具体缺陷描述）/ `aclnn_cann` / `precision`
- **诊断结论**：缺陷类型 + 位置（函数 + 行）+ 对照参考的偏差
- **修复方向**：change type（bug fix / interface change / 加分支 / 补变体）+ `modification_description`（op-adaptation M-scope 直接消费）

> **诊断方法论引用**（不另起炉灶）：[`cpp-guide`](../../op-adaptation/references/guides/cpp-guide.md)（标准结构 + 找相似算子——判断"正确实现该长啥样"）、[`operator-aclnn-analysis.md`](../references/operator-aclnn-analysis.md)（算子语义 / 正反向）、op-test 的 `cpu_op_exec` / `supported_op_exec`（CPU 参考实现）。
>
> **反幻觉**（同 op-delimitation 阶段 5 纪律）：缺陷必须落在 **Read 过的函数体具体行 + 对照参考的偏差**；**不得**只凭 grep 命中或症状猜下结论。**诊断不出具体缺陷 → 默认走出口 b/c，不硬填 `modification_description` 让 op-adaptation 盲修。**

---

## 阶段 2 · 边界切换 + 仓准备

**前置**：阶段 1 诊断出口 = **a（op_plugin_adapter）**。出口 b/c 不进本阶段（直接结单）。

1. **声明边界切换**（写入 `运行记录.md` 新增「修复子流程 · 边界切换」段）：
   - `boundary: READONLY_RELEASED`（解除只读分析态）
   - `env: LOCAL_DEV_NPU`（后续操作在本地 NPU 开发环境）
   - 切换时刻 + 由谁确认。
2. **确认本地 op-plugin 仓**（op-adaptation 硬前提）：
   - 用户给出可写 op-plugin 仓绝对路径；`ls <path>/op_plugin/config/op_plugin_functions.yaml` 只读确认存在。
   - **仓干净性检查**（M-gate 只 warn 不阻断，但为让阶段 5 的 `git diff` 可干净分离修复，本流程要求干净）：`git -C <path> status --porcelain` 为空；非空 → 提示用户先 commit/stash 或在新 worktree/分支跑，**不擅自 `git stash`**。
3. **编译环境就绪**：C-build 不可跳过，确认本地 `ci/build.sh` 可用 + NPU 在位（`python -c "import torch;print(torch.npu.is_available())"`）。

> 远端只读边界在本阶段及之后 **不生效**（操作对象是本地 op-plugin 仓，不是远端 DTS 环境）。但 **不得** 借此回远端做任何造改删——远端在 flow-fix 全程保持只读。

---

## 阶段 3 · 产出 handoff 包

落盘到 `{opName}_pta_atk/handoff/根因分析与修复交接.md`。**handoff 模式照搬 issue-pipeline**：上游（op-delimitation + 阶段 1 诊断）写定界与修复方向 → 下游（op-adaptation）读它作 M-discover / M-scope 输入 → 结果**追加回同一文档**。所有锚点 / 路径 / 数值必须来自真实证据，**不得臆造**。

**必填字段**（7 段）：

**§0 边界与元信息**

```yaml
issue_id:                  # 可选；多 case 归并用
ticket_root:               # {opName}_pta_atk/
boundary: READONLY_RELEASED   # 阶段 2 声明（解除只读分析态）
env: LOCAL_DEV_NPU
op_plugin_repo:               # 本地可写 op-plugin 仓绝对路径（op-adaptation project_root）
switch_at:                    # 边界切换时刻 + 确认人
```

**§1 定界结论**（来自 `分析结论.md` 定界块，原值搬入）

- 定界标签（单算子 3.1 / 组合类精度 / 真精度 / 功能·环境→aclnn类，写明主次）
- **根因层级**（阶段 1 诊断产出，**必填**）：`op_plugin_adapter`（进本包）/ `aclnn_cann`（转 CANN，不进）/ `precision`（超范围，不进）
- 主责 aclnn / API（按失败子用例逐项；传导链写明根因节点 vs 从属）
- 计数与依据（`TORCH_NPU_LOGS=op_plugin` 日志路径 + `analyze/` 计数稿）

**§2 op-adaptation 输入**（M-discover / M-scope 消费）

```yaml
op_name:                  # 无 _vN 后缀；= op_adapt.py discover --op-name <此值>
aclnn_op:                 # 定界锁定的主责 aclnn 名
op_api:                   # torch API
modification_description: # ★阶段 1 诊断产出的修复方向（非人工拍/非 adaptation 猜）：change type + 改哪个函数怎么改；须细到 M-scope 能派生文件集
change_type_guess:        # interface change / add variant / backward supplement / bug fix(no interface)（M-scope 复核）
```

> `modification_description` 示例（接口变更须参数级 diff）：`interface change: forward 参数 tokens: Tensor → tokens_size_0: int + tokens_dtype: ScalarType；影响 op_plugin_functions.yaml / opapi/Xxx.cpp / _meta_registrations.py / derivatives.yaml(rebind)`

**§3 源码锚点**（M-discover grep 交叉验证）

```yaml
cpp_anchor:    # op_plugin/ops/opapi/XxxKernelNpuOpApi.cpp:行 + 片段（调用点 + 诊断定位缺陷行）
yaml_anchor:   # op_plugin/config/op_plugin_functions.yaml:行
meta_anchor:   # op_plugin/python/meta/_meta_registrations.py:行
derivatives:   # 是否涉及 derivatives.yaml rebind（是/否/待 M-scope 问用户）
```

**§4 数值证据**（原样搬 [`templates/numeric-evidence.md`](../templates/numeric-evidence.md)；回归对比以此为基线）

| 项目 | 结果 | 备注 |
|------|------|------|
| UT 正向/反向 max diff | | `ut_forward`/`ut_backward.py` |
| allclose | PASS/FAIL | rtol/atol 若自定义须写明 |
| 确定性 50 次 | PASS/FAIL | 首次分叉轮次 / Max diff |
| Dump 首个分叉算子 / max diff | | op_name / `.pt` 全量对比 |
| 日志路径 | | `运行日志/...` 或 `analyze/...` |

**§5 原始 atk task 命令**（阶段 5 回归重跑用，来自 `运行记录.md`，**原样粘贴禁止改写/缩小**）+ golden / 期望行为 + 失败用例编号。

**§6 op-adaptation 回填**（阶段 4 末由 op-adaptation 产出写入）

```yaml
checklist_path:    # <op_plugin_repo>/CHECKLIST.md（Modify Mode 段）
summary_path:      # <op_plugin_repo>/notes/summary_YYYY-MM-DD.md
files_modified: [] # summary §4 File List
build_status:      # C-build 通过/失败
test_status:       # C-test 结果（PASS/FAIL/SKIP-N/A）
c_test_triggered:  # bool；op-test 是否按 scope 派生拉起
verify_report:     # op_adapt.py verify JSON 摘要
```

**§7 预期 scope**（供 M-scope 校验，非强制）

- 预期改动文件类型（cpp/yaml/meta/derivatives/test/json，基于 change_type_guess）
- **Scope Control**：op-adaptation 只改 M-scope 确认文件；执行中发现新依赖须暂停重新派生（见 [`op-adaptation/modify-guide.md`](../../op-adaptation/references/guides/modify-guide.md)）。

> op-adaptation 自身产物（CHECKLIST.md@仓根、notes/summary_*.md）通过 §6 双向链接，不强行统一目录。

---

## 阶段 4 · 调用 op-adaptation(MODIFY)（下游）

**前置**：cwd 已在 op-plugin 仓内（否则 `find_repo_root` 失败）；handoff 包就绪（`modification_description` 已由阶段 1 诊断给出，op-adaptation 据此 M-scope，不猜）。

1. **M-discover**：用 handoff 的 `op_name` 跑 `op_adapt.py discover --op-name <op_name>`，拿 grep 状态报告；与 handoff 的源码锚点交叉验证。
2. **M-scope**：基于 `modification_description` + grep 结果派生 change type 与受影响文件清单；处理 **derivatives rebind 问题**（须问用户）；呈现 scope 请用户确认。
3. **M-gate**：`git status --porcelain`（阶段 2 已保证干净，此处复核）。
4. **M-execute**：按 scope 文件 → C-* 步骤子集执行（C-verify / C-build / C-summary 必跑；C-yaml / C-cpp / C-meta / C-derivatives 按 scope）。
5. **C-test**：**按 scope 派生**——interface change / add variant / backward supplement 等 change type 会 **自动拉起 op-test**（算子级 UT 自检，强制走 Skill 工具，不手写）。纯 bug fix 不改接口且 scope 不含 test 时才不拉。**flow-fix 不干预 C-test 的派生**，但回归报告须记录 `c_test_triggered`。
6. **回填 handoff 占位字段**：`checklist_path` / `summary_path` / `files_modified[]`（取 summary §4）/ `build_status` / `test_status` / `c_test_triggered`。

> op-adaptation 的产物落在 **op-plugin 仓内**（CHECKLIST.md@仓根、notes/summary_*.md）；op-delimitation 的产物落在 **pta 目录**。两者通过 handoff 文档双向链接，**不**强行统一。

---

## 阶段 5 · 回归验证（op-delimitation 主控）

> 这是「修复后重新验证」的真正落点——**由 op-delimitation 自己跑回原始 atk task 命令**，不是 op-adaptation 的 op-test UT（那只验证算子本身）。借鉴 issue-pipeline [`stages/verify.md`](../../torch-npu-issue-pipeline-core/stages/verify.md)。

1. **装新编译 wheel**：`pip install dist/torch_npu-*.whl`（本地开发环境，非远端；用户授权后操作）。
2. **原用例回归**：从 `运行记录.md` 取 **原始 atk task 命令**（ATK `atk task`），本地 NPU 上 **原样重跑**，`tee` 到 `{opName}_pta_atk/运行日志/regression_full_*.log`。
   - **通过标准**：退出码 0 + 无 `FAILED`/`Error`/`RuntimeError` + 数值证据块（与修复前对比，max diff 收敛）。
   - **禁止**用替代命令、缩小输入、skip 用例来制造通过。
3. **近邻用例验证**（条件触发）：公共路径改动 / 共享底层实现 / 接口变更 → 选 2–5 个近邻变体（不同 dtype/layout）逐个跑；出现 `FAIL（新引入）`= 回归，立即停止。
4. **修复红线自检**（5 项，**全部通过** 方可进阶段 6；搬自 issue-pipeline `hard-red-lines` + `npu-op-integration` 反 CPU 兜底，适配到本地 op-plugin 仓 `$OP_PLUGIN_REPO`）：

   - **检查 1 · 反 CPU 兜底（核心红线）**：`git -C "$OP_PLUGIN_REPO" diff --name-only | while read f; do grep -nE "\.cpu\(\)|to\(['\"]cpu['\"]\)|to_device.*cpu" "$OP_PLUGIN_REPO/$f"; done` —— **命中即拦**。NPU 路径不得把张量搬 CPU 算完搬回；同款反模式（mock/fake/常量返回、`@skipIf(is_npu)`、`os.environ["NPU_DISABLE_*"]` 等让 NPU 不真实参与计算的绕路）一并拦。
   - **检查 2 · 未掩盖**：`git -C "$OP_PLUGIN_REPO" diff -- '**/disabled_testcases.json' '**/test_*'` —— 新增 `disabled_testcases` / `skipIf(is_npu)` / `pytest.skip` 须确认必要，**禁止把 skip/disable 当修复收尾**。
   - **检查 3 · scope 无越界**：`git -C "$OP_PLUGIN_REPO" diff --stat` —— 全部在 M-scope 确认文件内、与本次定界修复直接相关、无无关变更；**适配层边界**：改动文件只能是 yaml / `opapi/*.cpp` / meta / derivatives / test / schema·allowlist，**不得触及** CANN/aclnn 底层（否则根因层级判错，回阶段 1 重判）。
   - **检查 4 · 根因-修复一致**：修复方式匹配定界标签（单算子→改对应 opapi 主责算子；组合类→修首个分叉；功能·环境→修 aclnn 调用路径，非 monkey-patch）。
   - **检查 5 · 结果落 NPU**：`python -c "import torch,torch_npu; ...; assert r.device.type=='npu'"` —— 修复后结果张量在 NPU，不得回落 CPU。

**回归报告**（落盘 `{opName}_pta_atk/回归报告.md`，骨架）：

- **摘要**：定界标签 / 修复轮次 round `__`/3 / 回归结论（PASS · FAIL-循环中 · FAIL-熔断）
- §1 修复版本就绪：本地仓 / 新 wheel 路径 / 安装记录 / `torch.npu.is_available()`
- §2 原用例回归：原始命令 / tee 日志 / 退出码 / 错误片段 / 数值对比（修复前→后 max diff）
- §3 近邻用例（条件触发）：用例 / 退出码 / 分类（PASS / FAIL 新引入 / FAIL 原有 / SKIP）
- §4 红线自检：5 项勾选
- §5 修改文件清单（来自 handoff §6 回填）
- §6 循环记录（round > 1 时）：每轮修复方向 / 失败现象 / 归因
- §7 移交 report：分支 C（已修复 / 未完全修复）

**回归失败处理**（写进回归报告）：

| 失败现象 | 处理 |
|---|---|
| 原用例仍 FAIL（修复未生效） | 回阶段 4 让 op-adaptation 再修（计入循环） |
| 近邻 `FAIL（新引入）`= 回归 | 回阶段 4 缩小改动范围（计入循环） |
| 修复红线未过（如反 CPU 兜底） | 回阶段 4 重选修复方向（计入循环） |

---

## 回归循环状态机（≤3 次）

```text
阶段4 修复 ──► 阶段5 回归 ──PASS──► 阶段6 移交 report
                  │
                  └──FAIL──► 回阶段4（round += 1）
                              │
                              └── round > 3 ──► 熔断：记录待办，移交 report（标记「未完全修复」）
```

- **循环上限 3 次**（借鉴 issue-pipeline `diagnose↔verify` 3 次；语义同：跨轮次意味着根因判断可能错了）。
- op-adaptation 内部按 scope 派生的 C-test 自检、以及级联算子修复 **不计入** 本 3 次（那是单次进入 op-adaptation 的内部细节）。
- **多轮 FAIL 且每轮都修不掉同一缺陷 → 回阶段 1 重诊**（可能根因层级判错，实际是 aclnn 底层；阶段 1 改判出口 b 转 CANN）。
- 熔断 ≠ 静默失败：必须在回归报告写明「已尝试 N 轮修复方向 + 最后一轮失败现象」，进 report **分支 C · 未完全修复** 分支。

---

<a id="修复范围铁律"></a>

## 修复范围（铁律）

**只修 op-plugin 适配层**（= op-adaptation 生成范围）：

- ✅ **可修**：`op_plugin_functions.yaml`（注册缺失/错）、`derivatives.yaml`（binding 错）、`op_plugin/ops/opapi/*.cpp`（aclnn 包装 bug：参数传递、dtype 分支、shape 推导、TORCH_CHECK、`_out`/`_` 变体、DO_COMPATIBILITY）、`_meta_registrations.py`、test、schema/allowlist。
- ❌ **不修 · 转 CANN 团队**：Ascend C 算子、aclnn 底层实现（CANN toolkit 内 aclnn C++/kernel）。op-plugin 包装正确但 aclnn 本身报错/算错 → 不进 op-adaptation，结单标「转 CANN」。
- ❌ **不修 · 超范围**：精度问题（数值精度发散）→ 结单标「精度，超出 pta 适配修复范围」。

op-adaptation 改的文件**只能**是上述适配层文件；**不得触及** CANN toolkit / aclnn 实现 / Ascend C kernel / 上游 PyTorch（`aten/`/`c10/`/`torch/`，`test/` 适配除外）。**转 CANN / 超范围由阶段 1 诊断出口 b/c 决定**，不靠日志猜。

---

<a id="能力边界"></a>

## 能力边界（三段分工）

| 段 | 负责 | 粒度 | 能 / 不能 |
|---|---|---|---|
| **定界** | op-delimitation | 日志级 | **能**：算子名 + 调用点 + 症状 + 排除环境/用例。**不能**：代码层级（适配层 vs aclnn 底层）、具体缺陷行 |
| **诊断** | flow-fix 阶段 1 | 代码级 | **能**：读 op-plugin 源码定位具体缺陷 + 推导修复方向 + 三出口确诊。**不能**：改码（只读） |
| **执行** | op-adaptation(MODIFY) | 实现级 | **能**：按 modification_description 改 cpp/yaml/meta + 编译 + 算子 test。**不能**：从症状诊断缺陷（要明确方案才动手） |

诊断段是三段之间的**桥**——没有它，定界（日志级）和执行（要方案）对不上，op-adaptation 盲修。

---

## 原则

1. **前后关系铁律**：op-adaptation **永不自行启动**——只能由本流程在「**阶段 1 诊断确诊 `op_plugin_adapter` + handoff 就绪**」后调用；修复完由 op-delimitation 接手回归。**任何让 op-adaptation 反向触发或绕过诊断的路径都是反模式。**
2. **诊断不可跳**：阶段 1 诊断是 flow-fix 的存在理由——跳过诊断 = handoff 修改方向无人填 = op-adaptation 盲修。诊断不出具体缺陷 → 走出口 b/c，**不强修**。
3. **不臆造、不替代**：源码锚点 / op_name / 修复方向必须来自诊断段真实证据；回归必须用 **原始 atk task 命令**，不得用替代命令 / 缩小输入 / skip 制造通过。
4. **远端只读边界状态相关**：flow-fix 阶段操作 **本地** op-plugin 仓，远端只读边界 **不生效**；但 **不得** 借此回远端造改删，远端全程只读。
5. **修复红线恒生效**：反 CPU 兜底（`grep -nE "\.cpu\(\)|to\(['\"]cpu['\"]\)"` 命中即拦）、禁改上游 PyTorch（`aten/`/`c10/`/`torch/`，`test/` 适配除外）、根因-修复一致性——见 §阶段 5 修复红线自检。
6. **不超循环**：≤3 次；多轮修不掉同一缺陷 → 回阶段 1 重诊（可能层级判错）；熔断后如实记录，不假装修复。

---

## 信息不足时

列缺失项 **[待]**（典型：主责算子名 / 调用点锚点缺失 → 无法诊断；本地 op-plugin 仓路径未给；编译环境未就绪；原始 atk task 命令与定界对不齐；NPU 不在位）。**可先做阶段 1 诊断**（只读，读本地或远端源码即可推进），但 **不得** 在缺编译环境 / 缺原始 atk task 命令时进阶段 4–5。
