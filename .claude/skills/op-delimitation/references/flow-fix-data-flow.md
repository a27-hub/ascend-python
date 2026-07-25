# flow-fix 修复子流程 · 数据流与架构

> 本文档描述 **op-delimitation（定界）→ flow-fix（诊断）→ op-adaptation（执行）→ 回归** 的完整数据流、各 skill 边界、数据传递字段与边界状态机。
>
> 配套：主编排 [`skills/op-delimitation-flow-fix.md`](../skills/op-delimitation-flow-fix.md)（**handoff 字段 / 回归报告 / 5 项红线均内联**在 §阶段 3、§阶段 5）。本套件无独立的 handoff/regression/red-lines 文件。

## 定位

flow-fix 是**定界之后的次级路由**（不是 Gate 0 出口）。它补上 op-delimitation（日志级定界）与 op-adaptation（执行级修改）之间的**代码级诊断**——否则 handoff 的修改方向无人填，op-adaptation 只能盲修。

```
op-delimitation(定界·日志级) ──► flow-fix(诊断·代码级) ──► op-adaptation(执行·实现级)
   给算子名+调用点               读码定位缺陷+推导修复方向      照方案改+编+测
```

## 三段粒度分工（图例）

| 段 | skill | 粒度 | 能 | 不能 | 边界状态 |
|---|---|---|---|---|---|
| **定界** | flow-atk ◄原有 | **日志级** | 算子名 + 调用点 + 症状 + 排除环境/用例 | 代码层级（适配层 vs aclnn 底层）、具体缺陷行 | 远端只读·分析态 |
| **诊断** | **flow-fix 阶段 1** ★ | **代码级** | 读 op-plugin 源码定位具体缺陷 + 推导修复方向 + 三出口确诊 | 改码（只读） | 只读分析态 |
| **执行** | op-adaptation(MODIFY) ◄原有 | **实现级** | 按 modification_description 改 cpp/yaml/meta + 编译 + 算子 test | 从症状诊断（要明确方案才动手） | 本地可写·修复态 |

诊断段是三段之间的**桥**：`modification_description` 由它产出（有据可依），op-adaptation 拿明确方案才动手。

## 数据流图（全流程）

```text
[任务输入]
  API名 / atk task 命令 / 失败用例 / heaven日志 / 产物根
   │
   ▼
┌─── orchestrator ◄原有 ───────────────────────────┐
│  Gate 0 二出口分流 + 开场契约                       │
└───────────────────────────────────────────────────┘
   │ Gate 0
   ├── 极小出口 ──────────────────────────────► 礼貌退出
   └── 分支 A · ATK (atk task/case) ──► flow-atk
                           │
                           ▼
┌─── flow-atk ◄原有 ───────────────────────────┐
│  阶段3 ATK 执行 → 阶段4 归类 → 阶段4.1 定界 → 阶段5 源码锚点│
│  ┌── 产出 ────────────────────────────────────────────┐      │
│  │ 分析结论.md:                                          │      │
│  │   • 定界标签(单算子/组合类精度/真精度/功能·环境)      │      │
│  │   • 主责 aclnn 名                                     │      │
│  │   • 调用点锚点 opapi/Xxx.cpp:行  (P0>P1>P2)          │      │
│  │   • 数值证据(max diff / dump首分叉)                  │      │
│  │ 运行记录.md: 原始 atk task 命令(atk task)        │      │
│  └─────────────────────────────────────────────────────┘      │
│  边界状态: 远端只读·分析态                                    │
│  粒度: 日志级 (能→算子名+调用点+症状; 不能→代码层级/缺陷行)   │
└───────────────────────────────────────────────────────────────┘
                          │ 定界结论
                          │ (算子名 + 调用点 + 症状 + 数值证据)
                          ▼
┌─── flow-fix ◄核心 ───────────────────────────────────────────┐
│  次级路由 · 定界初筛                                            │
│  ┌──────────────────────────────────────────────────┐         │
│  │ 真精度(默认)/环境/用例/超时/调度/A2A3 ──► 不进     │         │
│  │ 单算子 / aclnn类 / 组合类 / (真精度疑似dtype) ► 进│         │
│  └──────────────────────────────────────────────────┘         │
│                                                                │
│  ▼ 阶段 1 · pta 缺陷诊断  ★核心 (只读分析态)                   │
│  ┌──────────────────────────────────────────────────┐         │
│  │ ① discover --op-name  定位算子在 yaml/cpp/meta 全貌│        │
│  │ ② Read 调用点 cpp 函数体 (函数体读完, 非只看锚点行) │         │
│  │ ③ 缺陷分析 (对照 cpp-guide 标准结构):              │         │
│  │     参数处理 / dtype分支 / shape推导 / TORCH_CHECK │         │
│  │     / aclnn变体选择 / 正反向binding / _out_变体    │         │
│  │ ④ 对比参考:                                        │         │
│  │     CPU 参考(cpu_op_exec) + 算子.md/_def.cpp +     │         │
│  │     数值证据(哪步开始偏)                           │         │
│  │ ⑤ 三出口判定                                       │         │
│  └──────────────────────────────────────────────────┘         │
│  ┌── 产出 ────────────────────────────────────────────┐       │
│  │ • 根因层级(确诊,非日志猜):                          │       │
│  │     op_plugin_adapter / aclnn_cann / precision     │       │
│  │ • 诊断结论: 缺陷类型 + 位置(函数+行) + 对照偏差     │       │
│  │ • 修复方向: change type + modification_description │       │
│  └─────────────────────────────────────────────────────┘       │
│  边界状态: 只读分析态 (不改码)                                 │
│  粒度: 代码级 (能→定位缺陷+推导方向+三出口; 不能→改码)         │
└───────────────────────────────────────────────────────────────┘
        │ 三出口
        ├── b aclnn/CANN 底层 ─────► 转 CANN 团队 (结单, 不切边界)
        ├── c 精度发散 ───────────► 标精度超范围 (结单, 不切边界)
        └── a 适配层缺陷明确 ─────┐
                                   │
                                   ▼
┌─── flow-fix 阶段 2 · 边界切换 + 仓准备 ─────────────────────┐
│  boundary = READONLY_RELEASED (解除只读分析态)               │
│  env = LOCAL_DEV_NPU                                          │
│  本地 op-plugin 仓干净检查 (git status --porcelain 空)       │
│  编译环境就绪 (ci/build.sh + NPU 在位)                        │
│  边界状态: 只读分析态 ──► 本地可写修复态 (仅本地仓)          │
└──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─── flow-fix 阶段 3 · handoff 包（字段内联于 flow-fix §阶段3）┐
│  落盘: handoff/根因分析与修复交接.md                          │
│  ┌── 必填 ────────────────────────────────────────────┐      │
│  │ • 根因层级 + 诊断结论 (阶段1产出, 原值搬入)          │      │
│  │ • modification_description ★(阶段1诊断产出, 非人工拍)│      │
│  │ • op_name / aclnn_op / op_api                       │      │
│  │ • 源码锚点 (调用点 + 诊断定位的缺陷行)              │      │
│  │ • 数值证据块 / boundary / env                       │      │
│  │ • 回传占位: checklist_path/summary_path/            │      │
│  │   files_modified/build_status/test_status/         │      │
│  │   c_test_triggered                                  │      │
│  └─────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────┘
                                   │ handoff 包
                                   ▼
┌─── op-adaptation (MODIFY) ◄原有·未改 ───────────────────────┐
│  M-discover → M-scope → M-gate → M-execute(C-cpp) → C-build │
│  ┌── 产出 ────────────────────────────────────────────┐      │
│  │ • CHECKLIST.md @仓根                                │      │
│  │ • summary @notes/summary_{date}.md                  │      │
│  │ • files_modified[]                                  │      │
│  │ • 新编译 wheel (dist/torch_npu-*.whl)               │      │
│  └─────────────────────────────────────────────────────┘      │
│                                                                │
│  C-test (按 scope 派生) ──────► ┌── op-test ◄原有·未改 ────┐  │
│  自动拉起 (强制 Skill 工具)      │ 算子级 UT 自检             │  │
│  纯 bug fix 不含 test 才不拉     │ 产出: UT 文件 + 执行结果   │  │
│                                 └────────────────────────────┘  │
│  回填 handoff 占位字段 (追加回同一文档, 双向链接)             │
│  边界状态: 本地可写修复态                                      │
│  粒度: 实现级 (能→照方案改+编+测; 不能→从症状诊断)           │
└───────────────────────────────────────────────────────────────┘
                                   │ 新 wheel + files_modified
                                   ▼
┌─── flow-fix 阶段 5 · 回归验证（主控 = op-delimitation）──────┐
│  ① 装新 wheel                                                  │
│  ② 原始 atk task 命令本地重跑 ── tee regression_full_*.log           │
│  ③ 近邻用例验证 (条件触发: 接口变更/共享实现)                  │
│  ④ 5 项红线自检（内联于 flow-fix §阶段5）                     │
│     CPU兜底 / 未掩盖 / scope越界 / 根因一致 / 结果落NPU       │
│  ┌── 产出 ────────────────────────────────────────────┐       │
│  │ 回归报告.md（骨架内联于 flow-fix §阶段5）:          │       │
│  │   • PASS/FAIL + 循环记录(round)                     │       │
│  │   • files_modified / c_test_triggered               │       │
│  └─────────────────────────────────────────────────────┘       │
│  边界状态: 本地可写修复态                                      │
└───────────────────────────────────────────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
     原用例仍FAIL /          近邻新引入回归 /         修复红线未过
     修复未生效              (接口/共享实现)         (如反CPU兜底)
            │                      │                      │
            └──────────┬───────────┴──────────────────────┘
                       │ FAIL (round += 1)
                       ▼
              ┌────────────────────────────────────────┐
              │ round ≤ 3 ──► 回 op-adaptation 阶段4   │
              │              再修 (按诊断方向)          │
              │                                        │
              │ round > 3 ──► 熔断: report 分支C       │
              │              ·未完全修复·               │
              │                                        │
              │ 多轮修不掉 ──► 回阶段1 重诊             │
              │ 同一缺陷     (可能层级判错,实为 aclnn   │
              │              底层 → 改判出口 b 转 CANN) │
              └────────────────────────────────────────┘
                       │ PASS
                       ▼
┌─── report 分支 C ◄原有(加分节) ──────────────────────────────┐
│  结单                                                          │
│  ┌── 产出 ────────────────────────────────────────────┐       │
│  │ • 分析结论.md (定界+诊断结论+修复位置)              │       │
│  │ • 运行记录.md (原始命令+边界切换记录)               │       │
│  │ • 回归报告.md (PASS 证据+files_modified)           │       │
│  └─────────────────────────────────────────────────────┘       │
└───────────────────────────────────────────────────────────────┘
```

## 关键数据传递（字段级）

```text
①  op-delimitation(定界) ──► flow-fix
   定界标签 / 主责 aclnn 名 / 调用点(opapi/Xxx.cpp:行) / 症状 / 数值证据 / 原始 atk task 命令

②  flow-fix 阶段1(诊断) ──► 阶段3 handoff
   根因层级(op_plugin_adapter/aclnn_cann/precision) / 诊断结论(缺陷+位置) /
   修复方向(change type + modification_description)

③  flow-fix ──► op-adaptation(handoff 包, 双向)
   → op_name / modification_description(诊断产出) / 源码锚点 / 数值证据 / boundary
   ← 回填: files_modified / build_status / test_status / c_test_triggered /
          checklist_path / summary_path

④  op-adaptation C-test ──► op-test
   scope 派生的 test 需求 ──► UT 文件 + 执行结果

⑤  op-adaptation ──► flow-fix 阶段5(回归)
   新编译 wheel / files_modified

⑥  flow-fix 阶段5 ──► report
   回归报告(PASS/熔断) / 循环记录 / files_modified
```

## 边界状态机（贯穿全流程）

```text
远端只读·分析态 ──────────────────────────────────────────► 远端只读·分析态
  (orchestrator / flow-atk / flow-fix阶段1诊断)   ↓ 仅在阶段1出口 a
                                                                    │确诊 op_plugin_adapter
                          ┌─────────────────────────────────────────┘
                          ▼
                    本地可写·修复态 ──────────────────────────────► 本地可写·修复态
                 (flow-fix阶段2/3 / op-adaptation / 阶段5回归)         │ PASS
                                                                      ▼
                                                                 结单(report)

  铁律: 可写态仅限本地 op-plugin 仓; 远端全程只读, 修复红线恒生效
        修复仅限 op-plugin 适配层; aclnn/CANN 底层转 CANN, 精度超范围
```

## 六阶段一览

| 阶段 | 主编排一句 | 边界状态 | 正文权威 |
|------|------------|---------|-----------|
| **1** | 代码级诊断：discover + Read 函数体 + 缺陷分析 + 对比 CPU 参考 + 三出口确诊 | 只读分析态 | flow-fix §阶段 1 + cpp-guide + operator-aclnn-analysis |
| 2 | 确诊适配层后边界切换；本地 op-plugin 仓干净；定位 project_root | 只读 → 可写 | flow-fix §阶段 2 + hard-constraints |
| 3 | 填 handoff（modification_description = 诊断产出） | 可写 | flow-fix §阶段 3（handoff 字段内联） |
| 4 | cwd 切仓；调 op-adaptation MODIFY 全流程；C-test 按 scope 派生 | 可写 | op-adaptation/SKILL.md + modify-guide.md |
| 5 | 装新 wheel；原始 atk task 命令重跑；近邻条件触发；5 项红线自检 | 可写 | flow-fix §阶段 5（回归报告 + 红线内联） |
| 6 | 填回归报告；回 report 分支 C 结单 | 可写 | report/SKILL.md |

## 修复范围（铁律）

- ✅ **可修**：`op_plugin_functions.yaml` / `derivatives.yaml` / `op_plugin/ops/opapi/*.cpp` / `_meta_registrations.py` / test / schema·allowlist（= op-adaptation 生成范围）。
- ❌ **不修 · 转 CANN 团队**：Ascend C 算子 / aclnn 底层实现（阶段 1 诊断出口 b）。
- ❌ **不修 · 超范围**：精度问题（阶段 1 诊断出口 c）。
- **不得触及**：CANN toolkit / aclnn 实现 / Ascend C kernel / 上游 PyTorch（`test/` 适配除外）。

## 文件清单（收敛后）

| 类型 | 文件 | 作用 |
|---|---|---|
| **核心** | `skills/op-delimitation-flow-fix.md` | 修复子流程主编排，**自包含**（handoff 字段 §阶段3 + 回归报告 + 5 项红线 §阶段5 均内联） |
| **核心** | `skills/op-delimitation-orchestrator/SKILL.md` | 套件总调度：Gate 0 + 次级路由 + 结单门控 |
| **核心** | `SKILL.md` | 套件总入口与导航 |
| **核心** | `report/op-delimitation-report/SKILL.md` | 结单出口（分支 C 接修复） |
| **核心** | `knowledge/hard-constraints.md` | 跨流程硬约束（边界切换状态机，单一权威） |
| **可选** | `CLAUDE.md` / `AGENTS.md` | 平台 slash / skill 发现表 |
| **可选** | `references/flow-fix-data-flow.md` | 本架构参考文档 |
| **未改** | `op-adaptation` / `op-test` / `flow-atk` | 保持独立可用 |

> 原 `templates/fix-handoff.md` · `templates/regression-report.md` · `knowledge/fix-red-lines.md` 的内容已**内联进 flow-fix §阶段 3 / §阶段 5**；`stages/flow-fix.md`(symlink) 已删除（冗余）。
