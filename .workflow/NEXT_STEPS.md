# NEXT_STEPS — 下一步计划

> 用户任务的**索引**。记录任务概要和状态，链接到详细任务说明。
> 主 agent 通过本文件快速了解任务全貌，具体开发时再读取详细文档。
>
> **详细任务说明**存放在 [reference/next_steps/](reference/next_steps/) 目录下。

## 任务 1：初始化工作流并确认需求

**任务状态**：已完成

**简要说明**：初始化 .workflow/ 工作流文件，分析代码库和设计文档，与用户确认接口 schema 和实现方案中的关键问题。

**完成状态**:
- [x] 分析仓库代码结构和参考提交模式
- [x] 阅读设计文档，提取 PyTorch 接口约束
- [x] 初始化 GOAL.md，记录待确认问题
- [x] 记录关键发现在 raw/docs/
- [x] 用户确认 GOAL.md 中的 6 个待确认问题（Q1-Q6）
- [x] 根据确认结果初始化 PLAN.md

## 任务 2：实现 PyTorch 接口封装

**任务状态**：待开始

**简要说明**：在 op-plugin 仓库中完成 npu_qkv_rms_norm_rope_cache_with_kscale 的 PyTorch 接口封装（含 in-place 和 functional 两个变体），包括 YAML 配置、C++ 适配、schema 测试和 Python 单元测试。

**详细计划**：[PLAN.md](PLAN.md)

**完成状态**:
- [ ] 任务 1.1: YAML 配置（op_plugin_functions.yaml）— 2 条 func 定义
- [ ] 任务 1.2: C++ 适配实现（QkvRmsNormRopeCacheWithKScaleKernelNpuOpApi.cpp）— 2 个函数
- [ ] 任务 1.3: Schema 测试注册（torch_npu_OpApi_schema_all.json）— 2 条
- [ ] 任务 1.4: Python 单元测试（test_npu_qkv_rms_norm_rope_cache_with_kscale.py）
- [ ] 任务 2.1: C++ 编译检查（LSP）
- [ ] 任务 2.2: Python 测试验证
- [ ] 任务 2.3: 代码走查（CHECKLIST.md）
