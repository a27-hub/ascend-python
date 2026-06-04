# raw/ — 原始文档索引

> 本目录保存所有原始文档，**不可修改**。其他文件是衍生物，这里是 truth source。
> 本索引帮助 agent 按需检索，实现渐进式披露。

## 目录结构

```
raw/
├── INDEX.md        ← 本文件（检索入口）
├── spec/           ← 需求规格说明书（实现无关的原理说明，视项目需求而定）
├── docs/           ← 参考文档（工作中产生的走读、分析、设计文档等）
└── design/         ← 用户建立的原始设计文档
```

## spec/ — 需求规格

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [spec/gmmfr_op.md](spec/gmmfr_op.md) | 参考算子原型设计说明书 | 需要理解参考算子原理时 |

## docs/ — 参考文档

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [docs/01_codebase_analysis.md](docs/01_codebase_analysis.md) | 代码库分析：仓库定位、新增算子标准流程、已有相似算子分析、接口推导 | 了解仓库结构和开发模式时 |
| [docs/02_yaml_schema_design_draft.md](docs/02_yaml_schema_design_draft.md) | YAML Schema 设计草案：新算子的 YAML 配置方案、C++ 适配草案、注意事项 | 确定接口 schema 和实现方案时 |

## design/ — 原始设计文档

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [design/qkv_rms_norm_rope_cache_with_kscale_design.md](design/qkv_rms_norm_rope_cache_with_kscale_design.md) | 算子完整设计文档（1660 行）：接口定义、shape/dtype、计算公式、tiling、kernel 分层、交付件 | 需要理解算子完整语义和约束时 |
| [design/qkv_rms_norm_rope_cache_with_kscale_tasks/](design/qkv_rms_norm_rope_cache_with_kscale_tasks/) | 拆分设计文档目录（8 个子文档）：算子原型、交付件纵览、golden、tiling、kernel、同步、实现拆解 | 需要某个视角的详细设计时 |

## 检索场景速查

| 我需要... | 读哪个文件 |
|-----------|-----------|
| 了解仓库结构和新增算子流程 | `docs/01_codebase_analysis.md` |
| 了解算子的数学公式和接口定义 | `design/qkv_rms_norm_rope_cache_with_kscale_tasks/01_operator_proto_and_formula.md` |
| 了解需要交付哪些文件 | `design/qkv_rms_norm_rope_cache_with_kscale_tasks/02_deliverables_overview.md` |
| 了解 YAML 配置和 C++ 适配方案 | `docs/02_yaml_schema_design_draft.md` |
| 了解完整设计约束和公式 | `design/qkv_rms_norm_rope_cache_with_kscale_design.md` |
| 查看用户的详细任务要求 | `.workflow/reference/next_steps/` 下对应文件（NEXT_STEPS.md 为索引） |
