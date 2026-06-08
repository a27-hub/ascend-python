# CHANGELIST — 变更记录

> 记录项目演进和变更，便于后期回溯和审查。
> **只追加，不删除已有条目。**

---

## 变更记录格式

```
### YYYY-MM-DD | 变更标题

**类型**: 🆕 新功能 | 🔧 修改 | 🐛 修复 | 🗑️ 删除 | 📝 文档
**范围**: 影响范围

#### 变更内容
- 具体变更了什么

#### 关键决策
- 做了什么决策，为什么

#### 影响范围
- 对现有功能的影响
```

---

### 2026-06-05 | 工作流初始化

**类型**: 📝 文档
**范围**: .workflow/ 目录

#### 变更内容
- 初始化 GOAL.md：定义项目背景、总体目标、阶段拆解、约束与成功标准、6 个待确认问题
- 初始化 NEXT_STEPS.md：记录当前任务索引（任务 1 进行中，任务 2 待开始）
- 初始化 CHECKLIST.md：定义阶段一质量检查项（YAML 配置、C++ 适配、Schema 测试、Python 测试、代码风格）
- 新增 raw/docs/01_codebase_analysis.md：代码库分析（仓库定位、标准流程、相似算子分析、接口推导）
- 新增 raw/docs/02_yaml_schema_design_draft.md：YAML Schema 设计草案（3 种方案对比、C++ 适配草案）
- 更新 raw/INDEX.md：登记新增的 docs 和 design 文档索引

#### 关键决策
- 将工作范围限定为 op-plugin 仓库的 PyTorch 接口封装层，不涉及底层 ACLNN kernel 开发
- 基于参考提交 c2a961c90fc 和 6c6718f5 总结出 4 文件标准交付模式
- 基于已有 npu_qkv_rms_norm_rope_cache 算子推导新算子的 PyTorch schema
- 识别出 6 个需要用户确认的关键问题（Q1-Q6）

#### 影响范围
- 不影响现有代码，仅初始化工作流文件
- 等待用户确认后将进入 PLAN.md 规划和实现阶段

---

### 2026-06-05 | 需求确认与 PLAN.md 初始化

**类型**: 📝 文档
**范围**: .workflow/GOAL.md, .workflow/PLAN.md, .workflow/NEXT_STEPS.md

#### 变更内容
- 用户确认 Q1-Q6 全部 6 个待确认问题
- 更新 GOAL.md：将"待确认问题"替换为"已确认决策"（D1-D6）
- 更新 GOAL.md：目标拆解增加 functional 变体说明
- 初始化 PLAN.md：详细的 4+3 任务实现计划，含 YAML 内容、C++ 实现要点、JSON 内容、测试场景
- 更新 NEXT_STEPS.md：任务 1 标记已完成，任务 2 展开为 7 个子任务

#### 关键决策
- **D5 确认使用 EXEC_NPU_NO_FORMAT_CHECK_CMD**：基于代码库分析，所有 KV cache 类算子均使用此宏
- **D6 确认使用 apply_tensor_without_format + TensorWrapper**：基于代码库中量化类算子的 fp8 输出创建模式
- **D2 确认需要 functional 变体**：clone in-place 参数，返回 5 个输出
- **D3 确认版本范围 [v2.7, newest]**
- **D4 确认不需要 check_aclnn_kernel_available**：ACLNN 接口在开发中，按需求文档交付

#### 影响范围
- PLAN.md 已就绪，可立即进入实现阶段
- 不影响现有代码

---

### 2026-06-05 | PyTorch 接口封装实现完成

**类型**: 🆕 新功能
**范围**: op_plugin/config, op_plugin/ops/opapi, test/core_tests, test/test_custom_ops

#### 变更内容
- 新增 `op_plugin/ops/opapi/QkvRmsNormRopeCacheWithKScaleKernelNpuOpApi.cpp`（171 行）
  - in-place 函数 `npu_qkv_rms_norm_rope_cache_with_kscale`
  - functional 函数 `npu_qkv_rms_norm_rope_cache_with_kscale_functional`
  - 共享验证逻辑 `validate_inputs` 辅助函数
- 修改 `op_plugin/config/op_plugin_functions.yaml`：新增 2 条 func 定义
- 修改 `test/core_tests/torch_npu_OpApi_schema_all.json`：新增 2 条 schema 注册
- 新增 `test/test_custom_ops/test_npu_qkv_rms_norm_rope_cache_with_kscale.py`（119 行）
  - 2 个正常路径测试 + 6 个报错路径测试

#### 关键决策
- 使用 `validate_inputs` 辅助函数避免 in-place 和 functional 之间的验证代码重复
- 使用 `TensorWrapper{tensor, aclDataType::ACL_FLOAT8_E4M3FN}` 包装 fp8 输出和 in-place cache tensor
- functional 变体 clone 3 个 in-place 参数（k_cache, v_cache, k_scale_cache），返回 5 个输出
- Python 测试仅验证 shape/dtype/error 行为，不实现 golden 计算（golden 由 kernel 侧负责）

#### 影响范围
- 新增 2 个文件，修改 2 个文件
- 不影响已有算子功能
- CHECKLIST.md 阶段一全部检查项通过
