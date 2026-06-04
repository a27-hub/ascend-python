# GOAL — 总体目标

> 总体目标导向文档，牵引核心方向不偏离，保证跨会话目标一致。

## 项目背景

op-plugin 是 Ascend Extension for PyTorch (torch_npu) 的算子插件库，为 PyTorch 框架开发者提供 NPU 算子库调用能力。其核心职责是将底层 ACLNN 算子接口封装为 PyTorch 自定义 API，使开发者能通过 `torch_npu.npu_xxx()` 调用 NPU 算子。

现有一个新的融合算子 `QkvRmsNormRopeCacheWithKScale` 需要添加 PyTorch 接口。该算子面向 Transformer decode/prefill 流程，在一个 kernel 内融合以下操作：
- Q/K/V 拆分
- Q/K RMSNorm
- Q/K RoPE（旋转位置编码）
- Q/K rotation matmul（Cube 右乘）
- Q/K 动态 FP8 量化
- V FP8 量化
- KV cache 写回（含 K scale cache）

首期仅支持 **arch35/Ascend950** 平台。

## 总体目标

为 `QkvRmsNormRopeCacheWithKScale` 算子在 op-plugin 仓库中封装 PyTorch 接口 `npu_qkv_rms_norm_rope_cache_with_kscale`，使用户能通过 `torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(...)` 调用底层 ACLNN 算子 `aclnnQkvRmsNormRopeCacheWithKScale`。

**范围限定**：本任务仅涉及 op-plugin 仓库的 PyTorch 接口封装层，不涉及底层 ACLNN 算子的 kernel/tiling/op_host 开发（后者由 ops-transformer 仓库负责）。

## 目标拆解

### 阶段一：PyTorch 接口封装

在 op-plugin 仓库中完成新算子的 PyTorch 接口交付（含 in-place 和 functional 两个变体），包含以下 4 个文件：

- [ ] 任务 1.1: 在 `op_plugin/config/op_plugin_functions.yaml` 的 custom 段新增 2 条 func 定义（in-place + functional）
- [ ] 任务 1.2: 新增 `op_plugin/ops/opapi/QkvRmsNormRopeCacheWithKScaleKernelNpuOpApi.cpp` C++ 适配实现（含 2 个函数）
- [ ] 任务 1.3: 在 `test/core_tests/torch_npu_OpApi_schema_all.json` 注册 2 条 schema 测试条目
- [ ] 任务 1.4: 新增 `test/test_custom_ops/test_npu_qkv_rms_norm_rope_cache_with_kscale.py` Python 单元测试

### 阶段二：验证与交付

- [ ] 任务 2.1: 确认 C++ 适配代码编译通过（LSP 检查）
- [ ] 任务 2.2: 确认 Python 测试代码语法和逻辑正确
- [ ] 任务 2.3: 代码走查，确保与已有算子风格一致

## 约束与成功标准

### 约束

1. **平台限制**：仅支持 Ascend950/arch35，C++ 适配层需显式校验 `SocVersion >= Ascend950`
2. **接口风格**：遵循仓库已有 custom 算子的封装模式（参考 `npu_qkv_rms_norm_rope_cache`、`npu_dual_level_quant_matmul` 等）
3. **可选输入处理**：`q_rotation`、`k_rotation`、`v_scale` 在 ACLNN 接口中为 optional，当前场景 checker 强制非空
4. **in-place 输出**：`k_cache`、`v_cache`、`k_scale_cache` 为输入输出复用空间（in-place）
5. **命名规范**：文件命名、函数命名遵循仓库既有规范

### 成功标准

1. YAML 配置正确注册 func schema
2. C++ 适配层完成输入校验、输出分配、ACLNN 调用
3. Schema 测试条目已注册
4. Python 单元测试覆盖基本场景（正常路径 + 报错路径）
5. 代码风格与仓库已有算子一致

## 已确认决策

### D1: PyTorch 接口 schema（已确认）

**In-place 版本**：
```
npu_qkv_rms_norm_rope_cache_with_kscale(
    Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin,
    Tensor slot_mapping, Tensor(a!) k_cache, Tensor(b!) v_cache,
    Tensor(c!) k_scale_cache, Tensor cu_seq_len, int[3] head_nums,
    *, Tensor? q_rotation=None, Tensor? k_rotation=None,
    Tensor? v_scale=None, float epsilon=1e-6
) -> (Tensor, Tensor)   // q_out, q_scale
```

**Functional 版本**：
```
npu_qkv_rms_norm_rope_cache_with_kscale_functional(
    Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin,
    Tensor slot_mapping, Tensor k_cache, Tensor v_cache,
    Tensor k_scale_cache, Tensor cu_seq_len, int[3] head_nums,
    *, Tensor? q_rotation=None, Tensor? k_rotation=None,
    Tensor? v_scale=None, float epsilon=1e-6
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)   // q_out, q_scale, k_cache, v_cache, k_scale_cache
```

### D2: 需要 functional 变体（已确认）

需要。functional 变体将 k_cache/v_cache/k_scale_cache 从 in-place 改为 const 输入，clone 后调用 ACLNN，返回所有 5 个输出。

### D3: op_api 版本范围（已确认）

`op_api: [v2.7, newest]`

### D4: ACLNN kernel 可用性（已确认）

ACLNN 接口在开发中，按需求文档开发，适配层只需直接调用，不需要 `check_aclnn_kernel_available`。

### D5: 使用 EXEC_NPU_NO_FORMAT_CHECK_CMD（已确认）

**决定：使用 `EXEC_NPU_NO_FORMAT_CHECK_CMD`。**

依据：代码库中所有 KV cache 类算子（`QkvRmsNormRopeCacheNpuOpApi`、`KvRmsNormRopeCacheNpuOpApi`、`FlashAttention` 等）均使用 `EXEC_NPU_NO_FORMAT_CHECK_CMD`。该宏跳过 format-based caching 查找，适用于涉及特殊 format tensor（如 fp8_e4m3fn cache）的复杂 attention/cache 算子。

### D6: fp8 tensor 创建方式（已确认）

**决定：使用 `apply_tensor_without_format` + `TensorWrapper`。**

依据：代码库中量化类算子（`QuantizeKernelNpuOpApi`、`RotateQuantKernelNpuOpApi`、`DynamicQuantKernelNpuOpApi`）创建 fp8 输出均采用此模式：
```cpp
at::Tensor q_out = npu_preparation::apply_tensor_without_format(
    shape, options.dtype(at::kFloat8_e4m3fn));
TensorWrapper q_out_wrapper = {q_out, aclDataType::ACL_FLOAT8_E4M3FN};
EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnOp, ..., q_out_wrapper);
```

对于 in-place 的 k_cache/v_cache（用户传入的 fp8 tensor），也需要用 TensorWrapper 包装以确保 ACL dtype 正确传递。

## 相关链接

- [实现计划](./PLAN.md)
- [用户下一步计划](./NEXT_STEPS.md)
- [检查清单](./CHECKLIST.md)
- [变更记录](./CHANGELIST.md)
- [代码库分析](./raw/docs/01_codebase_analysis.md)
- [YAML Schema 设计草案](./raw/docs/02_yaml_schema_design_draft.md)
- [算子设计文档](./raw/design/qkv_rms_norm_rope_cache_with_kscale_design.md)
- [拆分设计文档目录](./raw/design/qkv_rms_norm_rope_cache_with_kscale_tasks/README.md)
