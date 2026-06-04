# PLAN — 实现计划

> 将 GOAL.md 的总体目标拆分为可逐步执行的阶段计划。
> 与 TODO 的区别：这是具有全局视角的任务规划/编排方案，TODO 是子任务的执行计划。

## 阶段一：PyTorch 接口封装

### 任务 1.1: YAML 配置

**目标**: 在 `op_plugin/config/op_plugin_functions.yaml` 的 `custom:` 段新增 2 条 func 定义

**目标文件**: `op_plugin/config/op_plugin_functions.yaml`

**完成标准**:
- in-place 版本 schema 正确注册，包含 `Tensor(a!)`、`Tensor(b!)`、`Tensor(c!)` 标记
- functional 版本 schema 正确注册，无 `!` 标记，返回 5 个 Tensor
- `op_api: [v2.7, newest]`
- 建议插入位置：在已有 `npu_qkv_rms_norm_rope_cache` 条目（行 5879-5883）之后

**YAML 内容**:

```yaml
  - func: npu_qkv_rms_norm_rope_cache_with_kscale(Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor(c!) k_scale_cache, Tensor cu_seq_len, int[3] head_nums, *, Tensor? q_rotation=None, Tensor? k_rotation=None, Tensor? v_scale=None, float epsilon=1e-6) -> (Tensor, Tensor)
    op_api: [v2.7, newest]

  - func: npu_qkv_rms_norm_rope_cache_with_kscale_functional(Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, Tensor k_cache, Tensor v_cache, Tensor k_scale_cache, Tensor cu_seq_len, int[3] head_nums, *, Tensor? q_rotation=None, Tensor? k_rotation=None, Tensor? v_scale=None, float epsilon=1e-6) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
    op_api: [v2.7, newest]
```

### 任务 1.2: C++ 适配实现

**目标**: 新增 `QkvRmsNormRopeCacheWithKScaleKernelNpuOpApi.cpp`，实现 in-place 和 functional 两个函数

**目标文件**: `op_plugin/ops/opapi/QkvRmsNormRopeCacheWithKScaleKernelNpuOpApi.cpp`

**完成标准**:
- 文件命名遵循 `<OpName>KernelNpuOpApi.cpp` 规范
- 命名空间 `op_api`
- 平台检查 `SocVersion >= Ascend950`
- 输入校验：shape、dtype、head_nums 长度/值、可选输入非空
- fp8 输出使用 `apply_tensor_without_format` + `TensorWrapper`
- in-place 参数使用 `TensorWrapper` 包装
- 使用 `EXEC_NPU_NO_FORMAT_CHECK_CMD`
- functional 变体 clone in-place 参数后调用

**实现要点**:

1. **头文件引入**:
   ```cpp
   #include "op_plugin/OpApiInterface.h"
   #include "op_plugin/utils/op_api_common.h"
   ```

2. **平台检查**:
   ```cpp
   TORCH_CHECK(c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
       "This interface is supported only on the Ascend950 platform.",
       OPS_ERROR(ErrCode::PARAM));
   ```

3. **输入校验**（关键项）:
   - `qkv.dim() == 3`, `qkv.dtype() == kBFloat16`
   - `head_nums.size() == 3`, 所有值 > 0, `head_nums[2] == head_nums[1]` (Nv == Nk)
   - `qkv.size(1) == Nq + Nk + Nv`
   - `q_gamma/k_gamma`: shape `[D]`, dtype `kFloat`
   - `cos_sin`: shape `[MaxSeqLen, D]`, dtype `kFloat`
   - `slot_mapping`: shape `[T]`, dtype `kInt`
   - `cu_seq_len`: shape `[Batch+1]`, dtype `kInt`
   - `k_cache/v_cache`: dtype `kFloat8_e4m3fn`
   - `k_scale_cache`: dtype `kFloat`
   - `q_rotation/k_rotation`: 非空, shape `[D,D]`, dtype `kBFloat16`
   - `v_scale`: 非空, shape `[Nv]`, dtype `kFloat`

4. **输出分配**:
   ```cpp
   // q_out: [T, Nq, D], fp8_e4m3fn
   at::Tensor q_out = npu_preparation::apply_tensor_without_format(
       {T, Nq, D}, qkv.options().dtype(at::kFloat8_e4m3fn));
   TensorWrapper q_out_wrapper = {q_out, aclDataType::ACL_FLOAT8_E4M3FN};

   // q_scale: [T, Nq], fp32
   at::Tensor q_scale = npu_preparation::apply_tensor_without_format(
       {T, Nq}, qkv.options().dtype(at::kFloat));
   ```

5. **in-place 参数包装**:
   ```cpp
   TensorWrapper k_cache_wrapper = {k_cache, aclDataType::ACL_FLOAT8_E4M3FN};
   TensorWrapper v_cache_wrapper = {v_cache, aclDataType::ACL_FLOAT8_E4M3FN};
   // k_scale_cache 是 fp32，可直接传递或包装
   ```

6. **EXEC 调用**:
   ```cpp
   EXEC_NPU_NO_FORMAT_CHECK_CMD(
       aclnnQkvRmsNormRopeCacheWithKScale,
       qkv, q_gamma, k_gamma, cos_sin, slot_mapping,
       k_cache_wrapper, v_cache_wrapper, k_scale_cache, cu_seq_len,
       q_rotation_v, k_rotation_v, v_scale_v,
       head_nums, epsilon,
       q_out_wrapper, q_scale);
   ```

7. **functional 变体**:
   ```cpp
   at::Tensor k_cache_clone = k_cache.clone();
   at::Tensor v_cache_clone = v_cache.clone();
   at::Tensor k_scale_cache_clone = k_scale_cache.clone();
   // 调用 ACLNN，返回 (q_out, q_scale, k_cache_clone, v_cache_clone, k_scale_cache_clone)
   ```

**参考文件**:
- `op_plugin/ops/opapi/QkvRmsNormRopeCacheNpuOpApi.cpp` — 最相似的算子，结构模板
- `op_plugin/ops/opapi/KvRmsNormRopeCacheNpuOpApi.cpp` — TensorWrapper 用法参考
- `op_plugin/ops/opapi/DualLevelQuantMatmulKernelNpuOpApi.cpp` — 平台检查 + fp8 TensorWrapper 参考

### 任务 1.3: Schema 测试注册

**目标**: 在 `test/core_tests/torch_npu_OpApi_schema_all.json` 注册 2 条 schema

**目标文件**: `test/core_tests/torch_npu_OpApi_schema_all.json`

**完成标准**:
- in-place 和 functional 两条 func schema 字符串与 YAML 完全一致
- version 数组为 `["v2.7", "newest"]`

**JSON 内容**:
```json
  "func: npu_qkv_rms_norm_rope_cache_with_kscale(Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor(c!) k_scale_cache, Tensor cu_seq_len, int[3] head_nums, *, Tensor? q_rotation=None, Tensor? k_rotation=None, Tensor? v_scale=None, float epsilon=1e-6) -> (Tensor, Tensor)": {
    "version": ["v2.7", "newest"]
  },
  "func: npu_qkv_rms_norm_rope_cache_with_kscale_functional(Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, Tensor k_cache, Tensor v_cache, Tensor k_scale_cache, Tensor cu_seq_len, int[3] head_nums, *, Tensor? q_rotation=None, Tensor? k_rotation=None, Tensor? v_scale=None, float epsilon=1e-6) -> (Tensor, Tensor, Tensor, Tensor, Tensor)": {
    "version": ["v2.7", "newest"]
  },
```

### 任务 1.4: Python 单元测试

**目标**: 新增 `test_npu_qkv_rms_norm_rope_cache_with_kscale.py`

**目标文件**: `test/test_custom_ops/test_npu_qkv_rms_norm_rope_cache_with_kscale.py`

**完成标准**:
- 测试类继承 `TestCase`，使用 `@SupportedDevices(['Ascend950'])`
- 覆盖正常路径：基本 shape/dtype 验证
- 覆盖报错路径：缺失 q_rotation/k_rotation/v_scale、head_nums 不合法、shape 不匹配
- 使用 `run_tests()` 入口

**测试场景**:

1. **正常路径**: 构造合法输入 → 调用 `torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale` → 验证输出 shape/dtype
2. **functional 变体**: 调用 functional 版本 → 验证 5 个输出 shape/dtype
3. **报错 — 缺失 q_rotation**: `q_rotation=None` → 期望报错
4. **报错 — 缺失 k_rotation**: `k_rotation=None` → 期望报错
5. **报错 — 缺失 v_scale**: `v_scale=None` → 期望报错
6. **报错 — head_nums 长度错误**: `head_nums=[1,2]` → 期望报错
7. **报错 — Nv != Nk**: `head_nums=[4,2,3]` → 期望报错
8. **报错 — shape 不匹配**: `qkv.shape[1] != sum(head_nums)` → 期望报错

**参考文件**:
- `test/test_custom_ops/test_npu_qkv_rms_norm_rope_cache.py` — 相似算子测试
- `test/test_custom_ops/test_npu_dual_level_quant_matmul.py` — Ascend950 测试模板

## 阶段二：验证与交付

### 任务 2.1: C++ 编译检查

**目标**: 使用 LSP 检查 C++ 适配文件无语法/类型错误

**目标文件**: `op_plugin/ops/opapi/QkvRmsNormRopeCacheWithKScaleKernelNpuOpApi.cpp`

**完成标准**: `lsp_diagnostics` 无 error 级别问题

### 任务 2.2: Python 测试验证

**目标**: 确认 Python 测试代码语法正确、逻辑合理

**目标文件**: `test/test_custom_ops/test_npu_qkv_rms_norm_rope_cache_with_kscale.py`

**完成标准**: Python 语法检查通过，测试场景覆盖完整

### 任务 2.3: 代码走查

**目标**: 对照 CHECKLIST.md 逐项检查，确保与已有算子风格一致

**完成标准**: CHECKLIST.md 阶段一所有检查项通过

## 当前进度

- [ ] 阶段一：PyTorch 接口封装
- [ ] 阶段二：验证与交付

## 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| ACLNN 接口签名与预期不一致 | C++ 适配编译失败 | 适配层代码按设计文档编写，ACLNN 接口就绪后联调 |
| fp8 TensorWrapper 传递方式不正确 | 运行时 dtype 不匹配 | 参考 KvRmsNormRopeCacheNpuOpApi 的 TensorWrapper 用法 |
| functional 变体返回顺序错误 | 用户调用结果异常 | 严格对照已有 npu_qkv_rms_norm_rope_cache_functional 的返回顺序 |
| YAML schema 字符串与 JSON 中不完全一致 | schema 测试失败 | 从 YAML 直接复制，不做手动改写 |
