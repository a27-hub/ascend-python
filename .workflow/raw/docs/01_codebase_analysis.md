# 代码库分析与关键发现

> 初始化工作流时对仓库代码、参考提交、设计文档的分析记录。

## 1. 仓库定位

op-plugin 是 Ascend Extension for PyTorch (torch_npu) 的算子插件库，为 PyTorch 框架开发者提供 NPU 算子库调用能力。核心职责是将底层 ACLNN/ACLOP 算子封装为 PyTorch 自定义 API。

## 2. 新增算子的标准流程（基于参考提交分析）

### 2.1 参考提交 c2a961c90fc — npu_dual_level_quant_matmul

涉及 4 个文件：

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `op_plugin/config/op_plugin_functions.yaml` | 修改 | 在 custom 段新增 func 定义，配置 op_api 版本 |
| `op_plugin/ops/opapi/DualLevelQuantMatmulKernelNpuOpApi.cpp` | 新增 | C++ 适配实现，包含输入校验、输出分配、EXEC_NPU_CMD 调用 |
| `test/core_tests/torch_npu_OpApi_schema_all.json` | 修改 | 注册 schema 测试条目 |
| `test/test_custom_ops/test_npu_dual_level_quant_matmul.py` | 新增 | Python 单元测试 |

### 2.2 参考提交 6c6718f5 — _npu_swiglu_backward_mx_quant_with_dual_axis

同样涉及 4 个文件，模式一致：

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `op_plugin/config/op_plugin_functions.yaml` | 修改 | 新增 func 定义 |
| `op_plugin/ops/opapi/SwigluBackwardMxQuantWithDualAxisOpApi.cpp` | 新增 | C++ 适配实现 |
| `test/core_tests/torch_npu_OpApi_schema_all.json` | 修改 | 注册 schema |
| `test/test_custom_ops/test_npu_swiglu_backward_mx_quant_with_dual_axis.py` | 新增 | Python 单元测试 |

### 2.3 标准模式总结

新增一个 PyTorch 自定义算子接口的最小交付件：

1. **YAML 配置** (`op_plugin_functions.yaml`)：定义 func schema + op_api 版本范围
2. **C++ 适配** (`op_plugin/ops/opapi/<OpName>KernelNpuOpApi.cpp`)：
   - 引入 `OpApiInterface.h`、`op_api_common.h`
   - 在 `op_api` 命名空间实现函数
   - 输入校验（平台、shape、dtype 等）
   - 输出 tensor 分配
   - `EXEC_NPU_CMD(aclnn<OpName>, ...)` 调用底层 ACLNN
3. **Schema 测试** (`torch_npu_OpApi_schema_all.json`)：注册 func schema
4. **Python 单元测试** (`test/test_custom_ops/test_<op_name>.py`)：功能验证

## 3. 已有相似算子分析 — npu_qkv_rms_norm_rope_cache

### 3.1 YAML 配置（行 5879-5883）

```yaml
- func: npu_qkv_rms_norm_rope_cache(Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos, Tensor sin,
    Tensor index, Tensor(a!) q_out, Tensor(b!) k_cache, Tensor(c!) v_cache, int[4] qkv_size, int[3] head_nums,
    *, Tensor? k_scale=None, Tensor? v_scale=None, Tensor? k_offset=None, Tensor? v_offset=None,
    float epsilon=1e-6, str cache_mode='PA_NZ', bool is_output_qkv=False) -> (Tensor, Tensor, Tensor)
  op_api: [v2.6, newest]

- func: npu_qkv_rms_norm_rope_cache_functional(...) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
  op_api: [v2.6, newest]
```

关键设计选择：
- in-place 输出使用 `Tensor(a!)` 标记（q_out, k_cache, v_cache）
- 有 functional 变体（clone 后调用，返回所有输出）
- 版本范围 `[v2.6, newest]`

### 3.2 C++ 适配 (`QkvRmsNormRopeCacheNpuOpApi.cpp`)

- 使用 `EXEC_NPU_NO_FORMAT_CHECK_CMD` 而非 `EXEC_NPU_CMD`（因为涉及特殊 format 的 tensor）
- 有 `prepare_before_quant` 辅助函数处理额外输出分配
- 返回 `std::tie(...)` 元组

## 4. 新算子 QkvRmsNormRopeCacheWithKScale 的 PyTorch 接口推导

基于设计文档 ACLNN 接口和已有相似算子模式，推导 PyTorch schema：

### 4.1 必选输入

| 参数 | PyTorch 类型 | 说明 |
|------|-------------|------|
| qkv | Tensor | bf16, [T, Nq+Nk+Nv, D] |
| q_gamma | Tensor | fp32, [D] |
| k_gamma | Tensor | fp32, [D] |
| cos_sin | Tensor | fp32, [MaxSeqLen, D] |
| slot_mapping | Tensor | int32, [T] |
| k_cache | Tensor(a!) | fp8_e4m3fn, [BlockNum, Nk, BlockSize, D] in-place |
| v_cache | Tensor(b!) | fp8_e4m3fn, [BlockNum, Nv, BlockSize, D] in-place |
| k_scale_cache | Tensor(c!) | fp32, [BlockNum, Nk, BlockSize, 1] in-place |
| cu_seq_len | Tensor | int32, [Batch+1] |
| head_nums | int[3] | [Nq, Nk, Nv] |

### 4.2 可选输入（当前场景强制非空）

| 参数 | PyTorch 类型 | 说明 |
|------|-------------|------|
| q_rotation | Tensor? | bf16, [D, D] |
| k_rotation | Tensor? | bf16, [D, D] |
| v_scale | Tensor? | fp32, [Nv] |

### 4.3 属性

| 参数 | PyTorch 类型 | 默认值 | 说明 |
|------|-------------|--------|------|
| epsilon | float | 1e-6 | RMSNorm epsilon |

### 4.4 输出

| 输出 | dtype | shape | 说明 |
|------|-------|-------|------|
| q_out | fp8_e4m3fn | [T, Nq, D] | Q 分支输出 |
| q_scale | fp32 | [T, Nq] | Q 分支 scale |

### 4.5 推导的 schema

```
npu_qkv_rms_norm_rope_cache_with_kscale(
    Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin,
    Tensor slot_mapping, Tensor(a!) k_cache, Tensor(b!) v_cache,
    Tensor(c!) k_scale_cache, Tensor cu_seq_len, int[3] head_nums,
    *, Tensor? q_rotation=None, Tensor? k_rotation=None,
    Tensor? v_scale=None, float epsilon=1e-6
) -> (Tensor, Tensor)
```

## 5. 关键差异：新算子 vs 已有 npu_qkv_rms_norm_rope_cache

| 维度 | 已有算子 | 新算子 |
|------|---------|--------|
| cos/sin | 分开传入 cos, sin | 合并为 cos_sin [MaxSeqLen, D] |
| index | 通用 index | slot_mapping [T]，映射公式固定 |
| q_out | in-place 参数 | 普通输出（设计文档未标记 in-place） |
| k_scale_cache | 无 | 新增 in-place 输出 |
| cu_seq_len | 无 | 新增前缀和输入 |
| q_rotation/k_rotation | 无 | 新增可选输入（当前强制非空） |
| v_scale | 可选 | 可选（当前强制非空） |
| qkv_size | 显式传入 | 不需要（从 tensor shape 推导） |
| cache_mode | 有 | 无（首期固定） |
| is_output_qkv | 有 | 无 |
| 平台限制 | 无显式限制 | 仅 Ascend950/arch35 |

## 6. 待确认项

见 GOAL.md 中的待确认问题列表。
