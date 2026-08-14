# npu_sparse_flash_attention 支持 value=None 特性设计文档

## 1 总体概述

| 项目 | 内容 |
| ---- | ---- |
| 特性名称 | `npu_sparse_flash_attention` / `npu_sparse_flash_attention_grad` 支持 `value=None` |
| 关联 PR | https://gitcode.com/Ascend/op-plugin/pull/5725 （commit `fbd6004c`） |
| 一句话概述 | 将 `value` 参数由必填 `Tensor` 改为可空 `Tensor?`，在 CANN >= 9.2.0 的平台上允许 `value=None`（含 A5/Ascend950），更低版本显式报错；autograd 反向与 meta 注册同步适配。 |
| 涉及模块 | 算子 schema 配置（`op_plugin_functions.yaml`、`derivatives.yaml`）、OpApi 前向/反向 kernel、版本门控工具（`op_api_common.h`）、Python meta 注册、schema 同步文件、自定义算子测试 |

## 2 背景与目标

### 2.1 问题背景

`npu_sparse_flash_attention` 是面向稀疏注意力（QSFA 等）场景的融合算子，支持带 `block_table`/`sinks` 的 KV Cache 推理形态。在部分推理/解码场景中，`value` 张量由 KV Cache 内部维护（如 `sinks` 形态的 V2 接口），调用方无需也无法在每次解码时显式传入独立的 `value`，但当前接口强制要求 `value` 为非空 `Tensor`，导致：

- 无法表达“不传 value”的合法语义，调用方被迫构造冗余张量；
- 多一次不必要的输入拷贝/显存占用。

CANN 9.2.0 起，`aclnnSparseFlashAttention` 算子层支持可空 `value` 输入（含 A5/Ascend950 平台），为插件层放开 `value=None` 提供了前提。

### 2.2 当前现状

- schema 中 `value` 为必填 `Tensor`（位置参数，位于 `sparse_indices` 之前）；
- 前向 kernel 直接对 `value` 做非空校验并透传 aclnn；
- 反向 kernel 对 `value` 做维度校验并构造 `d_value`；
- Python meta 注册将 `value` 列入必填参数校验，`value=None` 在 fake/meta 路径下会直接报错；
- 无版本/SoC 门控，是否支持完全取决于底层 aclnn 接口。

### 2.3 设计目标

1. `value` 由 `Tensor` 变为 `Tensor?`，保持位置参数不变（仍是第 3 个参数），存量按位置传参的调用无感知；
2. `value=None` 仅在 CANN >= 9.2.0 时合法，更低版本显式抛出 `NOT_SUPPORT` 错误；
3. eager（C++ kernel）、meta/fake（Python 注册）两层门控条件保持一致；
4. 反向对 `value=None` 返回空张量 `d_value`，输出元组形状保持 5 元不变，autograd 契约不受影响。

## 3 总体设计

### 3.1 对外接口设计

前向 schema（仅 `value` 参数发生变化，其余不变）：

```diff
- npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, *, ...)
+ npu_sparse_flash_attention(Tensor query, Tensor key, Tensor? value, Tensor sparse_indices, float scale_value, *, ...)
```

反向 schema 同步变更：

```diff
- npu_sparse_flash_attention_grad(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, Tensor d_out, ...)
+ npu_sparse_flash_attention_grad(Tensor query, Tensor key, Tensor? value, Tensor sparse_indices, Tensor d_out, ...)
```

说明：

- `value` 保持为位置参数（非 keyword-only），`Tensor?` 同时接受 `Tensor` 与 `None`，无需新增重载即可保持向后兼容；
- 前向返回 `(out, softmax_max, softmax_sum)` 不变；反向返回 `(d_query, d_key, d_value, d_query_rope, d_key_rope)` 不变，`value=None` 时 `d_value` 为形状 `[0]` 的空张量；
- `value=None` 时反向输入依赖关系不变：`d_value` 为空张量，derivatives.yaml 中的梯度映射公式无需修改（`value` 由 `npu_sparse_flash_attention_grad` 的第 3 个输入承接）。

### 3.2 版本门控设计

新增 C++ 门控函数（`op_plugin/utils/op_api_common.h`）：

```cpp
// Check if npu_sparse_flash_attention supports value=None: CANN >= 9.2.0.
inline bool IsSupportSparseFlashAttentionNoneValue()
{
    static const bool is_support = []() -> bool {
        return op_plugin::utils::is_gte_cann_version_920();
    }();
    return is_support;
}
```

设计要点：

- 条件仅为 **CANN 版本**：`is_gte_cann_version_920()`（CANN >= 9.2.0），不再按 SoC 区分（A2/A3/A5 均支持）；
- 使用 `static const bool` 局部静态变量，门控结果进程内只计算一次（CANN 版本在进程运行期不会变化）。

Python 侧（meta 注册）使用等价条件：

```python
get_cann_version() >= "9.2.0"
```

### 3.3 模块改动

#### 3.3.1 配置层（schema）

- `op_plugin/config/op_plugin_functions.yaml`：前向与反向的 `func` 签名中 `Tensor value` 改为 `Tensor? value`；
- `op_plugin/config/derivatives.yaml`：同名签名同步修改，梯度映射公式不变（`value` 在反向输入中位置不变）；
- `test/core_tests/torch_npu_OpApi_schema_all.json`：schema 基准文件同步更新。

#### 3.3.2 前向 kernel（SparseFlashAttentionKernelNpuOpApi.cpp）

```cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const c10::optional<at::Tensor>& value,   // 原 const at::Tensor&
    ...
```

- 在既有 `query`/`key`/`sparse_indices` 非空校验之后新增门控：

```cpp
if (!value.has_value()) {
  TORCH_CHECK(
      IsSupportSparseFlashAttentionNoneValue(),
      "npu_sparse_flash_attention: value=None is only supported with CANN >= 9.2.0.",
      OPS_ERROR(ErrCode::NOT_SUPPORT));
}
```

- `value` 继续直接透传给 `aclnnSparseFlashAttention` / `aclnnSparseFlashAttentionV2`（经 `EXEC_NPU_NO_FORMAT_CHECK_CMD` 宏统一参数转换），可空语义由算子层处理；`sinks` 形态（V2 接口）的调用路径保持不变。

#### 3.3.3 反向 kernel（SparseFlashAttentionGradKernelNpuOpApi.cpp）

- 签名 `const at::Tensor& value` → `const c10::optional<at::Tensor>& value`；
- 入参归一化：`const at::Tensor& value_const = value.value_or(at::Tensor());`
- 维度校验改为分支：
  - `value_const.defined()`：保留原 3/4 维校验；
  - 未定义（`value=None`）：走与 3.2 节相同的门控校验，不满足则抛 `NOT_SUPPORT`；
- 输出构造改为分支：`value=None` 时 `d_value = at::empty({0}, query.options())`，否则沿用 `apply_tensor_without_format(value_const)`；
- aclnn 调用处统一传入归一化后的 `value_const`。

#### 3.3.4 Python meta 注册（_meta_registrations.py）

- 前向 `npu_sparse_flash_attention_forward`：`value` 移出 `require_param` 必填字典；`value is None` 时走门控校验（等价条件见 3.2 节），否则保留原 `numel() > 0` 校验；
- 反向 `npu_sparse_flash_attention_grad_meta`：`value is None` 时走门控校验且 `d_value = torch.empty([0], dtype=query.dtype, device='meta')`，否则保留原 `value.new_empty(...)` 逻辑。

这样 torch.compile / export 的 fake-tensor 路径与 eager 行为一致，不会出现“eager 可跑、图模式报错”的分裂。

### 3.4 行为矩阵

| 场景 | CANN >= 9.2.0 | CANN < 9.2.0 |
| ---- | ------------- | ------------- |
| `value` 传非空 Tensor | 原行为不变 | 原行为不变 |
| `value=None`（前向） | 正常执行 | 抛 `NOT_SUPPORT`，错误信息含 "only supported" |
| `value=None`（反向） | 正常执行，`d_value` 为空张量 `[0]` | 抛 `NOT_SUPPORT` |
| `value=None` + `sinks`（V2 路径） | 门控通过后透传，未专门测试（见 6 节） | 抛 `NOT_SUPPORT` |

## 4 兼容性分析

- **存量调用**：`value` 位置不变，按位置传 Tensor 的既有调用无任何影响；
- **类型兼容**：`Tensor?` 参数同时接受 `Tensor` 与 `None`，无需新 overload，schema 变更最小；
- **输出契约**：前向 3 元、反向 5 元返回不变，`value=None` 时反向的 `d_value` 为空张量，autograd 链路（derivatives.yaml）无需变化；
- **行为收紧**：此前 `value=None` 在旧平台上会以类型不匹配等底层错误失败，本特性改为显式 `NOT_SUPPORT` 错误，错误信息更明确，属于行为收紧而非破坏。

## 5 测试设计

新增用例 `test_sfa_value_none`（`test/test_custom_ops/test_npu_sparse_flash_attention.py`）：

- 构造 4D `query`（b=4, s1=1, n1=128, dn=512）、`key`（b=4, s2=8192, n2=1, dn=512）、`sparse_indices`（b, s1, n2, 2048），均为 float16 NPU 张量；
- 平台支持时：`torch_npu.npu_sparse_flash_attention(query, key, None, sparse_indices, scale_value, sparse_mode=3)` 正常返回，校验输出形状为 `(b, s1, n1, dn)`；
- 平台不支持时：断言抛出 `RuntimeError` 且错误信息包含 `"only supported"`。

现有覆盖缺口（建议后续补充）：

1. 反向路径未覆盖：`value=None` 时的 `npu_sparse_flash_attention_grad`（含 d_value 空张量返回）无用例；
2. `sinks`（V2 接口）+ `value=None` 组合未覆盖；
3. meta/fake 路径（torch.compile / export）下 `value=None` 的门控行为未覆盖；
4. 无性能对比数据（见 6 节）。

## 6 风险与遗留问题

1. **双端门控条件重复**：C++（`IsSupportSparseFlashAttentionNoneValue`）与 Python meta 各维护一份等价条件，存在将来漂移的风险。若后续 CANN 支持范围变化，需同步修改两处及本测试中的平台判断；
2. **可空语义依赖 CANN 版本**：`value=None` 是否真正可用由 CANN 9.2.0 起的算子层决定（A2/A3/A5 均已支持），插件层门控是前置保护；若 CANN 后续版本行为变化，门控条件需随之调整；
3. **性能收益未量化**：`value=None` 预期减少一次输入张量的显存占用与拷贝，但 PR 未附性能对比数据，建议补充解码场景下的显存/时延对比。

## 7 附录

### 7.1 变更文件清单

| 文件 | 变更内容 |
| ---- | -------- |
| `op_plugin/config/op_plugin_functions.yaml` | 前向/反向 schema `value` 改为 `Tensor?` |
| `op_plugin/config/derivatives.yaml` | 同名签名同步修改 |
| `op_plugin/ops/opapi/SparseFlashAttentionKernelNpuOpApi.cpp` | 前向 kernel 签名改为 optional + 门控 |
| `op_plugin/ops/opapi/SparseFlashAttentionGradKernelNpuOpApi.cpp` | 反向 kernel 签名改为 optional + 分支处理 |
| `op_plugin/utils/op_api_common.h` | 新增 `IsSupportSparseFlashAttentionNoneValue()` |
| `op_plugin/python/meta/_meta_registrations.py` | 前向/反向 meta 注册适配 `value=None` |
| `test/core_tests/torch_npu_OpApi_schema_all.json` | schema 基准同步 |
| `test/test_custom_ops/test_npu_sparse_flash_attention.py` | 新增 `test_sfa_value_none` 用例 |

### 7.2 参考资料

- PR 5725：https://gitcode.com/Ascend/op-plugin/pull/5725
- commit `fbd6004c`：feat: support value=None for npu_sparse_flash_attention on CANN >= 9.2.0 and A2/A3 SoC（后续修订：SoC 限制已放开，仅保留 CANN >= 9.2.0 门控）
