# YAML Schema 设计草案

> 新算子 npu_qkv_rms_norm_rope_cache_with_kscale 的 YAML 配置和 C++ 适配设计草案。

## 1. YAML 配置草案

### 方案 A：q_out/q_scale 作为返回值

```yaml
  - func: npu_qkv_rms_norm_rope_cache_with_kscale(Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor(c!) k_scale_cache, Tensor cu_seq_len, int[3] head_nums, *, Tensor? q_rotation=None, Tensor? k_rotation=None, Tensor? v_scale=None, float epsilon=1e-6) -> (Tensor, Tensor)
    op_api: [v2.7, newest]
```

返回 `(q_out, q_scale)`。

### 方案 B：q_out 也作为 in-place 参数（与已有算子风格一致）

```yaml
  - func: npu_qkv_rms_norm_rope_cache_with_kscale(Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, Tensor(d!) q_out, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor(c!) k_scale_cache, Tensor cu_seq_len, int[3] head_nums, *, Tensor? q_rotation=None, Tensor? k_rotation=None, Tensor? v_scale=None, float epsilon=1e-6) -> (Tensor)
    op_api: [v2.7, newest]
```

返回 `(q_scale)`，q_out 通过 in-place 参数写入。

### 方案 C：functional 变体

如果需要 functional 变体（不做 in-place，所有输出作为返回值）：

```yaml
  - func: npu_qkv_rms_norm_rope_cache_with_kscale_functional(Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, Tensor k_cache, Tensor v_cache, Tensor k_scale_cache, Tensor cu_seq_len, int[3] head_nums, *, Tensor? q_rotation=None, Tensor? k_rotation=None, Tensor? v_scale=None, float epsilon=1e-6) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
    op_api: [v2.7, newest]
```

返回 `(q_out, q_scale, k_cache_out, v_cache_out, k_scale_cache_out)`。

## 2. C++ 适配草案

```cpp
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_qkv_rms_norm_rope_cache_with_kscale(
    const at::Tensor& qkv,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    const at::Tensor& cos_sin,
    const at::Tensor& slot_mapping,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    at::Tensor& k_scale_cache,
    const at::Tensor& cu_seq_len,
    at::IntArrayRef head_nums,
    const c10::optional<at::Tensor>& q_rotation,
    const c10::optional<at::Tensor>& k_rotation,
    const c10::optional<at::Tensor>& v_scale,
    double epsilon)
{
    // 1. Platform check
    TORCH_CHECK(c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
                "This interface is supported only on the Ascend950 platform.",
                OPS_ERROR(ErrCode::PARAM));

    // 2. Input validation
    // ... shape, dtype checks ...

    // 3. Optional tensor handling
    const at::Tensor& q_rotation_v = c10::value_or_else(q_rotation, [] { return at::Tensor(); });
    const at::Tensor& k_rotation_v = c10::value_or_else(k_rotation, [] { return at::Tensor(); });
    const at::Tensor& v_scale_v = c10::value_or_else(v_scale, [] { return at::Tensor(); });

    // 4. Output allocation
    int64_t T = qkv.size(0);
    int64_t Nq = head_nums[0];
    int64_t D = qkv.size(2);
    // q_out: [T, Nq, D], fp8_e4m3fn
    // q_scale: [T, Nq], fp32
    at::Tensor q_out = npu_preparation::apply_tensor_without_format(
        {T, Nq, D}, qkv.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor q_scale = npu_preparation::apply_tensor_without_format(
        {T, Nq}, qkv.options().dtype(at::kFloat));

    // 5. Call ACLNN
    EXEC_NPU_CMD(aclnnQkvRmsNormRopeCacheWithKScale,
                 qkv, q_gamma, k_gamma, cos_sin, slot_mapping,
                 k_cache, v_cache, k_scale_cache, cu_seq_len,
                 q_rotation_v, k_rotation_v, v_scale_v,
                 head_nums, epsilon,
                 q_out, q_scale);

    return std::make_tuple(q_out, q_scale);
}
} // namespace op_api
```

## 3. 注意事项

1. `EXEC_NPU_CMD` vs `EXEC_NPU_NO_FORMAT_CHECK_CMD`：如果涉及 fp8 或特殊 format tensor，可能需要使用 `EXEC_NPU_NO_FORMAT_CHECK_CMD`
2. `TensorWrapper`：对于 fp8 输出，可能需要使用 `TensorWrapper` 指定 ACL dtype
3. `check_aclnn_kernel_available`：是否需要运行时检查 ACLNN kernel 可用性
4. 版本范围：需要确认 `[v2.7, newest]` 还是其他版本
