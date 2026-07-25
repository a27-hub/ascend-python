# npu_moe_token_permute_grad → npu_moe_token_permute_grad_v2

## Background

`npu_moe_token_permute_grad` 是 MoE token 置换算子的反向 kernel。`tokens` 和 `indices` 两个 Tensor 在实际反向传播中仅用于提取标量属性（`tokens.shape[0]`、`tokens.dtype`、`indices.dim()/shape[1]`），但作为 autograd 的 `saved_tensors` 被完整持有，导致显存浪费。

## Solution

新增 `npu_moe_token_permute_grad_v2`，接口改为直接接收标量值。旧版保留共存，derivatives.yaml 反向公式切换到新 variant。

```
旧: npu_moe_token_permute_grad(Tensor tokens, Tensor grad, Tensor indices, Tensor result1, bool padded_mode) -> Tensor
新: npu_moe_token_permute_grad_v2(Tensor grad, Tensor result1, int size_0, ScalarType dtype, int topK, bool padded_mode) -> Tensor
```

关键决策：
- **共存不替换**：旧版保留，为已有调用方留兼容路径
- **derivatives 换绑**：forward `npu_moe_token_permute` 的反向公式从旧 `_grad` 切到新 `_grad_v2`——不切的话 v2 不会被 autograd 调用
- **标量提取上移**：原 kernel 内部的 `tokens.shape[0]` / `tokens.dtype` / `indices.shape[1]` 逻辑，改为在 derivatives.yaml 公式中表达

## File Changes

### op_plugin_functions.yaml — 新增 v2 条目

```yaml
- func: npu_moe_token_permute_grad_v2(Tensor grad_permuted_tokens, Tensor sorted_indices, int tokens_size_0, ScalarType tokens_dtype, int num_topK, bool padded_mode=False) -> Tensor
  op_api: [v2.1, newest]
  gen_opapi:
    grad_tokens:
      size: 'npu_moe_token_permute_grad_v2_out_size(tokens_size_0, grad_permuted_tokens, sorted_indices)'
      dtype: tokens_dtype
    exec: aclnnMoeTokenPermuteGrad, grad_permuted_tokens, sorted_indices, num_topK, padded_mode, grad_tokens
```

- `tokens_dtype` 作为 `ScalarType` 参数直接传入，不再从 `tokens` Tensor 推断
- `num_topK` 用标量 int 替代 `indices` Tensor
- `size` 调用新 `_v2_out_size`，入参减少

### derivatives.yaml — 修改反向公式（本案例最关键一步）

```yaml
# 修改前
- name: npu_moe_token_permute(Tensor tokens, Tensor indices, int? num_out_tokens=None, bool padded_mode=False) -> (Tensor, Tensor)
  tokens: npu_moe_token_permute_grad(tokens, grad, indices, result1, padded_mode)

# 修改后
- name: npu_moe_token_permute(Tensor tokens, Tensor indices, int? num_out_tokens=None, bool padded_mode=False) -> (Tensor, Tensor)
  tokens: "npu_moe_token_permute_grad_v2(grad, result1, tokens.sym_size(0).expect_int(), tokens.scalar_type(), (indices.dim() == 1) ? 1 : indices.sym_size(1).expect_int(), padded_mode)"
```

标量提取逻辑从 kernel 内部上移到公式中：
- `tokens` → `tokens.sym_size(0).expect_int()` + `tokens.scalar_type()`
- `indices` → `(indices.dim() == 1) ? 1 : indices.sym_size(1).expect_int()`

不改此文件，v2 虽已注册但 autograd 不会调用。

### KernelNpuOutputSize.cpp — 新增 `_v2_out_size`

```cpp
c10::SmallVector<int64_t, SIZE> npu_moe_token_permute_grad_v2_out_size(
    const int64_t tokens_size_0,
    const at::Tensor &grad_permuted_tokens,
    const at::Tensor &sorted_indices)
{
    TORCH_CHECK(grad_permuted_tokens.dim() == DIM_2, "..." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(sorted_indices.dim() == DIM_1 || sorted_indices.dim() == DIM_2, "..." + OPS_ERROR(ErrCode::PARAM));
    int64_t num_tokens = tokens_size_0;
    c10::SmallVector<int64_t, SIZE> output_shape = {num_tokens, grad_permuted_tokens.size(1)};
    return output_shape;
}
```

| 对比 | 旧 `_out_size` | 新 `_v2_out_size` |
|------|---------------|-------------------|
| 入参 | 4 个 Tensor | 1 标量 + 2 Tensor |
| N 来源 | `tokens.size(0)` | `tokens_size_0` 直接使用 |
| 维度校验 | 校验 4 个 Tensor | 校验 2 个 Tensor |

### KernelNpuOutputSize.h — 新增声明

```cpp
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> npu_moe_token_permute_grad_v2_out_size(
    const int64_t tokens_size_0, const at::Tensor &grad_permuted_tokens, const at::Tensor &sorted_indices);
```

### _meta_registrations.py — 新增 v2 meta

```python
@impl(m, "npu_moe_token_permute_grad_v2")
def npu_moe_token_permute_grad_v2_meta(grad_permuted_tokens, sorted_indices,
                                        tokens_size_0, tokens_dtype, num_topK, padded_mode=False):
    torch._check(grad_permuted_tokens.dim() == 2, ...)
    torch._check(sorted_indices.dim() == 1, ...)
    N, D = tokens_size_0, grad_permuted_tokens.shape[1]
    return torch.empty((N, D), dtype=tokens_dtype, device=grad_permuted_tokens.device)
```

| 对比 | 旧 meta | 新 v2 meta |
|------|--------|-----------|
| N 来源 | `tokens.shape[0]` | `tokens_size_0` 参数 |
| dtype 来源 | `tokens.dtype` | `tokens_dtype` 参数 |
| device 来源 | `tokens.device` | `grad_permuted_tokens.device` |
| torch._check | 4 个 | 2 个 |

### test_fake_tensor.py — 追加测试

v2 测试（追加到已有方法内）：

```python
grad_tokens_v2 = torch_npu.npu_moe_token_permute_grad_v2(
    grad_permuted_tokens=grad_permuted_tokens_for_permute,
    sorted_indices=sorted_indices,
    tokens_size_0=tokens.shape[0],
    tokens_dtype=tokens.dtype,
    num_topK=topk,
    padded_mode=False,
)
self.assertEqual(grad_tokens_v2.dtype, tokens.dtype)
self.assertEqual(grad_tokens_v2.shape, tokens.shape)
```

旧算子补充用例：`test_npu_moe_token_unpermute_grad_mixed_dtype`（probs float32 vs grad bfloat16）和 `test_npu_moe_token_unpermute_grad_probs_none`（probs=None 边界）。

### torch_npu_OpApi_schema_all.json — 新增 schema

```json
"func: npu_moe_token_permute_grad_v2(Tensor grad_permuted_tokens, Tensor sorted_indices, int tokens_size_0, ScalarType tokens_dtype, int num_topK, bool padded_mode=False) -> Tensor": {
  "version": ["all_version"]
}
```

## Pitfalls

- **忘了做 derivatives 换绑**：v2 注册了但 autograd 仍调旧版。grep derivatives.yaml 确认 forward 的反向公式指向新 variant
- **sym_size() 返回 SymInt 而非 int64_t**：必须加 `.expect_int()`，否则类型不匹配
- **公式含特殊字符需双引号**：`?`、`:` 出现在 YAML 值中时必须用双引号包裹整个公式
- **修改已有算子时顺带检查旧测试覆盖**：追加 v2 测试时发现了旧算子 `probs=None` 和 `mixed_dtype` 两个边界盲区
