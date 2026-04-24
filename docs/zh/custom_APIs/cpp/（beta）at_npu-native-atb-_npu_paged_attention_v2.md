# （beta）at_npu::native::atb::_npu_paged_attention_v2

## 产品支持情况

当前为 beta 接口。请以 `torch_npu` 同构环境中实际可用的 NNAL/ATB 版本及约束说明为准。

## 功能说明

基于 paged attention 的 KV Cache 分页管理机制计算 attention score，支持基础场景和 alibi 场景。

当前文档覆盖的 C++ wrapper 变体包括：

- `at_npu::native::atb::_npu_paged_attention_v2`
- `at_npu::native::atb::_npu_paged_attention_v2_get_workspace`

## 定义文件

`op_plugin/include/atb_ops.h`

## 函数原型

```cpp
at::Tensor& at_npu::native::atb::_npu_paged_attention_v2(
    const at::Tensor& query, const at::Tensor& key_cache, const at::Tensor& block_table,
    c10::SymIntArrayRef context_lens, const c10::optional<at::Tensor>& value_cache,
    const c10::optional<at::Tensor>& mask, int64_t num_kv_heads, int64_t num_heads,
    double scale_value, int64_t mask_type, const c10::optional<at::Tensor>& workspace,
    at::Tensor& out);

at::Tensor at_npu::native::atb::_npu_paged_attention_v2_get_workspace(
    const at::Tensor& query, const at::Tensor& key_cache, const at::Tensor& block_table,
    c10::SymIntArrayRef context_lens, const c10::optional<at::Tensor>& value_cache,
    const c10::optional<at::Tensor>& mask, int64_t num_kv_heads, int64_t num_heads,
    double scale_value, int64_t mask_type, at::Tensor& out);
```

## 参数说明

`query`：query 张量。  
`key_cache`、`value_cache`：cache 好的 key/value 张量。  
`block_table`：每个 query 对应的 KV Cache block table。  
`context_lens`：每个 query 对应的上下文长度。  
`mask`：可选 mask 张量。  
`num_kv_heads`：KV 头数。  
`num_heads`：query 头数。  
`scale_value`：缩放系数。  
`mask_type`：mask 类型枚举。  
`workspace`：可选 workspace 张量。  
`out`：输出张量。  

变体差异：

- `_npu_paged_attention_v2` 直接执行计算并写回 `out`。
- `_npu_paged_attention_v2_get_workspace` 返回执行所需的 workspace 张量。

## 返回值说明

- `_npu_paged_attention_v2` 返回写回后的 `out` 张量引用。
- `_npu_paged_attention_v2_get_workspace` 返回 workspace 张量。

## 约束说明

- `block_size % 16 == 0`，推荐 `block_size = 128`。
- `block_size <= 128`。
- `block_table` 中元素取值需在 `[0, num_blocks)` 范围内。
- `head_size` 范围与 `key_cache/value_cache` 的数据排布相关，具体请以当前 NNAL/ATB 版本为准。
- 若使用 GQA 模式，需要满足 `num_heads > num_kv_heads` 且 `num_heads % num_kv_heads == 0`。
