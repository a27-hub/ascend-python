# at_npu::native::atb::_npu_paged_attention_v2

## 定义文件

op_plugin/include/atb_ops.h

## 函数原型

```cpp
at::Tensor& at_npu::native::atb::_npu_paged_attention_v2(
    const at::Tensor& query, const at::Tensor& key_cache, const at::Tensor& block_table,
    c10::SymIntArrayRef context_lens, const c10::optional<at::Tensor>& value_cache,
    const c10::optional<at::Tensor>& mask, int64_t num_kv_heads, int64_t num_heads,
    double scale_value, int64_t mask_type, const c10::optional<at::Tensor>& workspace,
    at::Tensor& out)

at::Tensor at_npu::native::atb::_npu_paged_attention_v2_get_workspace(
    const at::Tensor& query, const at::Tensor& key_cache, const at::Tensor& block_table,
    c10::SymIntArrayRef context_lens, const c10::optional<at::Tensor>& value_cache,
    const c10::optional<at::Tensor>& mask, int64_t num_kv_heads, int64_t num_heads,
    double scale_value, int64_t mask_type, at::Tensor& out)
```

## 功能说明

基于 paged attention 的 KV Cache 分页管理机制计算 attention score，支持基础场景和 alibi 场景。该接口族覆盖执行接口和 workspace 查询变体。

## 参数说明

query：Tensor 类型，query 张量。  
key_cache：Tensor 类型，cache 好的 key 张量。  
block_table：Tensor 类型，每个 query 对应的 KV Cache block table。  
context_lens：SymIntArrayRef 类型，每个 query 对应的上下文长度。  
value_cache：可选 Tensor 类型，cache 好的 value 张量。  
mask：可选 Tensor 类型，mask 张量。  
num_kv_heads：int64_t 类型，KV 头数。  
num_heads：int64_t 类型，query 头数。  
scale_value：double 类型，缩放系数。  
mask_type：int64_t 类型，mask 类型枚举。  
workspace：可选 Tensor 类型，workspace 张量。  
out：Tensor 类型，输出张量。
