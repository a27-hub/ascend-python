# at_npu::native::atb::_npu_flash_attention_prefix_v2

## 定义文件

op_plugin/include/atb_ops.h

## 函数原型

```cpp
at::Tensor& at_npu::native::atb::_npu_flash_attention_prefix_v2(
    const at::Tensor& query, const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& mask, c10::SymIntArrayRef seq_len,
    c10::SymIntArrayRef context_lens, const c10::optional<at::Tensor>& slopes,
    const int64_t kernel_type, const int64_t mask_type, int64_t num_kv_heads,
    int64_t num_heads, double scale_value, at::Tensor& out)
```

## 功能说明

基于 flash attention 机制计算 prefix 场景下的 attention score，支持压缩 mask 与 prefix cache 输入场景。

## 参数说明

query：Tensor 类型，query 张量。  
key_cache：Tensor 类型，cache 好的 key 张量。  
value_cache：Tensor 类型，cache 好的 value 张量。  
block_table：Tensor 类型，每个 batch 对应的 block 索引表。  
mask：Tensor 类型，压缩 mask 张量。  
seq_len：SymIntArrayRef 类型，query 对应的每个 batch 序列长度。  
context_lens：SymIntArrayRef 类型，key/value 对应的每个 batch 序列长度。  
slopes：可选 Tensor 类型，alibi 系数张量。  
kernel_type：int64_t 类型，内核精度类型。  
mask_type：int64_t 类型，mask 类型枚举。  
num_kv_heads：int64_t 类型，KV 头数。  
num_heads：int64_t 类型，query 头数。  
scale_value：double 类型，缩放系数。  
out：Tensor 类型，输出张量。
