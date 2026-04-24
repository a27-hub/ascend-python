# （beta）at_npu::native::atb::_npu_flash_attention_v2

## 定义文件

op_plugin/include/atb_ops.h

## 函数原型

```cpp
at::Tensor& at_npu::native::atb::_npu_flash_attention_v2(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    c10::SymIntArrayRef seq_len, const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& slopes, int64_t kernel_type, int64_t mask_type,
    double scale_value, int64_t num_heads, int64_t num_kv_heads, at::Tensor& out)
```

## 功能说明

基于 flash attention 机制计算 attention score，支持基础场景、普通 mask、alibi mask 以及压缩 mask 等场景。

## 参数说明

query：Tensor 类型，query 张量。  
key：Tensor 类型，key 张量。  
value：Tensor 类型，value 张量。  
seq_len：SymIntArrayRef 类型，序列长度数组。  
mask：可选 Tensor 类型，mask 张量。  
slopes：可选 Tensor 类型，alibi 系数张量。  
kernel_type：int64_t 类型，内核精度类型。  
mask_type：int64_t 类型，mask 类型枚举。  
scale_value：double 类型，缩放系数。  
num_heads：int64_t 类型，query 头数。  
num_kv_heads：int64_t 类型，KV 头数。  
out：Tensor 类型，输出张量。
