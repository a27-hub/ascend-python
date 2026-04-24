# （beta）at_npu::native::atb::npu_self_attention_prefix_encoder

## 定义文件

op_plugin/include/atb_ops.h

## 函数原型

```cpp
at::Tensor at_npu::native::atb::npu_self_attention_prefix_encoder(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& block_tables, c10::SymIntArrayRef seqlen,
    c10::SymIntArrayRef kv_seqlen, int64_t q_headnum, double qk_scale,
    int64_t kv_headnum, const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& slopes,
    c10::optional<c10::string_view> mask_type_opt)

at::Tensor& at_npu::native::atb::npu_self_attention_prefix_encoder_out(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& block_tables, c10::SymIntArrayRef seqlen,
    c10::SymIntArrayRef kv_seqlen, int64_t q_headnum, double qk_scale,
    int64_t kv_headnum, const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& slopes,
    c10::optional<c10::string_view> mask_type_opt, at::Tensor& output)
```

## 功能说明

MLA 场景下，使用分页管理的 KV Cache 计算 attention score，支持 prefix encoder 场景。该接口族覆盖基础输出和 `out` 变体。

## 参数说明

query：Tensor 类型，query 张量。  
key：Tensor 类型，key 张量。  
value：Tensor 类型，value 张量。  
block_tables：Tensor 类型，每个 query 对应的 KV Cache block table。  
seqlen：SymIntArrayRef 类型，query 对应的每个 batch 的序列长度。  
kv_seqlen：SymIntArrayRef 类型，key/value 对应的每个 batch 的序列长度。  
q_headnum：int64_t 类型，query 头数。  
qk_scale：double 类型，`Q*K^T` 后的缩放系数。  
kv_headnum：int64_t 类型，KV 头数。  
mask：可选 Tensor 类型，mask 张量。  
slopes：可选 Tensor 类型，alibi 相关输入。  
mask_type_opt：可选字符串枚举参数。  
output：Tensor 类型，`out` 变体的输出张量。
