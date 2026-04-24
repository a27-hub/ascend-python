# （beta）at_npu::native::atb::npu_self_attention_prefix_encoder

## 产品支持情况

当前为 beta 接口。请以 `torch_npu` 同构环境中实际可用的 NNAL/ATB 版本及约束说明为准。

## 功能说明

MLA 场景下，使用分页管理的 KV Cache 计算 attention score，支持 prefix encoder 场景。

当前文档覆盖的 C++ wrapper 变体包括：

- `at_npu::native::atb::npu_self_attention_prefix_encoder`
- `at_npu::native::atb::npu_self_attention_prefix_encoder_out`

## 定义文件

`op_plugin/include/atb_ops.h`

## 函数原型

```cpp
at::Tensor at_npu::native::atb::npu_self_attention_prefix_encoder(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& block_tables, c10::SymIntArrayRef seqlen,
    c10::SymIntArrayRef kv_seqlen, int64_t q_headnum, double qk_scale,
    int64_t kv_headnum, const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& slopes,
    c10::optional<c10::string_view> mask_type_opt);

at::Tensor& at_npu::native::atb::npu_self_attention_prefix_encoder_out(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& block_tables, c10::SymIntArrayRef seqlen,
    c10::SymIntArrayRef kv_seqlen, int64_t q_headnum, double qk_scale,
    int64_t kv_headnum, const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& slopes,
    c10::optional<c10::string_view> mask_type_opt, at::Tensor& output);
```

## 参数说明

`query`、`key`、`value`：query、key、value 张量。  
`block_tables`：每个 query 对应的 KV Cache block table。  
`seqlen`：query 对应的每个 batch 的序列长度。  
`kv_seqlen`：key/value 对应的每个 batch 的序列长度。  
`q_headnum`：query 头数。  
`qk_scale`：`Q*K^T` 后的缩放系数。  
`kv_headnum`：KV 头数。  
`mask`、`slopes`：可选输入张量。  
`mask_type_opt`：可选字符串枚举，指定 mask 类型。  

变体差异：

- `npu_self_attention_prefix_encoder` 返回新分配的输出张量。
- `npu_self_attention_prefix_encoder_out` 额外接收 `output` 作为输出张量。

## 返回值说明

- 基础变体返回 attention 输出张量。
- `*_out` 变体返回写回后的输出张量引用。

## 约束说明

- `qseqlen <= kv_seqlen`。
- `(kv_seqlen - qseqlen) % 128 == 0`。
- `mask_type` 的具体取值和输入布局约束，请以当前 NNAL/ATB 版本为准。
