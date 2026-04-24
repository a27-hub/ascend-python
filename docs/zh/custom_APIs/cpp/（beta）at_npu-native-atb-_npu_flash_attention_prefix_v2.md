# （beta）at_npu::native::atb::_npu_flash_attention_prefix_v2

## 产品支持情况

当前为 beta 接口。请以 `torch_npu` 同构环境中实际可用的 NNAL/ATB 版本及约束说明为准。

## 功能说明

基于 flash attention 机制计算 prefix 场景下的 attention score，支持压缩 mask 与 alibi 相关场景。

## 定义文件

`op_plugin/include/atb_ops.h`

## 函数原型

```cpp
at::Tensor& at_npu::native::atb::_npu_flash_attention_prefix_v2(
    const at::Tensor& query, const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& mask, c10::SymIntArrayRef seq_len,
    c10::SymIntArrayRef context_lens, const c10::optional<at::Tensor>& slopes,
    const int64_t kernel_type, const int64_t mask_type, int64_t num_kv_heads,
    int64_t num_heads, double scale_value, at::Tensor& out);
```

## 参数说明

`query`：query 张量。  
`key_cache`、`value_cache`：按 block 存放的 KV Cache 张量。  
`block_table`：每个 batch 对应的 block 索引表。  
`mask`：压缩 mask 张量。  
`seq_len`：query 对应的每个 batch 序列长度。  
`context_lens`：key/value 对应的每个 batch 序列长度。  
`slopes`：可选 alibi 系数张量。  
`kernel_type`：内核精度类型。  
`mask_type`：mask 类型枚举。  
`num_kv_heads`：KV 头数。  
`num_heads`：query 头数。  
`scale_value`：缩放系数。  
`out`：输出张量。  

## 返回值说明

返回写回后的 `out` 张量引用。

## 约束说明

- 当前默认 `calc_type` 为 `PREFIX_ENCODER`。
- `blockSize` 必须设置为 `128`。
- `PrefixCacheLen` 需要按 `128` 对齐。
- `mask_type` 仅支持压缩 mask 相关类型，具体请以当前 NNAL/ATB 版本为准。
- 需要满足 `num_heads >= num_kv_heads` 且 `num_heads` 为 `num_kv_heads` 的整数倍。
