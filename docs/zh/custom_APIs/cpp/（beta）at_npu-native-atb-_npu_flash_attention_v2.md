# （beta）at_npu::native::atb::_npu_flash_attention_v2

## 产品支持情况

当前为 beta 接口。请以 `torch_npu` 同构环境中实际可用的 NNAL/ATB 版本及约束说明为准。

## 功能说明

基于 flash attention 机制计算 attention score，支持基础场景、普通 mask、alibi mask 以及压缩 mask 等多种场景。

## 定义文件

`op_plugin/include/atb_ops.h`

## 函数原型

```cpp
at::Tensor& at_npu::native::atb::_npu_flash_attention_v2(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    c10::SymIntArrayRef seq_len, const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& slopes, int64_t kernel_type, int64_t mask_type,
    double scale_value, int64_t num_heads, int64_t num_kv_heads, at::Tensor& out);
```

## 参数说明

`query`、`key`、`value`：query、key、value 张量。  
`seq_len`：序列长度数组。  
`mask`：可选 mask 张量。  
`slopes`：可选 alibi 系数张量。  
`kernel_type`：内核精度类型。  
`mask_type`：mask 类型枚举。  
`scale_value`：缩放系数。  
`num_heads`：query 头数。  
`num_kv_heads`：KV 头数。  
`out`：输出张量。  

## 返回值说明

返回写回后的 `out` 张量引用。

## 约束说明

- 当前默认 `calc_type` 为 `PA_ENCODER`。
- 支持的 `mask_type` 取值包括 `UNDEFINED`、`NORM`、`ALIBI` 及压缩 mask 相关类型，具体请以当前 NNAL/ATB 版本为准。
- 若使用 GQA 模式，需要满足 `num_heads > num_kv_heads` 且 `num_heads % num_kv_heads == 0`。
- 在部分硬件与场景下，`head_size` 需要为 16 的倍数；为保证性能，建议设置为 64 的倍数。
