# （beta）at_npu::native::atb::npu_paged_cache_load

## 产品支持情况

当前为 beta 接口。请以 `torch_npu` 同构环境中实际可用的 NNAL/ATB 版本及约束说明为准。

## 功能说明

根据 `block_table` 中记录的 block ID 和 `context_lens` 中的序列长度信息，从分页管理的 `key_cache/value_cache` 中搬运并拼接出连续的 key/value 序列。

当前文档覆盖的 C++ wrapper 变体包括：

- `at_npu::native::atb::npu_paged_cache_load`
- `at_npu::native::atb::npu_paged_cache_load_out`

## 定义文件

`op_plugin/include/atb_ops.h`

## 函数原型

```cpp
std::tuple<at::Tensor, at::Tensor> at_npu::native::atb::npu_paged_cache_load(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& context_lens,
    const c10::optional<at::Tensor>& seq_starts, bool cumsum);

std::tuple<at::Tensor&, at::Tensor&> at_npu::native::atb::npu_paged_cache_load_out(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& context_lens,
    const c10::optional<at::Tensor>& seq_starts, bool cumsum,
    at::Tensor& key, at::Tensor& value);
```

## 参数说明

`key_cache`、`value_cache`：cache 好的 key/value 张量。  
`block_table`：记录每个 batch 序列在 cache 中对应 block ID 的张量。  
`context_lens`：每个 batch 的序列长度，或其累加和。  
`seq_starts`：可选张量，表示每个 batch 在 `block_table` 中的起始位置。  
`cumsum`：表示 `context_lens` 是否为累加和。  

变体差异：

- `npu_paged_cache_load` 返回新分配的 `(key, value)`。
- `npu_paged_cache_load_out` 额外接收 `key` 和 `value` 作为输出张量。

## 返回值说明

- 基础变体返回 `(key, value)`。
- `*_out` 变体返回写回后的 `(key, value)` 张量引用。

## 约束说明

- `key_cache` 中 `num_heads * head_size_k` 需要按 32B 对齐。
- `value_cache` 中 `num_heads * head_size_v` 需要按 32B 对齐。
- `context_lens` 支持 `[batch]` 或 `[batch + 1]` 形状，具体含义由 `cumsum` 决定。
