# （beta）at_npu::native::atb::npu_paged_cache_load

## 定义文件

op_plugin/include/atb_ops.h

## 函数原型

```cpp
std::tuple<at::Tensor, at::Tensor> at_npu::native::atb::npu_paged_cache_load(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& context_lens,
    const c10::optional<at::Tensor>& seq_starts, bool cumsum)

std::tuple<at::Tensor&, at::Tensor&> at_npu::native::atb::npu_paged_cache_load_out(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& context_lens,
    const c10::optional<at::Tensor>& seq_starts, bool cumsum,
    at::Tensor& key, at::Tensor& value)
```

## 功能说明

根据 `block_table` 中记录的 block ID 和 `context_lens` 中的序列长度信息，从分页管理的 `key_cache/value_cache` 中搬运并拼接出连续的 key/value 序列。该接口族覆盖基础输出和 `out` 变体。

## 参数说明

key_cache：Tensor 类型，cache 好的 key 张量。  
value_cache：Tensor 类型，cache 好的 value 张量。  
block_table：Tensor 类型，记录每个 batch 序列在 cache 中对应 block ID 的张量。  
context_lens：Tensor 类型，每个 batch 的序列长度，或其累加和。  
seq_starts：可选 Tensor 类型，每个 batch 在 `block_table` 中的起始位置。  
cumsum：bool 类型，表示 `context_lens` 是否为累加和。  
key：Tensor 类型，`out` 变体的 key 输出张量。  
value：Tensor 类型，`out` 变体的 value 输出张量。
