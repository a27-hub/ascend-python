# （beta）at_npu::native::atb::npu_ring_mla

## 产品支持情况

当前为 beta 接口。已知支持 Atlas 800I A2 推理产品和 Atlas 800I A3 推理产品。其他产品形态请以当前 NNAL/ATB 版本支持情况为准。

## 功能说明

面向长序列场景的 ring MLA 计算接口。通过分块计算并合并中间结果，降低长序列 attention 场景中的显存开销。

当前文档覆盖的 C++ wrapper 变体包括：

- `at_npu::native::atb::npu_ring_mla`
- `at_npu::native::atb::npu_ring_mla_out`

## 定义文件

`op_plugin/include/atb_ops.h`

## 函数原型

```cpp
std::tuple<at::Tensor, at::Tensor> at_npu::native::atb::npu_ring_mla(
    const at::Tensor& q_nope, const at::Tensor& q_rope, const at::Tensor& k_nope,
    const at::Tensor& k_rope, const at::Tensor& value, const at::Tensor& mask,
    const at::Tensor& seqlen, int64_t head_num, int64_t kv_head_num,
    const c10::optional<at::Tensor>& pre_out, const c10::optional<at::Tensor>& prev_lse,
    double qk_scale, c10::optional<c10::string_view> kernel_type_opt,
    c10::optional<c10::string_view> mask_type_opt,
    c10::optional<c10::string_view> input_layout_opt,
    c10::optional<c10::string_view> calc_type_opt);

std::tuple<at::Tensor&, at::Tensor&> at_npu::native::atb::npu_ring_mla_out(
    const at::Tensor& q_nope, const at::Tensor& q_rope, const at::Tensor& k_nope,
    const at::Tensor& k_rope, const at::Tensor& value, const at::Tensor& mask,
    const at::Tensor& seqlen, int64_t head_num, int64_t kv_head_num,
    const c10::optional<at::Tensor>& pre_out, const c10::optional<at::Tensor>& prev_lse,
    double qk_scale, c10::optional<c10::string_view> kernel_type_opt,
    c10::optional<c10::string_view> mask_type_opt,
    c10::optional<c10::string_view> input_layout_opt,
    c10::optional<c10::string_view> calc_type_opt,
    at::Tensor& output, at::Tensor& softmax_lse);
```

## 参数说明

`q_nope`、`q_rope`：query 的无位置编码和旋转位置编码分量。  
`k_nope`、`k_rope`：key 的无位置编码和旋转位置编码分量。  
`value`：value 张量。  
`mask`：mask 张量。  
`seqlen`：序列长度张量。  
`head_num`：query 头数。  
`kv_head_num`：KV 头数。  
`pre_out`、`prev_lse`：可选的前一块中间结果输入。  
`qk_scale`：缩放系数。  
`kernel_type_opt`、`mask_type_opt`、`input_layout_opt`、`calc_type_opt`：可选字符串枚举参数。  

变体差异：

- `npu_ring_mla` 返回 `(output, softmax_lse)`。
- `npu_ring_mla_out` 额外接收 `output` 和 `softmax_lse` 作为输出张量。

## 返回值说明

- 基础变体返回 `(output, softmax_lse)`。
- `*_out` 变体返回写回后的 `(output, softmax_lse)` 张量引用。

## 约束说明

- `maskType = MASK_TYPE_TRIU` 时才使用 `mask`。
- `inputLayout` 仅支持 `TYPE_BSND`。
- 对于每个 batch，需要满足 `kvSeqLen >= qSeqLen` 或 `kvSeqLen = 0`，且 `qSeqLen` 不可为 `0`。
- `kernel_type`、`mask_type`、`calc_type` 的可选值请以当前 NNAL/ATB 版本为准。
