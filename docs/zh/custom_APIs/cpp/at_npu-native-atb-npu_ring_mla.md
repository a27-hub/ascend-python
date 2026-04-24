# at_npu::native::atb::npu_ring_mla

## 定义文件

op_plugin/include/atb_ops.h

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
    c10::optional<c10::string_view> calc_type_opt)

std::tuple<at::Tensor&, at::Tensor&> at_npu::native::atb::npu_ring_mla_out(
    const at::Tensor& q_nope, const at::Tensor& q_rope, const at::Tensor& k_nope,
    const at::Tensor& k_rope, const at::Tensor& value, const at::Tensor& mask,
    const at::Tensor& seqlen, int64_t head_num, int64_t kv_head_num,
    const c10::optional<at::Tensor>& pre_out, const c10::optional<at::Tensor>& prev_lse,
    double qk_scale, c10::optional<c10::string_view> kernel_type_opt,
    c10::optional<c10::string_view> mask_type_opt,
    c10::optional<c10::string_view> input_layout_opt,
    c10::optional<c10::string_view> calc_type_opt, at::Tensor& output, at::Tensor& softmax_lse)
```

## 功能说明

面向长序列场景的 ring MLA 计算接口，通过分块计算并合并中间结果降低 attention 场景中的显存开销。该接口族覆盖基础输出和 `out` 变体。

## 参数说明

q_nope：Tensor 类型，无位置编码的 query 张量。  
q_rope：Tensor 类型，带旋转位置编码的 query 张量。  
k_nope：Tensor 类型，无位置编码的 key 张量。  
k_rope：Tensor 类型，带旋转位置编码的 key 张量。  
value：Tensor 类型，value 张量。  
mask：Tensor 类型，mask 张量。  
seqlen：Tensor 类型，序列长度张量。  
head_num：int64_t 类型，query 头数。  
kv_head_num：int64_t 类型，KV 头数。  
pre_out：可选 Tensor 类型，前一块输出中间结果。  
prev_lse：可选 Tensor 类型，前一块 `lse` 中间结果。  
qk_scale：double 类型，缩放系数。  
kernel_type_opt、mask_type_opt、input_layout_opt、calc_type_opt：可选字符串枚举参数。  
output：Tensor 类型，`out` 变体的输出张量。  
softmax_lse：Tensor 类型，`out` 变体的 `lse` 输出张量。

## 支持的型号?

- <term>Atlas 800I A2 推理产品</term>
- <term>Atlas 800I A3 推理产品</term>
