# （beta）at_npu::native::atb::npu_mla_preprocess

## 定义文件

op_plugin/include/atb_ops.h

## 函数原型

```cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
at_npu::native::atb::npu_mla_preprocess(
    const at::Tensor& input, const at::Tensor& gamma0, const at::Tensor& beta0,
    const at::Tensor& wdqkv, const at::Tensor& descale0, const at::Tensor& gamma1,
    const at::Tensor& beta1, const at::Tensor& wuq, const at::Tensor& descale1,
    const at::Tensor& gamma2, const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& wuk, const at::Tensor& kv_cache, const at::Tensor& kv_cache_rope,
    const at::Tensor& slotmapping, const c10::optional<at::Tensor>& quant_scale0,
    const c10::optional<at::Tensor>& quant_offset0, const c10::optional<at::Tensor>& bias0,
    const c10::optional<at::Tensor>& quant_scale1, const c10::optional<at::Tensor>& quant_offset1,
    const c10::optional<at::Tensor>& bias1, const c10::optional<at::Tensor>& ctkv_scale,
    const c10::optional<at::Tensor>& q_nope_scale, c10::optional<c10::string_view> cache_mode_opt,
    c10::optional<c10::string_view> quant_mode_opt)

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
at_npu::native::atb::npu_mla_preprocess_out(
    const at::Tensor& input, const at::Tensor& gamma0, const at::Tensor& beta0,
    const at::Tensor& wdqkv, const at::Tensor& descale0, const at::Tensor& gamma1,
    const at::Tensor& beta1, const at::Tensor& wuq, const at::Tensor& descale1,
    const at::Tensor& gamma2, const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& wuk, const at::Tensor& kv_cache, const at::Tensor& kv_cache_rope,
    const at::Tensor& slotmapping, const c10::optional<at::Tensor>& quant_scale0,
    const c10::optional<at::Tensor>& quant_offset0, const c10::optional<at::Tensor>& bias0,
    const c10::optional<at::Tensor>& quant_scale1, const c10::optional<at::Tensor>& quant_offset1,
    const c10::optional<at::Tensor>& bias1, const c10::optional<at::Tensor>& ctkv_scale,
    const c10::optional<at::Tensor>& q_nope_scale, c10::optional<c10::string_view> cache_mode_opt,
    c10::optional<c10::string_view> quant_mode_opt, at::Tensor& q_out0, at::Tensor& kv_cache_out0,
    at::Tensor& q_out1, at::Tensor& kv_cache_out1)
```

## 功能说明

融合 MLA 场景下 PagedAttention 输入预处理全过程，包括 RMSNorm、反量化、MatMul、RoPE 和 ReshapeAndCache 等步骤。该接口族覆盖基础输出和 `out` 变体。

## 参数说明

input：Tensor 类型，隐状态输入。  
gamma0、beta0、gamma1、beta1、gamma2：Tensor 类型，RMSNorm 相关输入。  
wdqkv、descale0、wuq、descale1、wuk：Tensor 类型，MatMul/反量化相关输入。  
cos、sin：Tensor 类型，RoPE 相关输入。  
kv_cache、kv_cache_rope：Tensor 类型，cache 输出相关张量。  
slotmapping：Tensor 类型，token 到 cache slot 的映射。  
quant_scale0、quant_offset0、bias0、quant_scale1、quant_offset1、bias1、ctkv_scale、q_nope_scale：可选 Tensor 类型，量化/偏置相关输入。  
cache_mode_opt：可选字符串枚举参数。  
quant_mode_opt：可选字符串枚举参数。  
q_out0、kv_cache_out0、q_out1、kv_cache_out1：Tensor 类型，`out` 变体的输出张量。
