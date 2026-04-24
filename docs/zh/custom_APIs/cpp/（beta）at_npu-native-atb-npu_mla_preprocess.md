# （beta）at_npu::native::atb::npu_mla_preprocess

## 产品支持情况

当前为 beta 接口。请以 `torch_npu` 同构环境中实际可用的 NNAL/ATB 版本及约束说明为准。

## 功能说明

融合 MLA 场景下 PagedAttention 输入预处理全过程，包括从隐状态输入开始经过 RMSNorm、反量化、MatMul、RoPE、ReshapeAndCache 等步骤。

当前文档覆盖的 C++ wrapper 变体包括：

- `at_npu::native::atb::npu_mla_preprocess`
- `at_npu::native::atb::npu_mla_preprocess_out`

## 定义文件

`op_plugin/include/atb_ops.h`

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
    c10::optional<c10::string_view> quant_mode_opt);
```

## 参数说明

`input`：隐状态输入。  
`gamma0`、`beta0`、`gamma1`、`beta1`、`gamma2`：RMSNorm 相关输入。  
`wdqkv`、`descale0`、`wuq`、`descale1`、`wuk`：MatMul/反量化相关输入。  
`cos`、`sin`：RoPE 相关输入。  
`kv_cache`、`kv_cache_rope`：写入 cache 的目标张量。  
`slotmapping`：token 到 cache slot 的映射。  
`quant_scale0`、`quant_offset0`、`bias0`、`quant_scale1`、`quant_offset1`、`bias1`、`ctkv_scale`、`q_nope_scale`：量化/偏置相关可选输入。  
`cache_mode_opt`：可选 cache 模式。  
`quant_mode_opt`：可选量化模式。  

变体差异：

- `npu_mla_preprocess` 返回四个输出张量。
- `npu_mla_preprocess_out` 额外接收 `q_out0`、`kv_cache_out0`、`q_out1`、`kv_cache_out1`。

## 返回值说明

- 基础变体返回 `(q_out0, kv_cache_out0, q_out1, kv_cache_out1)`。
- `*_out` 变体返回写回后的四个输出张量引用。

## 约束说明

- `tokenNum <= 1024`。
- `blockSize <= 128` 或 `blockSize = 256`。
- `cache_mode=int8_nzcache` 时，`blockSize = 128`。
- `cache_mode`、`quant_mode` 的具体可选值及各输入形状约束，请以当前 NNAL/ATB 版本为准。
