# （beta）at_npu::native::atb::npu_multi_head_latent_attention

## 定义文件

op_plugin/include/atb_ops.h

## 函数原型

```cpp
at::Tensor at_npu::native::atb::npu_multi_head_latent_attention(
    const at::Tensor& q_nope, const at::Tensor& q_rope, const at::Tensor& ctkv,
    const at::Tensor& k_rope, const at::Tensor& block_tables,
    c10::SymIntArrayRef context_lens, int64_t head_num, double qk_scale,
    int64_t kv_headnum, const c10::optional<at::Tensor>& mask,
    c10::OptionalArrayRef<c10::SymInt> qseqlen,
    const c10::optional<at::Tensor>& qk_descale,
    const c10::optional<at::Tensor>& pv_descale,
    c10::optional<c10::string_view> mask_type_opt,
    c10::optional<c10::string_view> calc_type_opt,
    c10::optional<c10::string_view> cache_mode_opt)

at::Tensor& at_npu::native::atb::npu_multi_head_latent_attention_out(
    const at::Tensor& q_nope, const at::Tensor& q_rope, const at::Tensor& ctkv,
    const at::Tensor& k_rope, const at::Tensor& block_tables,
    c10::SymIntArrayRef context_lens, int64_t head_num, double qk_scale,
    int64_t kv_headnum, const c10::optional<at::Tensor>& mask,
    c10::OptionalArrayRef<c10::SymInt> qseqlen,
    const c10::optional<at::Tensor>& qk_descale,
    const c10::optional<at::Tensor>& pv_descale,
    c10::optional<c10::string_view> mask_type_opt,
    c10::optional<c10::string_view> calc_type_opt,
    c10::optional<c10::string_view> cache_mode_opt, at::Tensor& output)

std::tuple<at::Tensor, at::Tensor>
at_npu::native::atb::npu_multi_head_latent_attention_lse(
    const at::Tensor& q_nope, const at::Tensor& q_rope, const at::Tensor& ctkv,
    const at::Tensor& k_rope, const at::Tensor& block_tables,
    c10::SymIntArrayRef context_lens, int64_t head_num, double qk_scale,
    int64_t kv_headnum, bool return_lse, const c10::optional<at::Tensor>& mask,
    c10::OptionalArrayRef<c10::SymInt> qseqlen,
    const c10::optional<at::Tensor>& qk_descale,
    const c10::optional<at::Tensor>& pv_descale,
    c10::optional<c10::string_view> mask_type_opt,
    c10::optional<c10::string_view> calc_type_opt,
    c10::optional<c10::string_view> cache_mode_opt,
    const c10::optional<at::Tensor>& workspace)

at::Tensor at_npu::native::atb::_npu_multi_head_latent_attention_lse_get_workspace(
    const at::Tensor& q_nope, const at::Tensor& q_rope, const at::Tensor& ctkv,
    const at::Tensor& k_rope, const at::Tensor& block_tables,
    c10::SymIntArrayRef context_lens, int64_t head_num, double qk_scale,
    int64_t kv_headnum, bool return_lse, const c10::optional<at::Tensor>& mask,
    c10::OptionalArrayRef<c10::SymInt> qseqlen,
    const c10::optional<at::Tensor>& qk_descale,
    const c10::optional<at::Tensor>& pv_descale,
    c10::optional<c10::string_view> mask_type_opt,
    c10::optional<c10::string_view> calc_type_opt,
    c10::optional<c10::string_view> cache_mode_opt)

std::tuple<at::Tensor&, at::Tensor&>
at_npu::native::atb::npu_multi_head_latent_attention_lse_out(
    const at::Tensor& q_nope, const at::Tensor& q_rope, const at::Tensor& ctkv,
    const at::Tensor& k_rope, const at::Tensor& block_tables,
    c10::SymIntArrayRef context_lens, int64_t head_num, double qk_scale,
    int64_t kv_headnum, bool return_lse, const c10::optional<at::Tensor>& mask,
    c10::OptionalArrayRef<c10::SymInt> qseqlen,
    const c10::optional<at::Tensor>& qk_descale,
    const c10::optional<at::Tensor>& pv_descale,
    c10::optional<c10::string_view> mask_type_opt,
    c10::optional<c10::string_view> calc_type_opt,
    c10::optional<c10::string_view> cache_mode_opt,
    const c10::optional<at::Tensor>& workspace, at::Tensor& output, at::Tensor& lse)
```

## 功能说明

MLA 场景下，使用分页管理的 KV Cache 计算 attention score，支持分离 `q_nope/q_rope` 与 `ctkv/k_rope` 输入。该接口族同时覆盖基础输出、`out` 输出、`lse` 输出以及 workspace 获取变体。

## 参数说明

q_nope：Tensor 类型，无位置编码的 query 张量。  
q_rope：Tensor 类型，带旋转位置编码的 query 张量。  
ctkv：Tensor 类型，无位置编码的 KV Cache 张量。  
k_rope：Tensor 类型，带旋转位置编码的 key 张量。  
block_tables：Tensor 类型，每个 query 对应的 KV Cache block 映射表。  
context_lens：SymIntArrayRef 类型，每个 query 对应的上下文长度。  
head_num：int64_t 类型，query 头数。  
qk_scale：double 类型，`Q*K^T` 后的缩放系数。  
kv_headnum：int64_t 类型，KV 头数。  
return_lse：bool 类型，`lse` 变体中用于指示是否返回 `lse`。  
mask：可选 Tensor 类型，mask 张量。  
qseqlen：可选序列长度信息。  
qk_descale、pv_descale：可选 Tensor 类型，量化/反量化相关输入。  
mask_type_opt、calc_type_opt、cache_mode_opt：可选字符串枚举参数。  
workspace：可选 Tensor 类型，`lse` 变体相关 workspace 输入。  
output：Tensor 类型，`out` 变体的输出张量。  
lse：Tensor 类型，`lse` 输出张量。
