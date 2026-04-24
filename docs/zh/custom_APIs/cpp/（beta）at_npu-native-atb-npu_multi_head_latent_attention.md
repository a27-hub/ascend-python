# （beta）at_npu::native::atb::npu_multi_head_latent_attention

## 产品支持情况

当前为 beta 接口。请以 `torch_npu` 同构环境中实际可用的 NNAL/ATB 版本及约束说明为准。

## 功能说明

MLA 场景下，使用分页管理的 KV Cache 计算 attention score，支持分离 `q_nope/q_rope` 与 `ctkv/k_rope` 的输入形式。

当前文档覆盖的 C++ wrapper 变体包括：

- `at_npu::native::atb::npu_multi_head_latent_attention`
- `at_npu::native::atb::npu_multi_head_latent_attention_out`
- `at_npu::native::atb::npu_multi_head_latent_attention_lse`
- `at_npu::native::atb::_npu_multi_head_latent_attention_lse_get_workspace`
- `at_npu::native::atb::npu_multi_head_latent_attention_lse_out`

## 定义文件

`op_plugin/include/atb_ops.h`

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
    c10::optional<c10::string_view> cache_mode_opt);

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
    c10::optional<c10::string_view> cache_mode_opt, at::Tensor& output);

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
    const c10::optional<at::Tensor>& workspace);
```

## 参数说明

`q_nope`：无位置编码的 query 张量。  
`q_rope`：带旋转位置编码的 query 张量。  
`ctkv`：无位置编码的 KV Cache 张量。  
`k_rope`：带旋转位置编码的 key 张量。  
`block_tables`：每个 query 对应的 KV Cache block 映射表。  
`context_lens`：每个 query 对应的上下文长度。  
`head_num`：query 头数。  
`qk_scale`：`Q*K^T` 后的缩放系数。  
`kv_headnum`：KV 头数。  
`mask`：可选 mask 张量。  
`qseqlen`：可选序列长度信息，部分 cache 模式下需要传入。  
`qk_descale`、`pv_descale`：量化/反量化相关的可选输入。  
`mask_type_opt`、`calc_type_opt`、`cache_mode_opt`：可选字符串枚举参数，用于指定 mask、计算类型和 cache 模式。  

变体差异：

- `npu_multi_head_latent_attention` 返回 attention 输出。
- `npu_multi_head_latent_attention_out` 额外接收 `output` 作为输出张量。
- `npu_multi_head_latent_attention_lse` 返回 `(output, lse)`。
- `_npu_multi_head_latent_attention_lse_get_workspace` 返回 LSE 变体所需的 workspace。
- `npu_multi_head_latent_attention_lse_out` 额外接收 `workspace`、`output` 和 `lse`。

## 返回值说明

- 基础变体返回 `Tensor` 类型 attention 输出。
- `*_out` 变体返回写回后的输出张量引用。
- `*_lse` 变体返回输出张量和 `lse` 张量。
- `*_get_workspace` 变体返回 workspace 张量。

## 约束说明

- `block_size <= 128`，建议取 `128`。
- `batch <= 8192`。
- `cache_mode=int8_nzcache` 时，不支持 `num_heads = 128`。
- `mask_type`、`calc_type`、`cache_mode` 的具体可选值及输入形状约束，请与当前 NNAL/ATB 版本保持一致。
