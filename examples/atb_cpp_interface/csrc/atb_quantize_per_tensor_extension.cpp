#include <ATen/ATen.h>
#include <torch/extension.h>

#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "op_plugin/include/atb_ops.h"

namespace {

using OptionalTensor = std::optional<at::Tensor>;
using OptionalString = std::optional<std::string>;
using IntList = std::vector<int64_t>;
using OptionalIntList = std::optional<std::vector<int64_t>>;

void check_npu_tensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be an NPU tensor, got device ",
        tensor.device());
}

void check_optional_npu_tensor(const OptionalTensor& tensor, const char* name)
{
    if (tensor.has_value()) {
        check_npu_tensor(*tensor, name);
    }
}

std::vector<c10::SymInt> to_symints(const IntList& values)
{
    std::vector<c10::SymInt> result;
    result.reserve(values.size());
    for (const auto value : values) {
        result.emplace_back(value);
    }
    return result;
}

c10::optional<c10::string_view> to_string_view(const OptionalString& value)
{
    if (!value.has_value()) {
        return c10::nullopt;
    }
    return c10::string_view(*value);
}

c10::optional<at::Tensor> to_c10_optional_tensor(const OptionalTensor& tensor)
{
    if (!tensor.has_value()) {
        return c10::nullopt;
    }
    return *tensor;
}

at::Tensor run_npu_paged_attention_quant(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value,
    const at::Tensor& block_table,
    const at::Tensor& context_lens,
    int64_t quant_type,
    int64_t outdata_type,
    const at::Tensor& k_descale,
    const at::Tensor& v_descale)
{
    check_npu_tensor(query, "query");
    at::Tensor out = at::zeros_like(query, query.options().dtype(static_cast<at::ScalarType>(outdata_type)));
    at_npu::native::atb::_npu_paged_attention_quant(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        num_heads,
        scale_value,
        block_table,
        context_lens,
        quant_type,
        outdata_type,
        k_descale,
        v_descale,
        out);
    return out;
}

std::tuple<at::Tensor, at::Tensor> run_npu_fused_add_topk_div(
    const at::Tensor& x,
    const at::Tensor& add_num,
    const OptionalTensor& mapping_num,
    const OptionalTensor& mapping_table,
    const OptionalString& activation_type,
    int64_t group_num,
    int64_t group_topk,
    int64_t n,
    int64_t k,
    bool is_norm,
    double scale,
    bool enable_expert_mapping)
{
    check_npu_tensor(x, "x");
    return at_npu::native::atb::npu_fused_add_topk_div(
        x,
        add_num,
        to_c10_optional_tensor(mapping_num),
        to_c10_optional_tensor(mapping_table),
        to_string_view(activation_type),
        group_num,
        group_topk,
        n,
        k,
        is_norm,
        scale,
        enable_expert_mapping);
}

std::tuple<at::Tensor, at::Tensor> run_npu_fused_add_topk_div_out(
    const at::Tensor& x,
    const at::Tensor& add_num,
    const OptionalTensor& mapping_num,
    const OptionalTensor& mapping_table,
    const OptionalString& activation_type,
    int64_t group_num,
    int64_t group_topk,
    int64_t n,
    int64_t k,
    bool is_norm,
    double scale,
    bool enable_expert_mapping,
    at::Tensor y,
    at::Tensor indices)
{
    at_npu::native::atb::npu_fused_add_topk_div_out(
        x,
        add_num,
        to_c10_optional_tensor(mapping_num),
        to_c10_optional_tensor(mapping_table),
        to_string_view(activation_type),
        group_num,
        group_topk,
        n,
        k,
        is_norm,
        scale,
        enable_expert_mapping,
        y,
        indices);
    return {y, indices};
}

at::Tensor run_npu_group_topk(const at::Tensor& self, int64_t k, int64_t group_num, int64_t n)
{
    at::Tensor out = self.clone();
    at_npu::native::atb::_npu_group_topk(out, k, group_num, n);
    return out;
}

at::Tensor run_npu_matmul_add_fp32(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& c)
{
    at::Tensor out = c.clone();
    at_npu::native::atb::_npu_matmul_add_fp32(x, weight, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> run_npu_mla_preprocess(
    const at::Tensor& input,
    const at::Tensor& gamma0,
    const at::Tensor& beta0,
    const at::Tensor& wdqkv,
    const at::Tensor& descale0,
    const at::Tensor& gamma1,
    const at::Tensor& beta1,
    const at::Tensor& wuq,
    const at::Tensor& descale1,
    const at::Tensor& gamma2,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& wuk,
    const at::Tensor& kv_cache,
    const at::Tensor& kv_cache_rope,
    const at::Tensor& slotmapping,
    const OptionalTensor& quant_scale0,
    const OptionalTensor& quant_offset0,
    const OptionalTensor& bias0,
    const OptionalTensor& quant_scale1,
    const OptionalTensor& quant_offset1,
    const OptionalTensor& bias1,
    const OptionalTensor& ctkv_scale,
    const OptionalTensor& q_nope_scale,
    const OptionalString& cache_mode,
    const OptionalString& quant_mode)
{
    return at_npu::native::atb::npu_mla_preprocess(
        input,
        gamma0,
        beta0,
        wdqkv,
        descale0,
        gamma1,
        beta1,
        wuq,
        descale1,
        gamma2,
        cos,
        sin,
        wuk,
        kv_cache,
        kv_cache_rope,
        slotmapping,
        to_c10_optional_tensor(quant_scale0),
        to_c10_optional_tensor(quant_offset0),
        to_c10_optional_tensor(bias0),
        to_c10_optional_tensor(quant_scale1),
        to_c10_optional_tensor(quant_offset1),
        to_c10_optional_tensor(bias1),
        to_c10_optional_tensor(ctkv_scale),
        to_c10_optional_tensor(q_nope_scale),
        to_string_view(cache_mode),
        to_string_view(quant_mode));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> run_npu_mla_preprocess_out(
    const at::Tensor& input,
    const at::Tensor& gamma0,
    const at::Tensor& beta0,
    const at::Tensor& wdqkv,
    const at::Tensor& descale0,
    const at::Tensor& gamma1,
    const at::Tensor& beta1,
    const at::Tensor& wuq,
    const at::Tensor& descale1,
    const at::Tensor& gamma2,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& wuk,
    const at::Tensor& kv_cache,
    const at::Tensor& kv_cache_rope,
    const at::Tensor& slotmapping,
    const OptionalTensor& quant_scale0,
    const OptionalTensor& quant_offset0,
    const OptionalTensor& bias0,
    const OptionalTensor& quant_scale1,
    const OptionalTensor& quant_offset1,
    const OptionalTensor& bias1,
    const OptionalTensor& ctkv_scale,
    const OptionalTensor& q_nope_scale,
    const OptionalString& cache_mode,
    const OptionalString& quant_mode,
    at::Tensor q_out0,
    at::Tensor kv_cache_out0,
    at::Tensor q_out1,
    at::Tensor kv_cache_out1)
{
    at_npu::native::atb::npu_mla_preprocess_out(
        input,
        gamma0,
        beta0,
        wdqkv,
        descale0,
        gamma1,
        beta1,
        wuq,
        descale1,
        gamma2,
        cos,
        sin,
        wuk,
        kv_cache,
        kv_cache_rope,
        slotmapping,
        to_c10_optional_tensor(quant_scale0),
        to_c10_optional_tensor(quant_offset0),
        to_c10_optional_tensor(bias0),
        to_c10_optional_tensor(quant_scale1),
        to_c10_optional_tensor(quant_offset1),
        to_c10_optional_tensor(bias1),
        to_c10_optional_tensor(ctkv_scale),
        to_c10_optional_tensor(q_nope_scale),
        to_string_view(cache_mode),
        to_string_view(quant_mode),
        q_out0,
        kv_cache_out0,
        q_out1,
        kv_cache_out1);
    return {q_out0, kv_cache_out0, q_out1, kv_cache_out1};
}

at::Tensor run_npu_multi_head_latent_attention(
    const at::Tensor& q_nope,
    const at::Tensor& q_rope,
    const at::Tensor& ctkv,
    const at::Tensor& k_rope,
    const at::Tensor& block_tables,
    const IntList& context_lens,
    int64_t head_num,
    double qk_scale,
    int64_t kv_headnum,
    const OptionalTensor& mask,
    const OptionalIntList& qseqlen,
    const OptionalTensor& qk_descale,
    const OptionalTensor& pv_descale,
    const OptionalString& mask_type,
    const OptionalString& calc_type,
    const OptionalString& cache_mode)
{
    auto context_lens_sym = to_symints(context_lens);
    c10::SymIntArrayRef context_ref(context_lens_sym);
    std::optional<std::vector<c10::SymInt>> qseqlen_sym;
    if (qseqlen.has_value()) {
        qseqlen_sym = to_symints(*qseqlen);
    }
    c10::OptionalArrayRef<c10::SymInt> qseqlen_ref =
        qseqlen_sym.has_value() ? c10::OptionalArrayRef<c10::SymInt>(*qseqlen_sym) : c10::nullopt;
    return at_npu::native::atb::npu_multi_head_latent_attention(
        q_nope,
        q_rope,
        ctkv,
        k_rope,
        block_tables,
        context_ref,
        head_num,
        qk_scale,
        kv_headnum,
        to_c10_optional_tensor(mask),
        qseqlen_ref,
        to_c10_optional_tensor(qk_descale),
        to_c10_optional_tensor(pv_descale),
        to_string_view(mask_type),
        to_string_view(calc_type),
        to_string_view(cache_mode));
}

at::Tensor run_npu_multi_head_latent_attention_out(
    const at::Tensor& q_nope,
    const at::Tensor& q_rope,
    const at::Tensor& ctkv,
    const at::Tensor& k_rope,
    const at::Tensor& block_tables,
    const IntList& context_lens,
    int64_t head_num,
    double qk_scale,
    int64_t kv_headnum,
    const OptionalTensor& mask,
    const OptionalIntList& qseqlen,
    const OptionalTensor& qk_descale,
    const OptionalTensor& pv_descale,
    const OptionalString& mask_type,
    const OptionalString& calc_type,
    const OptionalString& cache_mode,
    at::Tensor output)
{
    auto context_lens_sym = to_symints(context_lens);
    c10::SymIntArrayRef context_ref(context_lens_sym);
    std::optional<std::vector<c10::SymInt>> qseqlen_sym;
    if (qseqlen.has_value()) {
        qseqlen_sym = to_symints(*qseqlen);
    }
    c10::OptionalArrayRef<c10::SymInt> qseqlen_ref =
        qseqlen_sym.has_value() ? c10::OptionalArrayRef<c10::SymInt>(*qseqlen_sym) : c10::nullopt;
    at_npu::native::atb::npu_multi_head_latent_attention_out(
        q_nope,
        q_rope,
        ctkv,
        k_rope,
        block_tables,
        context_ref,
        head_num,
        qk_scale,
        kv_headnum,
        to_c10_optional_tensor(mask),
        qseqlen_ref,
        to_c10_optional_tensor(qk_descale),
        to_c10_optional_tensor(pv_descale),
        to_string_view(mask_type),
        to_string_view(calc_type),
        to_string_view(cache_mode),
        output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> run_npu_multi_head_latent_attention_lse(
    const at::Tensor& q_nope,
    const at::Tensor& q_rope,
    const at::Tensor& ctkv,
    const at::Tensor& k_rope,
    const at::Tensor& block_tables,
    const IntList& context_lens,
    int64_t head_num,
    double qk_scale,
    int64_t kv_headnum,
    bool return_lse,
    const OptionalTensor& mask,
    const OptionalIntList& qseqlen,
    const OptionalTensor& qk_descale,
    const OptionalTensor& pv_descale,
    const OptionalString& mask_type,
    const OptionalString& calc_type,
    const OptionalString& cache_mode,
    const OptionalTensor& workspace)
{
    auto context_lens_sym = to_symints(context_lens);
    c10::SymIntArrayRef context_ref(context_lens_sym);
    std::optional<std::vector<c10::SymInt>> qseqlen_sym;
    if (qseqlen.has_value()) {
        qseqlen_sym = to_symints(*qseqlen);
    }
    c10::OptionalArrayRef<c10::SymInt> qseqlen_ref =
        qseqlen_sym.has_value() ? c10::OptionalArrayRef<c10::SymInt>(*qseqlen_sym) : c10::nullopt;
    return at_npu::native::atb::npu_multi_head_latent_attention_lse(
        q_nope,
        q_rope,
        ctkv,
        k_rope,
        block_tables,
        context_ref,
        head_num,
        qk_scale,
        kv_headnum,
        return_lse,
        to_c10_optional_tensor(mask),
        qseqlen_ref,
        to_c10_optional_tensor(qk_descale),
        to_c10_optional_tensor(pv_descale),
        to_string_view(mask_type),
        to_string_view(calc_type),
        to_string_view(cache_mode),
        to_c10_optional_tensor(workspace));
}

at::Tensor run_npu_multi_head_latent_attention_lse_get_workspace(
    const at::Tensor& q_nope,
    const at::Tensor& q_rope,
    const at::Tensor& ctkv,
    const at::Tensor& k_rope,
    const at::Tensor& block_tables,
    const IntList& context_lens,
    int64_t head_num,
    double qk_scale,
    int64_t kv_headnum,
    bool return_lse,
    const OptionalTensor& mask,
    const OptionalIntList& qseqlen,
    const OptionalTensor& qk_descale,
    const OptionalTensor& pv_descale,
    const OptionalString& mask_type,
    const OptionalString& calc_type,
    const OptionalString& cache_mode)
{
    auto context_lens_sym = to_symints(context_lens);
    c10::SymIntArrayRef context_ref(context_lens_sym);
    std::optional<std::vector<c10::SymInt>> qseqlen_sym;
    if (qseqlen.has_value()) {
        qseqlen_sym = to_symints(*qseqlen);
    }
    c10::OptionalArrayRef<c10::SymInt> qseqlen_ref =
        qseqlen_sym.has_value() ? c10::OptionalArrayRef<c10::SymInt>(*qseqlen_sym) : c10::nullopt;
    return at_npu::native::atb::_npu_multi_head_latent_attention_lse_get_workspace(
        q_nope,
        q_rope,
        ctkv,
        k_rope,
        block_tables,
        context_ref,
        head_num,
        qk_scale,
        kv_headnum,
        return_lse,
        to_c10_optional_tensor(mask),
        qseqlen_ref,
        to_c10_optional_tensor(qk_descale),
        to_c10_optional_tensor(pv_descale),
        to_string_view(mask_type),
        to_string_view(calc_type),
        to_string_view(cache_mode));
}

std::tuple<at::Tensor, at::Tensor> run_npu_multi_head_latent_attention_lse_out(
    const at::Tensor& q_nope,
    const at::Tensor& q_rope,
    const at::Tensor& ctkv,
    const at::Tensor& k_rope,
    const at::Tensor& block_tables,
    const IntList& context_lens,
    int64_t head_num,
    double qk_scale,
    int64_t kv_headnum,
    bool return_lse,
    const OptionalTensor& mask,
    const OptionalIntList& qseqlen,
    const OptionalTensor& qk_descale,
    const OptionalTensor& pv_descale,
    const OptionalString& mask_type,
    const OptionalString& calc_type,
    const OptionalString& cache_mode,
    const OptionalTensor& workspace,
    at::Tensor output,
    at::Tensor lse)
{
    auto context_lens_sym = to_symints(context_lens);
    c10::SymIntArrayRef context_ref(context_lens_sym);
    std::optional<std::vector<c10::SymInt>> qseqlen_sym;
    if (qseqlen.has_value()) {
        qseqlen_sym = to_symints(*qseqlen);
    }
    c10::OptionalArrayRef<c10::SymInt> qseqlen_ref =
        qseqlen_sym.has_value() ? c10::OptionalArrayRef<c10::SymInt>(*qseqlen_sym) : c10::nullopt;
    at_npu::native::atb::npu_multi_head_latent_attention_lse_out(
        q_nope,
        q_rope,
        ctkv,
        k_rope,
        block_tables,
        context_ref,
        head_num,
        qk_scale,
        kv_headnum,
        return_lse,
        to_c10_optional_tensor(mask),
        qseqlen_ref,
        to_c10_optional_tensor(qk_descale),
        to_c10_optional_tensor(pv_descale),
        to_string_view(mask_type),
        to_string_view(calc_type),
        to_string_view(cache_mode),
        to_c10_optional_tensor(workspace),
        output,
        lse);
    return {output, lse};
}

at::Tensor run_npu_paged_attention(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value,
    const at::Tensor& block_table,
    const at::Tensor& context_lens,
    const OptionalTensor& workspace)
{
    at::Tensor out = at::zeros(
        {query.size(0), query.size(1), value_cache.size(3)},
        query.options().dtype(value_cache.scalar_type()));
    at_npu::native::atb::_npu_paged_attention(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        num_heads,
        scale_value,
        block_table,
        context_lens,
        out,
        to_c10_optional_tensor(workspace));
    return out;
}

at::Tensor run_npu_paged_attention_get_workspace(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value,
    const at::Tensor& block_table,
    const at::Tensor& context_lens)
{
    at::Tensor out = at::zeros(
        {query.size(0), query.size(1), value_cache.size(3)},
        query.options().dtype(value_cache.scalar_type()));
    return at_npu::native::atb::_npu_paged_attention_get_workspace(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        num_heads,
        scale_value,
        block_table,
        context_lens,
        out);
}

at::Tensor run_npu_paged_attention_v2(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& block_table,
    const IntList& context_lens,
    const OptionalTensor& value_cache,
    const OptionalTensor& mask,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value,
    int64_t mask_type,
    const OptionalTensor& workspace)
{
    auto context_lens_sym = to_symints(context_lens);
    c10::SymIntArrayRef context_ref(context_lens_sym);
    TORCH_CHECK(value_cache.has_value(), "value_cache is required for this example");
    at::Tensor out = at::empty(
        {query.size(0), query.size(1), value_cache->size(3)},
        query.options().dtype(value_cache->scalar_type()));
    at_npu::native::atb::_npu_paged_attention_v2(
        query,
        key_cache,
        block_table,
        context_ref,
        to_c10_optional_tensor(value_cache),
        to_c10_optional_tensor(mask),
        num_kv_heads,
        num_heads,
        scale_value,
        mask_type,
        to_c10_optional_tensor(workspace),
        out);
    return out;
}

at::Tensor run_npu_paged_attention_v2_get_workspace(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& block_table,
    const IntList& context_lens,
    const OptionalTensor& value_cache,
    const OptionalTensor& mask,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value,
    int64_t mask_type)
{
    auto context_lens_sym = to_symints(context_lens);
    c10::SymIntArrayRef context_ref(context_lens_sym);
    TORCH_CHECK(value_cache.has_value(), "value_cache is required for this example");
    at::Tensor out = at::empty(
        {query.size(0), query.size(1), value_cache->size(3)},
        query.options().dtype(value_cache->scalar_type()));
    return at_npu::native::atb::_npu_paged_attention_v2_get_workspace(
        query,
        key_cache,
        block_table,
        context_ref,
        to_c10_optional_tensor(value_cache),
        to_c10_optional_tensor(mask),
        num_kv_heads,
        num_heads,
        scale_value,
        mask_type,
        out);
}

at::Tensor run_npu_paged_attention_mla(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value,
    const at::Tensor& block_table,
    const at::Tensor& context_lens,
    int64_t mla_vheadsize)
{
    at::Tensor out = at::zeros(
        {query.size(0), query.size(1), mla_vheadsize},
        query.options());
    at_npu::native::atb::_npu_paged_attention_mla(
        query,
        key_cache,
        num_kv_heads,
        num_heads,
        scale_value,
        block_table,
        context_lens,
        mla_vheadsize,
        out);
    return out;
}

at::Tensor run_npu_paged_attention_splitfuse(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& context_lens,
    const at::Tensor& mask,
    const at::Tensor& seq_len,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value)
{
    at::Tensor out = at::zeros(
        {query.size(0), query.size(1), value_cache.size(3)},
        query.options().dtype(value_cache.scalar_type()));
    at_npu::native::atb::_npu_paged_attention_splitfuse(
        query,
        key_cache,
        value_cache,
        block_table,
        context_lens,
        mask,
        seq_len,
        num_kv_heads,
        num_heads,
        scale_value,
        out);
    return out;
}

std::tuple<at::Tensor, at::Tensor> run_npu_paged_cache_load(
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& context_lens,
    const OptionalTensor& seq_starts,
    bool cumsum)
{
    return at_npu::native::atb::npu_paged_cache_load(
        key_cache,
        value_cache,
        block_table,
        context_lens,
        to_c10_optional_tensor(seq_starts),
        cumsum);
}

std::tuple<at::Tensor, at::Tensor> run_npu_paged_cache_load_out(
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& context_lens,
    const OptionalTensor& seq_starts,
    bool cumsum,
    at::Tensor key,
    at::Tensor value)
{
    at_npu::native::atb::npu_paged_cache_load_out(
        key_cache,
        value_cache,
        block_table,
        context_lens,
        to_c10_optional_tensor(seq_starts),
        cumsum,
        key,
        value);
    return {key, value};
}

at::Tensor run_npu_quantize_per_tensor(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& zero_point)
{
    at::Tensor y = at::zeros(x.sizes(), x.options().dtype(at::kChar));
    at_npu::native::atb::_npu_quantize_per_tensor(x, scale, zero_point, y);
    return y;
}

at::Tensor run_npu_quant_rms_norm(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    const at::Tensor& scale,
    const at::Tensor& offset,
    double eps)
{
    at::Tensor output = at::zeros(x.sizes(), x.options().dtype(at::kChar));
    at_npu::native::atb::_npu_quant_rms_norm(x, gamma, beta, scale, offset, output, eps);
    return output;
}

std::tuple<at::Tensor, at::Tensor> run_npu_reshape_and_cache(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& slot_indices)
{
    at::Tensor key_cache_out = key_cache.clone();
    at::Tensor value_cache_out = value_cache.clone();
    at_npu::native::atb::_npu_reshape_and_cache(key, value, key_cache_out, value_cache_out, slot_indices);
    return {key_cache_out, value_cache_out};
}

at::Tensor run_npu_reshape_and_cache_siso(
    const at::Tensor& key,
    const at::Tensor& key_cache,
    const at::Tensor& slot_indices)
{
    at::Tensor key_cache_out = key_cache.clone();
    at_npu::native::atb::_npu_reshape_and_cache_siso(key, key_cache_out, slot_indices);
    return key_cache_out;
}

std::tuple<at::Tensor, at::Tensor> run_npu_ring_mla(
    const at::Tensor& q_nope,
    const at::Tensor& q_rope,
    const at::Tensor& k_nope,
    const at::Tensor& k_rope,
    const at::Tensor& value,
    const at::Tensor& mask,
    const at::Tensor& seqlen,
    int64_t head_num,
    int64_t kv_head_num,
    const OptionalTensor& pre_out,
    const OptionalTensor& prev_lse,
    double qk_scale,
    const OptionalString& kernel_type,
    const OptionalString& mask_type,
    const OptionalString& input_layout,
    const OptionalString& calc_type)
{
    return at_npu::native::atb::npu_ring_mla(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        value,
        mask,
        seqlen,
        head_num,
        kv_head_num,
        to_c10_optional_tensor(pre_out),
        to_c10_optional_tensor(prev_lse),
        qk_scale,
        to_string_view(kernel_type),
        to_string_view(mask_type),
        to_string_view(input_layout),
        to_string_view(calc_type));
}

std::tuple<at::Tensor, at::Tensor> run_npu_ring_mla_out(
    const at::Tensor& q_nope,
    const at::Tensor& q_rope,
    const at::Tensor& k_nope,
    const at::Tensor& k_rope,
    const at::Tensor& value,
    const at::Tensor& mask,
    const at::Tensor& seqlen,
    int64_t head_num,
    int64_t kv_head_num,
    const OptionalTensor& pre_out,
    const OptionalTensor& prev_lse,
    double qk_scale,
    const OptionalString& kernel_type,
    const OptionalString& mask_type,
    const OptionalString& input_layout,
    const OptionalString& calc_type,
    at::Tensor output,
    at::Tensor softmax_lse)
{
    at_npu::native::atb::npu_ring_mla_out(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        value,
        mask,
        seqlen,
        head_num,
        kv_head_num,
        to_c10_optional_tensor(pre_out),
        to_c10_optional_tensor(prev_lse),
        qk_scale,
        to_string_view(kernel_type),
        to_string_view(mask_type),
        to_string_view(input_layout),
        to_string_view(calc_type),
        output,
        softmax_lse);
    return {output, softmax_lse};
}

std::tuple<at::Tensor, at::Tensor> run_npu_rotary_embedding(
    const at::Tensor& positions,
    const at::Tensor& query,
    const at::Tensor& key,
    int64_t head_size,
    const at::Tensor& cos_sin_cache,
    bool is_neox_style)
{
    at::Tensor query_out = query.clone();
    at::Tensor key_out = key.clone();
    at_npu::native::atb::_npu_rotary_embedding(
        positions,
        query_out,
        key_out,
        head_size,
        cos_sin_cache,
        is_neox_style);
    return {query_out, key_out};
}

at::Tensor run_npu_flash_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& mask,
    const at::Tensor& seq_len,
    double scale_value,
    int64_t num_heads,
    int64_t num_kv_heads)
{
    at::Tensor out = at::empty(
        {query.size(0), query.size(1), value.size(2)},
        query.options().dtype(value.scalar_type()));
    at_npu::native::atb::_npu_flash_attention(
        query,
        key,
        value,
        mask,
        seq_len,
        scale_value,
        num_heads,
        num_kv_heads,
        out);
    return out;
}

at::Tensor run_npu_flash_attention_v2(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const IntList& seq_len,
    const OptionalTensor& mask,
    const OptionalTensor& slopes,
    int64_t kernel_type,
    int64_t mask_type,
    double scale_value,
    int64_t num_heads,
    int64_t num_kv_heads)
{
    auto seq_len_sym = to_symints(seq_len);
    c10::SymIntArrayRef seq_ref(seq_len_sym);
    at::Tensor out = at::empty(
        {query.size(0), query.size(1), value.size(2)},
        query.options().dtype(value.scalar_type()));
    at_npu::native::atb::_npu_flash_attention_v2(
        query,
        key,
        value,
        seq_ref,
        to_c10_optional_tensor(mask),
        to_c10_optional_tensor(slopes),
        kernel_type,
        mask_type,
        scale_value,
        num_heads,
        num_kv_heads,
        out);
    return out;
}

at::Tensor run_npu_flash_attention_prefix_v2(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& mask,
    const IntList& seq_len,
    const IntList& context_lens,
    const OptionalTensor& slopes,
    int64_t kernel_type,
    int64_t mask_type,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value)
{
    auto seq_len_sym = to_symints(seq_len);
    auto context_lens_sym = to_symints(context_lens);
    c10::SymIntArrayRef seq_ref(seq_len_sym);
    c10::SymIntArrayRef context_ref(context_lens_sym);
    at::Tensor out = at::empty_like(query);
    at_npu::native::atb::_npu_flash_attention_prefix_v2(
        query,
        key_cache,
        value_cache,
        block_table,
        mask,
        seq_ref,
        context_ref,
        to_c10_optional_tensor(slopes),
        kernel_type,
        mask_type,
        num_kv_heads,
        num_heads,
        scale_value,
        out);
    return out;
}

at::Tensor run_npu_self_attention_prefix_encoder(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& block_tables,
    const IntList& seqlen,
    const IntList& kv_seqlen,
    int64_t q_headnum,
    double qk_scale,
    int64_t kv_headnum,
    const OptionalTensor& mask,
    const OptionalTensor& slopes,
    const OptionalString& mask_type)
{
    auto seqlen_sym = to_symints(seqlen);
    auto kv_seqlen_sym = to_symints(kv_seqlen);
    c10::SymIntArrayRef seqlen_ref(seqlen_sym);
    c10::SymIntArrayRef kv_seqlen_ref(kv_seqlen_sym);
    return at_npu::native::atb::npu_self_attention_prefix_encoder(
        query,
        key,
        value,
        block_tables,
        seqlen_ref,
        kv_seqlen_ref,
        q_headnum,
        qk_scale,
        kv_headnum,
        to_c10_optional_tensor(mask),
        to_c10_optional_tensor(slopes),
        to_string_view(mask_type));
}

at::Tensor run_npu_self_attention_prefix_encoder_out(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& block_tables,
    const IntList& seqlen,
    const IntList& kv_seqlen,
    int64_t q_headnum,
    double qk_scale,
    int64_t kv_headnum,
    const OptionalTensor& mask,
    const OptionalTensor& slopes,
    const OptionalString& mask_type,
    at::Tensor output)
{
    auto seqlen_sym = to_symints(seqlen);
    auto kv_seqlen_sym = to_symints(kv_seqlen);
    c10::SymIntArrayRef seqlen_ref(seqlen_sym);
    c10::SymIntArrayRef kv_seqlen_ref(kv_seqlen_sym);
    at_npu::native::atb::npu_self_attention_prefix_encoder_out(
        query,
        key,
        value,
        block_tables,
        seqlen_ref,
        kv_seqlen_ref,
        q_headnum,
        qk_scale,
        kv_headnum,
        to_c10_optional_tensor(mask),
        to_c10_optional_tensor(slopes),
        to_string_view(mask_type),
        output);
    return output;
}

at::Tensor run_npu_flash_attention_qlens(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& mask,
    const at::Tensor& seq_len,
    const at::Tensor& context_lens,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value)
{
    at::Tensor out = at::empty(
        {query.size(0), query.size(1), value_cache.size(3)},
        query.options().dtype(value_cache.scalar_type()));
    at_npu::native::atb::_npu_flash_attention_qlens(
        query,
        key_cache,
        value_cache,
        block_table,
        mask,
        seq_len,
        context_lens,
        num_kv_heads,
        num_heads,
        scale_value,
        out);
    return out;
}

at::Tensor run_npu_flash_attention_unpad(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_len,
    double scale_value,
    int64_t num_heads,
    int64_t num_kv_heads)
{
    at::Tensor out = at::empty(
        {query.size(0), query.size(1), value.size(2)},
        query.options().dtype(value.scalar_type()));
    at_npu::native::atb::_npu_flash_attention_unpad(
        query,
        key,
        value,
        seq_len,
        scale_value,
        num_heads,
        num_kv_heads,
        out);
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_npu_paged_attention_quant", &run_npu_paged_attention_quant);
    m.def("npu_fused_add_topk_div", &run_npu_fused_add_topk_div);
    m.def("npu_fused_add_topk_div_out", &run_npu_fused_add_topk_div_out);
    m.def("_npu_group_topk", &run_npu_group_topk);
    m.def("_npu_matmul_add_fp32", &run_npu_matmul_add_fp32);
    m.def("npu_mla_preprocess", &run_npu_mla_preprocess);
    m.def("npu_mla_preprocess_out", &run_npu_mla_preprocess_out);
    m.def("npu_multi_head_latent_attention", &run_npu_multi_head_latent_attention);
    m.def("npu_multi_head_latent_attention_out", &run_npu_multi_head_latent_attention_out);
    m.def("npu_multi_head_latent_attention_lse", &run_npu_multi_head_latent_attention_lse);
    m.def("_npu_multi_head_latent_attention_lse_get_workspace", &run_npu_multi_head_latent_attention_lse_get_workspace);
    m.def("npu_multi_head_latent_attention_lse_out", &run_npu_multi_head_latent_attention_lse_out);
    m.def("_npu_paged_attention", &run_npu_paged_attention);
    m.def("_npu_paged_attention_get_workspace", &run_npu_paged_attention_get_workspace);
    m.def("_npu_paged_attention_v2", &run_npu_paged_attention_v2);
    m.def("_npu_paged_attention_v2_get_workspace", &run_npu_paged_attention_v2_get_workspace);
    m.def("_npu_paged_attention_mla", &run_npu_paged_attention_mla);
    m.def("_npu_paged_attention_splitfuse", &run_npu_paged_attention_splitfuse);
    m.def("npu_paged_cache_load", &run_npu_paged_cache_load);
    m.def("npu_paged_cache_load_out", &run_npu_paged_cache_load_out);
    m.def("_npu_quantize_per_tensor", &run_npu_quantize_per_tensor);
    m.def("_npu_quant_rms_norm", &run_npu_quant_rms_norm);
    m.def("_npu_reshape_and_cache", &run_npu_reshape_and_cache);
    m.def("_npu_reshape_and_cache_siso", &run_npu_reshape_and_cache_siso);
    m.def("npu_ring_mla", &run_npu_ring_mla);
    m.def("npu_ring_mla_out", &run_npu_ring_mla_out);
    m.def("_npu_rotary_embedding", &run_npu_rotary_embedding);
    m.def("_npu_flash_attention", &run_npu_flash_attention);
    m.def("_npu_flash_attention_v2", &run_npu_flash_attention_v2);
    m.def("_npu_flash_attention_prefix_v2", &run_npu_flash_attention_prefix_v2);
    m.def("npu_self_attention_prefix_encoder", &run_npu_self_attention_prefix_encoder);
    m.def("npu_self_attention_prefix_encoder_out", &run_npu_self_attention_prefix_encoder_out);
    m.def("_npu_flash_attention_qlens", &run_npu_flash_attention_qlens);
    m.def("_npu_flash_attention_unpad", &run_npu_flash_attention_unpad);
}
