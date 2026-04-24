#include "third_party/op-plugin/op_plugin/include/atb_ops.h"

#include <cstdint>
#include <iostream>
#include <string_view>

template <typename Fn>
inline void PrintSymbol(std::string_view name, Fn fn)
{
    auto address = reinterpret_cast<std::uintptr_t>(reinterpret_cast<const void *>(fn));
    std::cout << name << " -> 0x" << std::hex << address << std::dec << '\n';
}

int main()
{
    std::cout << "[ATB C++ Interface Smoke]" << '\n';
    PrintSymbol("_npu_paged_attention_quant", &at_npu::native::atb::_npu_paged_attention_quant);
    PrintSymbol("npu_fused_add_topk_div", &at_npu::native::atb::npu_fused_add_topk_div);
    PrintSymbol("npu_fused_add_topk_div_out", &at_npu::native::atb::npu_fused_add_topk_div_out);
    PrintSymbol("_npu_group_topk", &at_npu::native::atb::_npu_group_topk);
    PrintSymbol("_npu_matmul_add_fp32", &at_npu::native::atb::_npu_matmul_add_fp32);
    PrintSymbol("npu_mla_preprocess", &at_npu::native::atb::npu_mla_preprocess);
    PrintSymbol("npu_mla_preprocess_out", &at_npu::native::atb::npu_mla_preprocess_out);
    PrintSymbol("npu_multi_head_latent_attention", &at_npu::native::atb::npu_multi_head_latent_attention);
    PrintSymbol("npu_multi_head_latent_attention_out", &at_npu::native::atb::npu_multi_head_latent_attention_out);
    PrintSymbol("npu_multi_head_latent_attention_lse", &at_npu::native::atb::npu_multi_head_latent_attention_lse);
    PrintSymbol("_npu_multi_head_latent_attention_lse_get_workspace", &at_npu::native::atb::_npu_multi_head_latent_attention_lse_get_workspace);
    PrintSymbol("npu_multi_head_latent_attention_lse_out", &at_npu::native::atb::npu_multi_head_latent_attention_lse_out);
    PrintSymbol("_npu_paged_attention", &at_npu::native::atb::_npu_paged_attention);
    PrintSymbol("_npu_paged_attention_get_workspace", &at_npu::native::atb::_npu_paged_attention_get_workspace);
    PrintSymbol("_npu_paged_attention_v2", &at_npu::native::atb::_npu_paged_attention_v2);
    PrintSymbol("_npu_paged_attention_v2_get_workspace", &at_npu::native::atb::_npu_paged_attention_v2_get_workspace);
    PrintSymbol("_npu_paged_attention_mla", &at_npu::native::atb::_npu_paged_attention_mla);
    PrintSymbol("_npu_paged_attention_splitfuse", &at_npu::native::atb::_npu_paged_attention_splitfuse);
    PrintSymbol("npu_paged_cache_load_out", &at_npu::native::atb::npu_paged_cache_load_out);
    PrintSymbol("npu_paged_cache_load", &at_npu::native::atb::npu_paged_cache_load);
    PrintSymbol("_npu_quantize_per_tensor", &at_npu::native::atb::_npu_quantize_per_tensor);
    PrintSymbol("_npu_quant_rms_norm", &at_npu::native::atb::_npu_quant_rms_norm);
    PrintSymbol("_npu_reshape_and_cache", &at_npu::native::atb::_npu_reshape_and_cache);
    PrintSymbol("_npu_reshape_and_cache_siso", &at_npu::native::atb::_npu_reshape_and_cache_siso);
    PrintSymbol("npu_ring_mla", &at_npu::native::atb::npu_ring_mla);
    PrintSymbol("npu_ring_mla_out", &at_npu::native::atb::npu_ring_mla_out);
    PrintSymbol("_npu_rotary_embedding", &at_npu::native::atb::_npu_rotary_embedding);
    PrintSymbol("_npu_flash_attention", &at_npu::native::atb::_npu_flash_attention);
    PrintSymbol("_npu_flash_attention_v2", &at_npu::native::atb::_npu_flash_attention_v2);
    PrintSymbol("_npu_flash_attention_prefix_v2", &at_npu::native::atb::_npu_flash_attention_prefix_v2);
    PrintSymbol("npu_self_attention_prefix_encoder", &at_npu::native::atb::npu_self_attention_prefix_encoder);
    PrintSymbol("npu_self_attention_prefix_encoder_out", &at_npu::native::atb::npu_self_attention_prefix_encoder_out);
    PrintSymbol("_npu_flash_attention_qlens", &at_npu::native::atb::_npu_flash_attention_qlens);
    PrintSymbol("_npu_flash_attention_unpad", &at_npu::native::atb::_npu_flash_attention_unpad);
    std::cout << "count=34" << '\n';
    return 0;
}
