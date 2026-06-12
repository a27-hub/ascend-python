#include <torch/library.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>

TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m)
{
    m.impl("npu_add_rms_norm", torch::autograd::autogradNotImplementedFallback());
    m.impl("npu_apply_rotary_pos_emb", torch::autograd::autogradNotImplementedFallback());
    m.impl("npu_interleave_rope", torch::autograd::autogradNotImplementedFallback());
    m.impl("npu_moe_gating_top_k_softmax", torch::autograd::autogradNotImplementedFallback());
}