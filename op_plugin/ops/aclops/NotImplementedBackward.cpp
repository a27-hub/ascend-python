#include <torch/library.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>

// These npu_ operators do NOT have a backward (gradient) implementation.
//
// Goal: when the user backpropagates through them, print a WARNING that the
// operator has no backward support, WITHOUT interrupting training.

TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m)
{
    m.impl("npu_add_rms_norm", torch::autograd::basicAutogradNotImplementedFallback());
    m.impl("npu_apply_rotary_pos_emb", torch::autograd::basicAutogradNotImplementedFallback());
    m.impl("npu_interleave_rope", torch::autograd::basicAutogradNotImplementedFallback());
    m.impl("npu_moe_gating_top_k_softmax", torch::autograd::basicAutogradNotImplementedFallback());
}