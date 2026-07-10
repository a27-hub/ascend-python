#include <torch/library.h>
#include <torch/tensor.h> 
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <c10/util/Exception.h>

static auto customThrowFallback = [](const torch::Tensor& var,
const std::vectortorch::Tensor& inputs,
const std::vectortorch::Tensor& outputs) {
TORCH_CHECK(false, "\n========== ERROR ==========\n"
"Backward pass attempted through npu operator!\n"
"This operator does NOT support gradient computation.\n"
"Please ensure you are not calling .backward() on tensors\n"
"that depend on this operator.\n"
"============================\n");
return std::vectortorch::Tensor();
};

TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
m.impl("npu_add_rms_norm", customThrowFallback);
m.impl("npu_apply_rotary_pos_emb", customThrowFallback);
m.impl("npu_interleave_rope", customThrowFallback);
m.impl("npu_moe_gating_top_k_softmax", customThrowFallback);
}
