// csrc/AutogradFallback.cpp
#include <torch/extension.h>
#include <c10/util/Exception.h>

static auto customThrowFallback = [](
    const at::Tensor& var,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs) -> std::vector<at::Tensor> {
    
    TORCH_CHECK(false, 
        "\n========== ERROR ==========\n"
        "Backward pass attempted through npu operator!\n"
        "This operator does NOT support gradient computation.\n"
        "Please ensure you are not calling .backward() on tensors\n"
        "that depend on this operator.\n"
        "============================\n");
    return {};
};

TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
    m.impl("npu_add_rms_norm", customThrowFallback);
    m.impl("npu_apply_rotary_pos_emb", customThrowFallback);
    m.impl("npu_interleave_rope", customThrowFallback);
    m.impl("npu_moe_gating_top_k_softmax", customThrowFallback);
}