#include <torch/library.h>
#include <vector>
#include <string>

namespace {

void not_implemented_backward(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack)
{
    TORCH_WARN(op.operator_name().name,
               ":backward is not supported.",
               "This operator does not have a gradient registered.");
    size_t num_returns = op.schema().returns().size();
    for (size_t i = 0; i < num_returns; ++i) {
        stack->push_back(c10::nullopt);
    }
}

} // namespace

TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
    m.impl("npu_add_rms_norm", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
    m.impl("npu_apply_rotary_pos_emb", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
    m.impl("npu_interleave_rope", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
    m.impl("npu_moe_gating_top_k_softmax", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
}