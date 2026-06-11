// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
#include <torch/library.h>
#include <vector>
#include <string>
namespace {
void not_implemented_backward(
    const c10::OperatorHandle& op,
    torch::jit::Stack* /* stack */)
{
    TORCH_CHECK(
        false,
        op.operator_name().name,
        ": backward is not supported. ",
        "This operator does not have a gradient registered. ",
        "If you need to use this operator in training, ",
        "please contact the Huawei Ascend development team.");
}
} // namespace
TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
    // 逐个注册
    m.impl("npu_add_rms_norm", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
    m.impl("npu_apply_rotary_pos_emb", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
    m.impl("npu_interleave_rope", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
    m.impl("npu_moe_gating_top_k_softmax", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
}

