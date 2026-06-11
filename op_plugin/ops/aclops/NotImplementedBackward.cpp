// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
#include <torch/library.h>
#include <vector>
#include <string>
#include <iostream>

namespace {

void not_implemented_backward(
    const c10::OperatorHandle& op,
    torch::jit::Stack* /* stack */)
{
    std::cerr << op.operator_name().name 
              << ": backward is not supported. "
              << "This operator does not have a gradient registered."
              << std::endl;
}

} // namespace

TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
    m.impl("npu_add_rms_norm", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
    m.impl("npu_apply_rotary_pos_emb", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
    m.impl("npu_interleave_rope", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
    m.impl("npu_moe_gating_top_k_softmax", torch::CppFunction::makeFromBoxedFunction<&not_implemented_backward>());
}