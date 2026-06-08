// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.

#include <torch/library.h>

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

// Register AutogradPrivateUse1 intercept kernel for npu_add_rms_norm.
// When autograd tries to backprop through this op, this kernel throws
// a clear error instead of silently falling through VariableFallbackKernel
// and producing incorrect gradients.
TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
    m.impl("npu_add_rms_norm", not_implemented_backward);
}
