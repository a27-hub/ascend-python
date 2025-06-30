/**
 * @file QuantConvolutionKernelNpuOpApi.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor aclnnNotFoundConvolution(const at::Tensor &input)
{
    TORCH_CHECK(false, "aclnnQuantConvolution not found. " + OPS_ERROR(ErrCode::INTERNAL));
    return input;
}

at::Tensor convolution3d_quant(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
                                const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset,
                                const c10::optional<at::Tensor> &pertokenScaleOptional,
                                const at::IntArrayRef stride, const at::IntArrayRef padding, const at::IntArrayRef dilation,
                                bool transposed, const at::IntArrayRef output_padding, int64_t groups,
                                int64_t offsetx)
{
    DO_COMPATIBILITY(aclnnQuantConvolution, aclnnNotFoundConvolution(input));
    at::Tensor result = npu_preparation::apply_tensor_without_format(input); // Create output memory

    // calculate the output result of the NPU
    c10::SmallVector<int64_t, SIZE> out_size;
    out_size = op_infer::conv_npu_output_size(input, weight, bias, padding, output_padding, stride, dilation, groups,
                                              transposed);

    auto output = npu_preparation::apply_tensor_without_format(out_size, input.options().dtype(at::kBFloat16));
    EXEC_NPU_CMD(aclnnQuantConvolution, input, weight, bias, scale, offset, pertokenScaleOptional, stride, padding,
                 dilation, transposed, output_padding, groups, offsetx, "", output);
    return output;
}
} // namespace op_api