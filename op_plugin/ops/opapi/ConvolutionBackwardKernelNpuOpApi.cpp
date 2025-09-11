// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

const int IDX_0 = 0;
const int IDX_1 = 1;
const int IDX_2 = 2;
const int MIN_ELEMENT_3D = 3;
const int CONV2D_ATTR_SIZE = 2;
const int CONV3D_ATTR_SIZE = 3;
const int OUTPUT_MASK_SIZE = 3;

using npu_preparation = at_npu::native::OpPreparation;
using tensor_list3 = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

static inline c10::SmallVector<int64_t, op_infer::N> expand_dim_if_needed(
    at::IntArrayRef list_param,
    const char *param_name,
    int64_t expected_dim)
{
    if (list_param.size() == 1) {
        c10::SmallVector<int64_t, op_infer::N> expand_dim_param_vec;
        for (int64_t i = 0; i < expected_dim; i++) {
            expand_dim_param_vec.emplace_back(list_param[0]);
        }
        return expand_dim_param_vec;
    } else {
        return op_plugin::utils::convert_array_to_vector(list_param);
    }
}

static tensor_list3 _calc_convolution_backward(const at::Tensor &grad_output, const at::Tensor &input,
                                               const at::Tensor &weight, c10::optional<at::IntArrayRef> bias_sizes_opt,
                                               at::IntArrayRef stride, at::IntArrayRef padding,
                                               at::IntArrayRef dilation, bool transposed,
                                               at::IntArrayRef output_padding, int64_t groups,
                                               ::std::array<bool, 3> output_mask)
{
    int64_t k = weight.ndimension();
    int64_t dim = k - 2;
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());

    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    ASCEND_LOGI("_calc_convolution_backward exec with jit compile: %d, allow internal format: %d",
                is_jit_enable, is_allow_internel_format);
    if ((is_allow_internel_format || is_jit_enable) && (dim != 3)) {
        return acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                            transposed, output_padding, groups, output_mask);
    }

    c10::SmallVector<int64_t, op_infer::N> stride_expand = expand_dim_if_needed(stride, "stride", dim);
    stride = at::IntArrayRef(stride_expand);

    c10::SmallVector<int64_t, op_infer::N> padding_expand = expand_dim_if_needed(padding, "padding", dim);
    padding = at::IntArrayRef(padding_expand);

    c10::SmallVector<int64_t, op_infer::N> dilation_expand = expand_dim_if_needed(dilation, "dilation", dim);
    dilation = at::IntArrayRef(dilation_expand);

    c10::SmallVector<int64_t, op_infer::N> output_padding_expand =
        expand_dim_if_needed(output_padding, "output_padding", dim);
    output_padding = at::IntArrayRef(output_padding_expand);

    auto outputSizes =
        op_infer::conv2d_backward_npu_output_size(input, grad_output, weight);

    // construct the output tensor of the NPU
    at::Tensor gradInput;
    at::Tensor gradWeight;
    at::Tensor gradBias;

    gradInput = npu_preparation::apply_tensor_without_format(std::get<0>(outputSizes), input.options());
    gradWeight = npu_preparation::apply_tensor_without_format(std::get<1>(outputSizes), weight.options());
    if (output_mask[2]) {
        gradBias = npu_preparation::apply_tensor_without_format(*bias_sizes_opt, grad_output.options());
    } else {
        gradBias = npu_preparation::apply_tensor_without_format(std::get<2>(outputSizes), grad_output.options());
    }

    EXEC_NPU_CMD(aclnnConvolutionBackward, grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                 transposed, output_padding, groups, output_mask, cube_math_type, gradInput, gradWeight, gradBias);

    FLOP_COUNT(FlopCounter::conv_backward_flop, grad_output, input, weight, transposed, output_mask, gradInput, gradWeight);
    return std::make_tuple(std::move(gradInput), std::move(gradWeight), std::move(gradBias));
}

// length of output_mask is 3
tensor_list3 convolution_backward(const at::Tensor &grad_output, const at::Tensor &input, const at::Tensor &weight,
                                  c10::optional<at::IntArrayRef> bias_sizes_opt, at::IntArrayRef stride,
                                  at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                                  at::IntArrayRef output_padding, int64_t groups, ::std::array<bool, 3> output_mask)
{
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));
    op_plugin::utils::check_input_same_type_as_parameters(input, weight);
    return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed,
                                      output_padding, groups, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_tbc_backward(const at::Tensor &self, const at::Tensor &input,
                                                                 const at::Tensor &weight, const at::Tensor &bias,
                                                                 int64_t pad)
{
    DO_COMPATIBILITY(aclnnConvTbcBackward, acl_op::conv_tbc_backward(self, input, weight, bias, pad));
    // construct other inputs of the NPU
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
    at::IntArrayRef stride = {1, 1};
    at::IntArrayRef padding = {0, pad};
    at::IntArrayRef dilation = {1, 1};
    int64_t groups = 1;
    std::array<bool, 3> grad_input_mask = {1, 1, 1};
    // calculate outputSizes of every output
    auto outputSizes =
        op_infer::conv2d_backward_tbc_output_size(input, self, weight);

    // construct the output tensor of the NPU
    at::Tensor gradInput = npu_preparation::apply_tensor_without_format(std::get<0>(outputSizes), input.options());
    at::Tensor gradWeight = npu_preparation::apply_tensor_without_format(std::get<1>(outputSizes), weight.options());
    at::Tensor gradBias = npu_preparation::apply_tensor_without_format(std::get<2>(outputSizes), self.options());
    // execute hostapi
    EXEC_NPU_CMD(aclnnConvTbcBackward, self, input, weight, bias, pad, cube_math_type, gradInput, gradWeight, gradBias);
    return std::make_tuple(gradInput, gradWeight, gradBias);
}

tensor_list3 slow_conv_transpose2d_backward(const at::Tensor &grad_output, const at::Tensor &input,
                                            const at::Tensor &weight, at::IntArrayRef kernel_size,
                                            at::IntArrayRef stride, at::IntArrayRef padding,
                                            at::IntArrayRef output_padding, at::IntArrayRef dilation,
                                            std::array<bool, 3> output_mask)
{
    int64_t groups = 1;
    bool transposed = true;
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));
    return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed,
                                      output_padding, groups, output_mask);
}

tensor_list3 slow_conv_dilated2d_backward(const at::Tensor &grad_output, const at::Tensor &input,
                                          const at::Tensor &weight, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                          at::IntArrayRef padding, at::IntArrayRef dilation,
                                          std::array<bool, 3> output_mask)
{
    at::IntArrayRef output_padding = {0, 0};
    int64_t groups = 1;
    bool transposed = true;
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));
    return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed,
                                      output_padding, groups, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _slow_conv2d_backward(const at::Tensor &grad_output,
                                                                     const at::Tensor &input, const at::Tensor &weight,
                                                                     at::IntArrayRef kernel_size,
                                                                     at::IntArrayRef stride, at::IntArrayRef padding,
                                                                     std::array<bool, 3> output_mask)
{
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    at::IntArrayRef dilation = {1, 1};
    int64_t groups = 1;
    bool transposed = false;
    at::IntArrayRef output_padding = {0, 0};
    auto w_sizes = weight.sizes();
    at::Tensor orig_weight = npu_preparation::apply_tensor_without_format(w_sizes, weight.options());
    int64_t s1 = w_sizes[0];
    // get the product of w_sizes from the 1 index to the last
    int64_t s2 = c10::multiply_integers(w_sizes.slice(1));
    TORCH_CHECK(kernel_size[0] * kernel_size[1] != 0, "kernel_size should not be zero", OPS_ERROR(ErrCode::PARAM));
    s2 = s2 / (kernel_size[0] * kernel_size[1]);
    c10::SmallVector<int64_t, SIZE> slow_weight_size = {s1, s2, kernel_size[0], kernel_size[1]};
    weight.resize_(slow_weight_size);
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));

    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());

    /* k == 5 and groups > 3 currently unsupported by the binary file
      CheckForbidInternalFormat = False: turn on private format��CheckJitDisable = False: turn on JitCompile
    */
    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    ASCEND_LOGI("_slow_conv2d_backward exec with jit compile: %d, allow internal format: %d",
                is_jit_enable, is_allow_internel_format);
    if (is_allow_internel_format || is_jit_enable) {
        return acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                            transposed, output_padding, groups, output_mask);
    }

    auto outputSizes =
        op_infer::conv2d_backward_npu_output_size(input, grad_output, weight);

    // construct the output tensor of the NPU
    at::Tensor gradInput;
    at::Tensor gradWeight;
    at::Tensor gradBias;

    gradInput = npu_preparation::apply_tensor_without_format(std::get<0>(outputSizes), input.options());
    gradWeight = npu_preparation::apply_tensor_without_format(std::get<1>(outputSizes), weight.options());
    gradBias = npu_preparation::apply_tensor_without_format(std::get<2>(outputSizes), grad_output.options());

    EXEC_NPU_CMD(aclnnConvolutionBackward, grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                 transposed, output_padding, groups, output_mask, cube_math_type, gradInput, gradWeight, gradBias);
    auto orig_sizes = orig_weight.sizes();
    gradWeight.resize_(orig_sizes);
    return std::make_tuple(std::move(gradInput), std::move(gradWeight), std::move(gradBias));
}

tensor_list3 convolution_backward_overrideable(const at::Tensor &grad_output, const at::Tensor &input,
                                               const at::Tensor &weight, c10::IntArrayRef stride,
                                               c10::IntArrayRef padding, c10::IntArrayRef dilation, bool transposed,
                                               c10::IntArrayRef output_padding, int64_t groups,
                                               std::array<bool, 3> output_mask)
{
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));
    return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed,
                                      output_padding, groups, output_mask);
}

static at::Tensor _calc_convolution(const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups)
{
    int64_t k = weight.ndimension();
    int64_t inputK = input.ndimension();
    int64_t dim = k - 2;  // Subtract nonspatial dimensions: 2
    bool unBatch = false;

    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    // CheckForbidInternalFormat = False: turn on private formatCheckJitDisable = False: turn on JitCompile
    ASCEND_LOGI("_calc_convolution exec with jit compile: %d, allow internal format: %d",
        is_jit_enable,
        is_allow_internel_format);
    if ((is_allow_internel_format || is_jit_enable)) {
        return acl_op::_convolution(input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            false,
            false,
            false,
            false);
    }
    // Conv1D: 1, Conv2D: 2, Conv3D: 3
    if (dim != 1 && dim != 2 && dim != 3) {
        return at::Tensor();
    }

    c10::SmallVector<int64_t, op_infer::N> stride_expand = expand_dim_if_needed(stride, "stride", dim);
    stride = at::IntArrayRef(stride_expand);

    c10::SmallVector<int64_t, op_infer::N> padding_expand = expand_dim_if_needed(padding, "padding", dim);
    padding = at::IntArrayRef(padding_expand);

    c10::SmallVector<int64_t, op_infer::N> dilation_expand = expand_dim_if_needed(dilation, "dilation", dim);
    dilation = at::IntArrayRef(dilation_expand);

    c10::SmallVector<int64_t, op_infer::N> output_padding_expend = expand_dim_if_needed(output_padding, "output_padding", dim);
    output_padding = at::IntArrayRef(output_padding_expend);

    c10::SmallVector<int64_t, SIZE> out_size;

    out_size = op_infer::conv_npu_output_size(
        input, weight, bias, padding, output_padding, stride, dilation, groups, transposed);

    auto output = npu_preparation::apply_tensor_without_format(out_size, input.options());
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
    EXEC_NPU_CMD(aclnnConvolution,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output,
        cube_math_type);

    // input dim = 3 while conv2D: 2
    if (dim == 2 && inputK == 3) {
        c10::SmallVector<int64_t, SIZE> squeeze_size = {output.size(1), output.size(2), output.size(3)};
        output.resize_(squeeze_size);

        c10::SmallVector<int64_t, SIZE> squeeze_size_input = {input.size(1), input.size(2), input.size(3)};
        input.resize_(squeeze_size_input);
    }

    FLOP_COUNT(FlopCounter::conv_flop, input, weight, transposed, output);
    return output;
}

at::Tensor &conv_transpose2d_backward_input_out_nocheck(at::Tensor &grad_input, const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(stride.size() >= CONV2D_ATTR_SIZE,
        "stride has to contain more than 2 elements, but got ",
        stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= CONV2D_ATTR_SIZE,
        "padding has to contain more than 2 elements, but got ",
        padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= CONV2D_ATTR_SIZE,
        "dilation has to contain more than 2 elements, but got ",
        dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    const c10::optional<at::Tensor> bias = at::Tensor();
    grad_input =
        _calc_convolution(grad_output, weight, bias, stride, padding, dilation, false, output_padding, groups);
    return grad_input;
}

at::Tensor &conv_transpose2d_backward_weight_out_nocheck(at::Tensor &grad_weight, const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(stride.size() >= CONV2D_ATTR_SIZE,
        "stride has to contain more than 2 elements, but got ",
        stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= CONV2D_ATTR_SIZE,
        "padding has to contain more than 2 elements, but got ",
        padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= CONV2D_ATTR_SIZE,
        "dilation has to contain more than 2 elements, but got ",
        dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    bool transposed = true;
    std::array<bool, OUTPUT_MASK_SIZE> output_mask = {false, true, false};

    auto result = convolution_backward(grad_output,
        input,
        weight,
        bias_sizes_opt,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask);
    grad_weight = std::get<1>(result);
    return grad_weight;
}

at::Tensor &conv_transpose2d_backward_bias_out_nocheck(at::Tensor &grad_bias, const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(grad_output.dim() >= CONV2D_ATTR_SIZE,
        "grad_output has to be more than 2D, but got Tensor of dimension ",
        grad_output.dim(),
        OPS_ERROR(ErrCode::PARAM));
    at::Tensor grad_view = grad_output.contiguous().view({grad_output.size(0), grad_output.size(1), -1});
    op_api::sum_out(grad_view, c10::SmallVector<int64_t, N>{0, 2}, false, grad_view.scalar_type(), grad_bias);
    return grad_bias;
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &> conv_transpose2d_backward_out_nocheck(at::Tensor &grad_input,
    at::Tensor &grad_weight, at::Tensor &grad_bias, const at::Tensor &input, const at::Tensor &grad_output,
    const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride,
    at::IntArrayRef dilation, int64_t groups, std::array<bool, OUTPUT_MASK_SIZE> output_mask)
{
    if (output_mask[IDX_0]) {
        conv_transpose2d_backward_input_out_nocheck(
            grad_input, input, grad_output, weight, padding, output_padding, stride, dilation, groups);
    }
    if (output_mask[IDX_1]) {
        conv_transpose2d_backward_weight_out_nocheck(
            grad_weight, input, grad_output, weight, padding, output_padding, stride, dilation, groups);
    }
    if (output_mask[IDX_2]) {
        conv_transpose2d_backward_bias_out_nocheck(
            grad_bias, input, grad_output, weight, padding, output_padding, stride, dilation, groups);
    }
    return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose2d_backward(const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool, OUTPUT_MASK_SIZE> output_mask)
{
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;

    if (output_mask[IDX_0]) {
        int64_t grad_input_format = input.dtype() == at::kHalf ? ACL_FORMAT_NC1HWC0 : ACL_FORMAT_ND;
        grad_input = npu_preparation::apply_tensor_with_format(input, grad_input_format);
    }
    if (output_mask[IDX_1]) {
        grad_weight = npu_preparation::apply_tensor_with_format(
            weight.sizes(), weight.options().dtype(at::kFloat), npu_preparation::get_tensor_npu_format(weight));
    }
    if (output_mask[IDX_2]) {
        grad_bias =
            npu_preparation::apply_tensor_with_format({grad_output.size(1)}, grad_output.options(), ACL_FORMAT_NCHW);
    }

    conv_transpose2d_backward_out_nocheck(grad_input,
        grad_weight,
        grad_bias,
        input,
        grad_output,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        output_mask);
    return std::tie(grad_input, grad_weight, grad_bias);
}

at::Tensor &conv_transpose3d_backward_input_out_nocheck(at::Tensor &grad_input, const at::Tensor &grad_output,
    const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride,
    at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(stride.size() >= MIN_ELEMENT_3D,
        "stride has to contain more than 3 elements, but got ",
        stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= MIN_ELEMENT_3D,
        "padding has to contain more than 3 elements, but got ",
        padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= MIN_ELEMENT_3D,
        "dilation has to contain more than 3 elements, but got ",
        dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    const c10::optional<at::Tensor> bias = at::Tensor();
    grad_input =
        _calc_convolution(grad_output, weight, bias, stride, padding, dilation, false, output_padding, groups);
    return grad_input;
}

at::Tensor &conv_transpose3d_backward_weight_out_nocheck(at::Tensor &grad_weight, const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(stride.size() >= MIN_ELEMENT_3D,
        "stride has to contain more than 3 elements, but got ",
        stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= MIN_ELEMENT_3D,
        "padding has to contain more than 3 elements, but got ",
        padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= MIN_ELEMENT_3D,
        "dilation has to contain more than 3 elements, but got ",
        dilation.size(),
        OPS_ERROR(ErrCode::PARAM));

    std::array<bool, OUTPUT_MASK_SIZE> output_mask = {false, true, false};
    at::IntArrayRef input_size = weight.sizes();
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    bool transposed = true;

    auto result = convolution_backward(grad_output,
        input,
        weight,
        bias_sizes_opt,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask);
    grad_weight = std::get<1>(result);
    return grad_weight;
}

at::Tensor &conv_transpose3d_backward_bias_out_nocheck(at::Tensor &grad_bias, const at::Tensor &grad_output)
{
    TORCH_CHECK(grad_output.dim() >= MIN_ELEMENT_3D,
        "grad_output has to be more than 3D, but got Tensor of dimension ",
        grad_output.dim(),
        OPS_ERROR(ErrCode::PARAM));
    at::Tensor gradView =
        grad_output.contiguous().view({grad_output.size(0), grad_output.size(1), grad_output.size(2), -1});
    op_api::sum_out(gradView, c10::SmallVector<int64_t, N>{0, 2, 3}, false, gradView.scalar_type(), grad_bias);
    return grad_bias;
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &> conv_transpose3d_backward_out_nocheck(at::Tensor &grad_input,
    at::Tensor &grad_weight, at::Tensor &grad_bias, const at::Tensor &input, const at::Tensor &grad_output,
    const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride,
    at::IntArrayRef dilation, int64_t groups, std::array<bool, OUTPUT_MASK_SIZE> output_mask)
{
    if (output_mask[IDX_0]) {
        conv_transpose3d_backward_input_out_nocheck(
            grad_input, grad_output, weight, padding, output_padding, stride, dilation, groups);
    }
    if (output_mask[IDX_1]) {
        conv_transpose3d_backward_weight_out_nocheck(
            grad_weight, input, grad_output, weight, padding, output_padding, stride, dilation, groups);
    }
    if (output_mask[IDX_2]) {
        conv_transpose3d_backward_bias_out_nocheck(grad_bias, grad_output);
    }
    return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose3d_backward(const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool, OUTPUT_MASK_SIZE> output_mask)
{
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;

    if (output_mask[IDX_0]) {
        grad_input = npu_preparation::apply_tensor_with_format(input, ACL_FORMAT_NDC1HWC0);
    }
    if (output_mask[IDX_1]) {
        grad_weight = npu_preparation::apply_tensor_with_format(
            weight.sizes(), weight.options().dtype(at::kFloat), npu_preparation::get_tensor_npu_format(weight));
    }
    if (output_mask[IDX_2]) {
        grad_bias =
            npu_preparation::apply_tensor_with_format({grad_output.size(1)}, grad_output.options(), ACL_FORMAT_NCDHW);
    }
    conv_transpose3d_backward_out_nocheck(grad_input,
        grad_weight,
        grad_bias,
        input,
        grad_output,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        output_mask);
    return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose2d_backward_symint(const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool, OUTPUT_MASK_SIZE> output_mask)
{
    at::IntArrayRef padding_ref = c10::asIntArrayRefUnchecked(padding);
    at::IntArrayRef output_padding_ref = c10::asIntArrayRefUnchecked(output_padding);
    return npu_conv_transpose2d_backward(input, grad_output, weight, padding_ref, output_padding_ref, stride, dilation, groups, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose3d_backward_symint(const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool, OUTPUT_MASK_SIZE> output_mask)
{
    at::IntArrayRef padding_ref = c10::asIntArrayRefUnchecked(padding);
    at::IntArrayRef output_padding_ref = c10::asIntArrayRefUnchecked(output_padding);
    return npu_conv_transpose3d_backward(input, grad_output, weight, padding_ref, output_padding_ref, stride, dilation, groups, output_mask);
}

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
static std::tuple<at::Tensor, at::Tensor, at::Tensor> _calc_convolution_backward(
    const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight,
    const at::OptionalIntArrayRef bias_sizes_opt, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups,
    ::std::array<bool, 3> output_mask)
{
    int64_t k = weight.ndimension();
    int64_t dim = k - 2;
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());

    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    ASCEND_LOGI("_calc_convolution_backward exec with jit compile: %d, allow internal format: %d",
                is_jit_enable, is_allow_internel_format);
    // CheckForbidInternalFormat = False: turn on private format��CheckJitDisable = False: turn on JitCompile
    if ((is_allow_internel_format || is_jit_enable) && (dim != 3)) {
        return acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                            transposed, output_padding, groups, output_mask);
    }

    c10::SmallVector<int64_t, op_infer::N> stride_expand = expand_dim_if_needed(stride, "stride", dim);
    stride = at::IntArrayRef(stride_expand);

    c10::SmallVector<int64_t, op_infer::N> padding_expand = expand_dim_if_needed(padding, "padding", dim);
    padding = at::IntArrayRef(padding_expand);

    c10::SmallVector<int64_t, op_infer::N> dilation_expand = expand_dim_if_needed(dilation, "dilation", dim);
    dilation = at::IntArrayRef(dilation_expand);

    c10::SmallVector<int64_t, op_infer::N> output_padding_expand = expand_dim_if_needed(output_padding, "output_padding",
                                                                                        dim);
    output_padding = at::IntArrayRef(output_padding_expand);

    auto outputSizes = op_infer::conv2d_backward_npu_output_size(input, grad_output, weight);

    // construct the output tensor of the NPU
    at::Tensor gradInput;
    at::Tensor gradWeight;
    at::Tensor gradBias;

    gradInput = npu_preparation::apply_tensor_without_format(std::get<0>(outputSizes), input.options());
    gradWeight = npu_preparation::apply_tensor_without_format(std::get<1>(outputSizes), weight.options());

    // use 2nd dimension of outputSizes
    gradBias = npu_preparation::apply_tensor_without_format(std::get<2>(outputSizes), grad_output.options());

    int64_t input_dim = input.ndimension();
    at::optional<c10::IntArrayRef> bias_sizes = c10::nullopt;
    if (bias_sizes_opt.has_value()) {
        bias_sizes = bias_sizes_opt.value();
    }
    EXEC_NPU_CMD(aclnnConvolutionBackward, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed,
                 output_padding, groups, output_mask, cube_math_type, gradInput, gradWeight, gradBias);
    return std::make_tuple(std::move(gradInput), std::move(gradWeight), std::move(gradBias));
}

// length of output_mask is 3
std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  DO_COMPATIBILITY(aclnnConvolutionBackward, acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt,
                                                                          stride, padding, dilation, transposed,
                                                                          output_padding, groups, output_mask));
  return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                    transposed, output_padding, groups, output_mask);
}
#endif
}
