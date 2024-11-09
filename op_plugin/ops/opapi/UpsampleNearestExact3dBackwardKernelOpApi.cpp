// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/UpsampleConstants.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

bool checkNearestExact3dGradScales(float realScale_d, float realScale_h, float realScale_w)
{
    return !(realScale_d < NEAREST_GRAD_MAX_SCALE || realScale_h < NEAREST_GRAD_MAX_SCALE || realScale_w < NEAREST_GRAD_MAX_SCALE);
}

bool checkNearestExact3dGradUseFast(const at::Tensor &grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, double scales_d, double scales_h,
    double scales_w, at::Tensor &grad_input)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                              c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    double realScale_d = op_plugin::utils::compute_scale(false, grad_input.size(DIM_D), grad_output.size(DIM_D), scales_d);
    double realScale_h = op_plugin::utils::compute_scale(false, grad_input.size(DIM_H), grad_output.size(DIM_H), scales_h);
    double realScale_w = op_plugin::utils::compute_scale(false, grad_input.size(DIM_W), grad_output.size(DIM_W), scales_w);
    if (!is_support_nd_out || !checkNearestExact3dGradScales(realScale_d, realScale_h, realScale_w)) {
        return false;
    }
    return true;
}

at::Tensor &upsample_nearest_exact3d_grad_out_slow(const at::Tensor &grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size,
    c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor &grad_input)
{
    auto scalar_type = grad_output.scalar_type();
    at::Tensor gradOutput_slow = grad_output.cpu().to(at::ScalarType::Float);
    at::Tensor result_slow = at::_upsample_nearest_exact3d_backward(gradOutput_slow, output_size, input_size, scales_d, scales_h, scales_w);
    grad_input.copy_(result_slow.to(scalar_type));
    return grad_input;
}

at::Tensor upsample_nearest_exact3d_grad_slow(const at::Tensor &grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size,
    c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    auto scalar_type = grad_output.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output, input_size);
    at::Tensor gradOutput_slow = grad_output.cpu().to(at::ScalarType::Float);

    at::Tensor result_slow = at::_upsample_nearest_exact3d_backward(gradOutput_slow, output_size, input_size, scales_d, scales_h, scales_w);
    grad_input.copy_(result_slow.to(scalar_type));
    return grad_input;
}

at::Tensor &_upsample_nearest_exact3d_backward_out(const at::Tensor &grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size,
    c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor &grad_input)
{
    DO_COMPATIBILITY(aclnnUpsampleNearestExact3dGrad,
        upsample_nearest_exact3d_grad_out_slow(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input));

    npu_preparation::check_tensor({grad_output}, grad_input, grad_output, input_size);
    double scales_d_attr = scales_d.value_or(0);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    if (!checkNearestExact3dGradUseFast(grad_output, output_size, input_size, scales_d_attr, scales_h_attr, scales_w_attr, grad_input)) {
        return upsample_nearest_exact3d_grad_out_slow(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
    }
    EXEC_NPU_CMD(aclnnUpsampleNearestExact3dGrad, grad_output, output_size, input_size, scales_d, scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}

at::Tensor _upsample_nearest_exact3d_backward(const at::Tensor &grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size,
    c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(
        aclnnUpsampleNearestExact3dGrad, upsample_nearest_exact3d_grad_slow(grad_output, output_size, input_size, scales_d, scales_h, scales_w));

    double scales_d_attr = scales_d.value_or(0);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output, input_size);
    if (!checkNearestExact3dGradUseFast(grad_output, output_size, input_size, scales_d_attr, scales_h_attr, scales_w_attr, grad_input)) {
        return upsample_nearest_exact3d_grad_slow(grad_output, output_size, input_size, scales_d, scales_h, scales_w);
    }
    EXEC_NPU_CMD(aclnnUpsampleNearestExact3dGrad, grad_output, output_size, input_size, scales_d_attr, scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}

}  // namespace op_api
