// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> _npu_rms_norm_backward_quant(
    const at::Tensor &dy,
    const at::Tensor &x,
    const at::Tensor &rstd,
    const at::Tensor &gamma,
    const c10::optional<at::Tensor> &scale_x,
    const c10::optional<at::Tensor> &offset_x,
    bool div_mode,
    c10::string_view quant_mode,
    int64_t dst_type)
{
    TORCH_CHECK(dy.dim() >= 1, "dy must have at least 1 dimension, but got ", dy.dim(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension, but got ", x.dim(),
                OPS_ERROR(ErrCode::PARAM));

    // Convert string parameter to null-terminated char*
    std::string quant_mode_str(quant_mode);
    char *quant_mode_ptr = const_cast<char *>(quant_mode_str.c_str());

    // Convert dst_type to aclDataType
    aclDataType dx_acltype = static_cast<aclDataType>(dst_type);

    // dgamma_out: always FLOAT32, shape = gamma.shape
    at::Tensor dgamma_out = npu_preparation::apply_tensor_without_format(
        gamma.sizes(), gamma.options().dtype(at::ScalarType::Float));

    // dx_out: depends on whether quant is enabled
    at::Tensor dx_out;
    if (scale_x.has_value() && scale_x->defined()) {
        // Quant mode: output type determined by dst_type
        at::ScalarType dx_scalar = npu_preparation::convert_to_scalar_type(dx_acltype);
        dx_out = npu_preparation::apply_tensor_without_format(
            dy.sizes(), dy.options().dtype(dx_scalar));
        TensorWrapper dx_wrapper = {dx_out, dx_acltype};
        EXEC_NPU_CMD(aclnnRmsNormGradQuant, dy, x, rstd, gamma,
                      scale_x, offset_x, quant_mode_ptr, div_mode,
                      dx_acltype, dx_wrapper, dgamma_out);
    } else {
        // Non-quant mode: output same dtype as dy
        dx_out = npu_preparation::apply_tensor_without_format(
            dy.sizes(), dy.options());
        EXEC_NPU_CMD(aclnnRmsNormGradQuant, dy, x, rstd, gamma,
                      scale_x, offset_x, quant_mode_ptr, div_mode,
                      dx_acltype, dx_out, dgamma_out);
    }

    return std::make_tuple(std::move(dx_out), std::move(dgamma_out));
}

} // namespace op_api
