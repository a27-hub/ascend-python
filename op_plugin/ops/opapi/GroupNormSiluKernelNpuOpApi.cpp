// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

namespace op_api
{

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_group_norm_silu(const at::Tensor &input,
                                                                   const c10::optional<at::Tensor> &weight,
                                                                   const c10::optional<at::Tensor> &bias,
                                                                   int64_t group,
                                                                   double eps)
{
    const at::Tensor &weight_op = c10::value_or_else(weight, [] { return at::Tensor(); });
    const at::Tensor &bias_op = c10::value_or_else(bias, [] { return at::Tensor(); });

    // construct output for hostapi
    auto output = at_npu::native::OpPreparation::apply_tensor_without_format(input);

    auto mean_rstd_type = input.dtype();
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910_95)
    {
        if (weight_op.defined())
        {
            mean_rstd_type = weight_op.dtype();
        }
        else if (bias_op.defined())
        {
            mean_rstd_type = bias_op.dtype();
        }
    }

    at::SmallVector<int64_t, SIZE> mean_rstd_shape = {input.size(0), group};
    at::Tensor mean_out = at_npu::native::OpPreparation::apply_tensor_without_format(
        mean_rstd_shape, input.options().dtype(mean_rstd_type));
    at::Tensor rstd_out = at_npu::native::OpPreparation::apply_tensor_without_format(
        mean_rstd_shape, input.options().dtype(mean_rstd_type));

    // call HostAPI function
    EXEC_NPU_CMD(aclnnGroupNormSilu, input, weight, bias, group, eps, output, mean_out, rstd_out);
    return std::tie(output, mean_out, rstd_out);
}

} // namespace op_api
