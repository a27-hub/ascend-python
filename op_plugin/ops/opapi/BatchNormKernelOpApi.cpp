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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_out(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps,
    at::Tensor& out, at::Tensor& save_mean, at::Tensor& save_invstd) {
    DO_COMPATIBILITY(aclnnBatchNorm, acl_op::native_batch_norm_out(
        input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd));
    npu_preparation::check_tensor({input}, out, out.scalar_type(), out.sizes());
    npu_preparation::check_tensor({input}, save_mean, save_mean.scalar_type(), save_mean.sizes());
    npu_preparation::check_tensor({input}, save_invstd, save_invstd.scalar_type(), save_invstd.sizes());
    EXEC_NPU_CMD(aclnnBatchNorm, input, weight, bias, running_mean, running_var,
                 training, momentum, eps, out, save_mean, save_invstd);
    if (!training) {
        save_mean.copy_(running_mean.value());
        auto variance = running_var.value();
        if (variance.scalar_type() == at::kHalf || variance.scalar_type() == at::kBFloat16) {
            variance = variance.to(at::kFloat);
        }
        save_invstd.copy_(at::rsqrt(variance + eps));
    }
    return std::tie(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps) {
    DO_COMPATIBILITY(aclnnBatchNorm, acl_op::native_batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps));
    at::Tensor out = npu_preparation::apply_tensor_without_format(input.sizes(), input.options());
    auto stats_options = input.options().dtype(training ? at::kFloat : input.scalar_type());
    at::Tensor save_mean = npu_preparation::apply_tensor_without_format({input.size(1)}, stats_options);
    at::Tensor save_invstd = npu_preparation::apply_tensor_without_format({input.size(1)}, stats_options);
    op_api::native_batch_norm_out(input, weight, bias, running_mean, running_var,
                                  training, momentum, eps, out, save_mean, save_invstd);
    return std::make_tuple(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _native_batch_norm_legit(const at::Tensor &input,
    const c10::optional<at::Tensor> &weight, const c10::optional<at::Tensor> &bias, at::Tensor &running_mean,
    at::Tensor &running_var, bool training, double momentum, double eps) {
    return at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _native_batch_norm_legit(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, bool training, double momentum, double eps) {
    TORCH_CHECK(training, "_native_batch_norm_legit.no_stats requires training=True", OPS_ERROR(ErrCode::PARAM));
    return at::native_batch_norm(
        input, weight, bias, c10::nullopt, c10::nullopt, training, momentum, eps);
}

} // namespace op_api
