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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// aclnnSyncBatchNormGatherStats gathers cross-device mean/var and updates the
// running statistics. running_mean/running_var are updated in place by aclnn
// (non-const aclTensor* in the GetWorkspaceSize signature); only
// (batch_mean, batch_invstd) are returned. `input` is not consumed by aclnn —
// it carries the channel count C = input.size(1) used for output shaping.
std::tuple<at::Tensor, at::Tensor> batch_norm_gather_stats_update(
    const at::Tensor& input,
    const at::Tensor& sum_all,
    const at::Tensor& square_sum_all,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    double momentum,
    double eps,
    const at::Tensor& counts)
{
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2-dimensional.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(sum_all.dim() == 2, "sum_all must be 2-dimensional.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(sum_all.size(0) > 0, "The first dim of sum_all must be greater than 0.",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(square_sum_all.sizes() == sum_all.sizes(),
                "The shape of square_sum_all must be the same as sum_all.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(counts.dim() == 1, "counts must be 1-dimensional.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(counts.size(0) == sum_all.size(0),
                "The size of counts must be equal to the first dim of sum_all.", OPS_ERROR(ErrCode::PARAM));

    int64_t c_size = input.size(1);
    c10::SmallVector<int64_t, 1> out_shape = {c_size};
    at::TensorOptions options = sum_all.options();

    at::Tensor batch_mean = npu_preparation::apply_tensor_without_format(out_shape, options);
    at::Tensor batch_invstd = npu_preparation::apply_tensor_without_format(out_shape, options);

    // aclnn updates running_mean/running_var in place. When undefined, feed an
    // empty-shaped (C,) tensor (.md: mean/variance 支持空 Tensor) so the aclnn
    // shape constraint (shape == totalSum.size(1)) still holds; the discarded
    // result has no caller-visible effect.
    at::Tensor running_mean = (running_mean_opt.has_value() && running_mean_opt.value().defined())
        ? running_mean_opt.value()
        : at::empty(out_shape, options);
    at::Tensor running_var = (running_var_opt.has_value() && running_var_opt.value().defined())
        ? running_var_opt.value()
        : at::empty(out_shape, options);

    // aclnn declares momentum/eps as float; cast from double to avoid the
    // AArch64 low-32-bit misread that would otherwise yield NaN.
    float momentum_f = static_cast<float>(momentum);
    float eps_f = static_cast<float>(eps);

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnSyncBatchNormGatherStats, sum_all, square_sum_all, counts,
                                 running_mean, running_var, momentum_f, eps_f, batch_mean, batch_invstd);

    return std::make_tuple(batch_mean, batch_invstd);
}
}  // namespace op_api
