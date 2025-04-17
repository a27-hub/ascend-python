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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm_backward(
    const at::Tensor &dy,
    const at::Tensor &x,
    const at::Tensor &rstd,
    const at::Tensor &gamma,
    const c10::optional<at::Tensor> &dsum_opt)
{
    const at::Tensor &dsum = c10::value_or_else(dsum_opt, [] { return at::Tensor(); });
    auto d_sum = dsum.defined() ? dsum : at::zeros(x.sizes(), x.options());
    at::Tensor dx = npu_preparation::apply_tensor(x);
    at::Tensor dgamma = npu_preparation::apply_tensor(gamma.sizes(), gamma.options().dtype(at::kFloat), gamma);
    at_npu::native::OpCommand cmd;
    cmd.Name("RmsNormGrad")
        .Input(dy, "dy")
        .Input(x, "x")
        .Input(rstd, "rstd")
        .Input(gamma, "gamma")
        .Input(d_sum, "dsum")
        .Output(dx, "dx")
        .Output(dgamma, "dgamma")
        .Run();

    return std::make_tuple(dx, dx, dgamma);
}
} // namespace acl_op
