// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_quantize_add_layer_norm(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &gamma,
    const at::Tensor &beta,
    const at::Tensor &bias,
    const at::Tensor &scales,
    const c10::optional<at::Tensor> &zero_points,
    int64_t data_type,
    int64_t axis,
    double epsilon,
    bool additional_output)
{
    at::Tensor y = npu_preparation::apply_tensor(x1, x1.options().dtype(at::kChar));
    at::Tensor x = npu_preparation::apply_tensor(x1);
    const at::Tensor& zero_points_local = c10::value_or_else(zero_points, [] {return at::Tensor();});
    string qdtype = "torch.qint8";

    at_npu::native::OpCommand cmd;
    cmd.Name("QuantizeAddLayerNorm")
        .Input(x1, "x1")
        .Input(x2, "x2")
        .Input(gamma, "gamma")
        .Input(beta, "beta")
        .Input(bias, "bias")
        .Input(scales, "scales");
    if (zero_points_local.defined()) {
        cmd.Input(zero_points_local);
    } else {
        cmd.Input();
    }
    cmd.Output(y, "y")
    .Output(x, "x")
    .Attr("dtype", data_type)
    .Attr("axis", axis)
    .Attr("epsilon", static_cast<float>(epsilon))
    .Attr("additional_output", additional_output)
    .Run();
    return std::make_tuple(y, x);
}
} // namespace acl_op
