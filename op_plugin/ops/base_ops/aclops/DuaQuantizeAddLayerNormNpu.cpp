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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dua_quantize_add_layer_norm(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &gamma,
    const at::Tensor &beta,
    const at::Tensor &bias,
    const at::Tensor &scales1,
    const at::Tensor &scales2,
    const c10::optional<at::Tensor> &zero_points1,
    const c10::optional<at::Tensor> &zero_points2,
    int64_t data_type,
    int64_t axis,
    double epsilon,
    bool additional_output)
{
    at::Tensor y1 = npu_preparation::apply_tensor(x1, x1.options().dtype(at::kChar));
    at::Tensor y2 = npu_preparation::apply_tensor(x1, x1.options().dtype(at::kChar));
    at::Tensor x = npu_preparation::apply_tensor(x1);
    const at::Tensor& zero_points_local1 = c10::value_or_else(zero_points1, [] {return at::Tensor();});
    const at::Tensor& zero_points_local2 = c10::value_or_else(zero_points2, [] {return at::Tensor();});

    string qdtype = "torch.qint8";

    at_npu::native::OpCommand cmd;
    cmd.Name("DuaQuantizeAddLayerNorm")
        .Input(x1, "x1")
        .Input(x2, "x2")
        .Input(gamma, "gamma")
        .Input(beta, "beta")
        .Input(bias, "bias")
        .Input(scales1, "scales1")
        .Input(scales2, "scales2");
    if (zero_points_local1.defined()) {
        cmd.Input(zero_points_local1);
    } else {
        cmd.Input();
    }
    if (zero_points_local2.defined()) {
        cmd.Input(zero_points_local2);
    } else {
        cmd.Input();
    }

    cmd.Output(y1, "y1")
    .Output(y2, "y2")
    .Output(x, "x")
    .Attr("dtype", data_type)
    .Attr("axis", axis)
    .Attr("epsilon", static_cast<float>(epsilon))
    .Attr("additional_output", additional_output)
    .Run();
    return std::make_tuple(y1, y2, x);
}
} // namespace acl_op
