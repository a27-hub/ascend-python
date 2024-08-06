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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_add_layer_norm_quant(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &gamma,
    const at::Tensor &beta,
    const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &scales1,
    const c10::optional<at::Tensor> &scales2,
    const c10::optional<at::Tensor> &zero_points1,
    const c10::optional<at::Tensor> &zero_points2,
    int64_t quant_mode,
    double epsilon,
    bool additional_output)
{
    at::SmallVector<int64_t, SIZE> reduce_shape;
    for (uint64_t index = 0; index < x1.dim() - gamma.dim(); index++) {
        reduce_shape.emplace_back(x1.size(index));
    }
    reduce_shape.emplace_back(1);

    at::Tensor y1 = npu_preparation::apply_tensor(x1, x1.options().dtype(at::kChar));
    at::Tensor y2 = npu_preparation::apply_tensor(x1, x1.options().dtype(at::kChar));
    at::Tensor x = npu_preparation::apply_tensor(x1);
    at::Tensor out_scale1 = npu_preparation::apply_tensor(reduce_shape, x1.options().dtype(at::kFloat), x1);
    at::Tensor out_scale2 = npu_preparation::apply_tensor(reduce_shape, x1.options().dtype(at::kFloat), x1);

    const at::Tensor& bias_local = c10::value_or_else(bias, [] {return at::Tensor();});
    const at::Tensor& scales_local1 = c10::value_or_else(scales1, [] {return at::Tensor();});
    const at::Tensor& scales_local2 = c10::value_or_else(scales2, [] {return at::Tensor();});
    const at::Tensor& zero_points_local1 = c10::value_or_else(zero_points1, [] {return at::Tensor();});
    const at::Tensor& zero_points_local2 = c10::value_or_else(zero_points2, [] {return at::Tensor();});

    at_npu::native::OpCommand cmd;
    cmd.Name("AddLayerNormQuant")
        .Input(x1, "x1")
        .Input(x2, "x2")
        .Input(gamma, "gamma")
        .Input(beta, "beta");

    if (bias_local.defined()) {
        cmd.Input(bias_local);
    } else {
        cmd.Input();
    }
    if (scales_local1.defined()) {
        cmd.Input(scales_local1);
    } else {
        cmd.Input();
    }
    if (scales_local2.defined()) {
        cmd.Input(scales_local2);
    } else {
        cmd.Input();
    }
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

    string qmode = (quant_mode == 0) ? "dynamic" : "static";

    cmd.Output(y1, "y1")
    .Output(y2, "y2")
    .Output(x, "x")
    .Output(out_scale1, "out_scale1")
    .Output(out_scale2, "out_scale2")
    .Attr("quant_mode", qmode)
    .Attr("epsilon", static_cast<float>(epsilon))
    .Attr("additional_output", additional_output)
    .Run();
    return std::make_tuple(y1, y2, x, out_scale1, out_scale2);
}
} // namespace acl_op
