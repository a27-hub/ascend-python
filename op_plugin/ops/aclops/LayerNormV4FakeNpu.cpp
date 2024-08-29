// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_layer_norm_v4_fake(const at::Tensor &x, const at::Tensor &norm_shape, const at::Tensor &gamma, const at::Tensor &beta, double epsilon)
{
    at::SmallVector<int64_t, SIZE> shape;
    for (int64_t index = 0; index < x1.dim() - gamma.dim(); index++) {
        shape.emplace_back(x1.size(index));
    }
    shape.emplace_back(1);

    at::Tensor y = npu_preparation::apply_tensor(x1);
    at::Tensor mean = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);
    at::Tensor rstd = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);

    at_npu::native::OpCommand cmd;
    cmd.Name("LayerNormV4")
        .Input(x, "x")
        .Input(norm_shape, "normalized_shape")
        .Input(gamma, "gamma")
        .Input(beta, "beta")
        .Output(y, "y")
        .Output(mean, "mean")
        .Output(rstd, "rstd")
        .Attr("epsilon", static_cast<float>(epsilon))
        .Run();
    return std::make_tuple(y, mean, rstd);
}
} // namespace acl_op