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

std::tuple<at::Tensor, at::Tensor> npu_fake_dynamic_quant(const at::Tensor &x, const c10::optional<at::Tensor> &smooth_scales)
{
    at::SmallVector<int64_t, SIZE> shape;
    for (uint64_t index = 0; index < x.dim() - 1; index++) {
        shape.emplace_back(x.size(index));
    }
    shape.emplace_back(1);

    at::Tensor y = npu_preparation::apply_tensor(x);
    at::Tensor scale = npu_preparation::apply_tensor(shape, x.options().dtype(at::kFloat), x);
    const at::Tensor& smooth_local = c10::value_or_else(smooth_scales, [] {return at::Tensor();});

    at_npu::native::OpCommand cmd;
    cmd.Name("DynamicQuant")
        .Input(x, "x");
    if (smooth_local.defined()) {
        cmd.Input(smooth_local);
    } else {
        cmd.Input();
    }
    cmd.Output(y, "y")
        .Output(scale, "scale")
        .Run();
    return std::make_tuple(y, scale);
}
} // namespace acl_op
