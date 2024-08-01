// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_add_layer_norm_quant(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &scales1,
    const c10::optional<at::Tensor> &scales2,
    const c10::optional<at::Tensor> &zero_points1,
    const c10::optional<at::Tensor> &zero_points2,
    int64_t quant_mode,
    double epsilon,
    bool additional_output)
{
    const at::Tensor& bias_local = c10::value_or_else(bias, [] {return at::Tensor();});
    const at::Tensor& scales1_local = c10::value_or_else(scales1, [] {return at::Tensor();});
    const at::Tensor& scales2_local = c10::value_or_else(scales2, [] {return at::Tensor();});
    const at::Tensor& zero_points1_local = c10::value_or_else(zero_points1, [] {return at::Tensor();});
    const at::Tensor& zero_points2_local = c10::value_or_else(zero_points2, [] {return at::Tensor();});
    DO_COMPATIBILITY(aclnnAddLayerNormQuant, acl_op::npu_add_layer_norm_quant(x1, x2, gamma, beta, bias_local, scales1_local, scales2_local, zero_points1_local, zero_points2_local, quant_mode, epsilon, additional_output));
    at::SmallVector<int64_t, SIZE> shape;
    for (uint64_t index = 0; index < x1.dim() - gamma.dim(); index++) {
        shape.emplace_back(x1.size(index));
    }
    shape.emplace_back(1);

    at::Tensor y1 = npu_preparation::apply_tensor(x1, x1.options().dtype(at::kChar));
    at::Tensor y2 = npu_preparation::apply_tensor(x1, x1.options().dtype(at::kChar));
    at::Tensor x = npu_preparation::apply_tensor(x1);

    at::Tensor out_scale1 = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);
    at::Tensor out_scale2 = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);

    EXEC_NPU_CMD(aclnnAddLayerNormQuant, x1, x2, gamma, beta, bias_local, scales1_local, scales2_local, zero_points1_local, zero_points2_local, quant_mode, epsilon, additional_output, y1, y2, x, out_scale1, out_scale2);
    return std::make_tuple(y1, y2, x, out_scale1, out_scale2);
}
} // namespace op_api