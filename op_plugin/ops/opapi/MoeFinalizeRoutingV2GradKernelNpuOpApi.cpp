/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_moe_finalize_routing_v2_grad(
    const at::Tensor &grad_y, const at::Tensor &expanded_row_idx,
    const c10::optional<at::Tensor> &expanded_x, const c10::optional<at::Tensor> &scales,
    const c10::optional<at::Tensor> &expert_idx, const c10::optional<at::Tensor> &bias,
    int64_t drop_pad_mode, int64_t active_num, int64_t expert_num, int64_t expert_capacity)
{
    const at::Tensor &p_expanded_x = c10::value_or_else(expanded_x, [] { return at::Tensor(); });
    const at::Tensor &p_scales = c10::value_or_else(scales, [] { return at::Tensor(); });
    const at::Tensor &p_expert_idx = c10::value_or_else(expert_idx, [] { return at::Tensor(); });
    const at::Tensor &p_bias = c10::value_or_else(bias, [] { return at::Tensor(); });

    TORCH_CHECK(grad_y.dim() == 2,
        "The grad_y should be 2D, but got ", grad_y.dim(), "D.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(expanded_row_idx.dim() == 1,
        "The expanded_row_idx should be 1D, but got ", expanded_row_idx.dim(), "D.",
        OPS_ERROR(ErrCode::PARAM));

    int64_t r = grad_y.size(0);
    int64_t h = grad_y.size(1);

    at::Tensor grad_expanded_x;
    if (drop_pad_mode == 1) {
        grad_expanded_x = npu_preparation::apply_tensor_without_format(
            {expert_num, expert_capacity, h}, grad_y.options());
    } else {
        int64_t dim0 = expanded_row_idx.size(0);
        if (active_num > 0 && active_num < dim0) {
            dim0 = active_num;
        }
        grad_expanded_x = npu_preparation::apply_tensor_without_format(
            {dim0, h}, grad_y.options());
    }

    int64_t grad_scales_dim1 = p_scales.defined() ? p_scales.size(1) : 1;
    auto grad_scales_options = p_scales.defined() ? p_scales.options() : grad_y.options();
    at::Tensor grad_scales = npu_preparation::apply_tensor_without_format(
        {r, grad_scales_dim1}, grad_scales_options);

    EXEC_NPU_CMD(aclnnMoeFinalizeRoutingV2Grad, grad_y, expanded_row_idx, p_expanded_x, p_scales,
                 p_expert_idx, p_bias, drop_pad_mode, active_num, expert_num, expert_capacity,
                 grad_expanded_x, grad_scales);
    return std::make_tuple(grad_expanded_x, grad_scales);
}
} // namespace op_api
