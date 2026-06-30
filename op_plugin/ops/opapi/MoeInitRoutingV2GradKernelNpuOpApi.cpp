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

at::Tensor npu_moe_init_routing_v2_grad(const at::Tensor &grad_expanded_x, const at::Tensor &expanded_row_idx,
                                         int64_t top_k, int64_t drop_pad_mode, int64_t active_num)
{
    TORCH_CHECK(grad_expanded_x.dim() == 2 || grad_expanded_x.dim() == 3,
        "The grad_expanded_x should be 2D or 3D, but got ", grad_expanded_x.dim(), "D.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(expanded_row_idx.dim() == 1,
        "The expanded_row_idx should be 1D, but got ", expanded_row_idx.dim(), "D.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(top_k > 0,
        "The top_k should be greater than 0, but got ", top_k, ".",
        OPS_ERROR(ErrCode::PARAM));

    int64_t n = expanded_row_idx.size(0) / top_k;
    int64_t h = grad_expanded_x.size(grad_expanded_x.dim() - 1);

    at::Tensor grad_x = npu_preparation::apply_tensor_without_format({n, h}, grad_expanded_x.options());

    EXEC_NPU_CMD(aclnnMoeInitRoutingV2Grad, grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num,
                 grad_x);
    return grad_x;
}
} // namespace op_api
