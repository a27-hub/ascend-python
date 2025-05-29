// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    const int DIM_TWO = 2;

at::Tensor npu_moe_distribute_combine_teardown(const at::Tensor &expand_x, const at::Tensor &expert_ids,
                                               const at::Tensor &expand_idx, const at::Tensor &expert_scales,
                                               int8_t quant_expand_x, c10::string_view group_ep, int64_t ep_world_size,
                                               int64_t ep_rank_id, int64_t moe_expert_num,
                                               const c10::optional<at::Tensor> &x_active_mask,
                                               const c10::optional<at::Tensor> &shared_expert_x,
                                               int64_t expert_shard_type, int64_t shared_expert_num,
                                               int64_t shared_expert_rank_num, int64_t global_bs,
                                               int64_t comm_quant_mode, int64_t comm_type)
{
    TORCH_CHECK((expand_x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The expert_x and expert_ids should be 2D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(((expand_x.scalar_type() == at::kBFloat16) || (expand_x.scalar_type() == at::kHalf) || (expand_x.scalar_type() == at::kInt)) && (expert_ids.scalar_type() == at::kInt),
                "dtype of expand_x should be bfloat16, float16 or int, dtype of expert_ids should be int.", OPS_ERROR(ErrCode::PARAM));
    auto expand_x_size = expand_x.sizes();
    auto expert_ids_size = expert_ids.sizes();

    int64_t n = expert_ids_size[0];
    int64_t h = expand_x_size[1];
    int64_t global_bs_real = (global_bs == 0) ? (n * ep_world_size) : global_bs;

    char *group_ep_ptr = const_cast<char *>(group_ep.data());
    at::Tensor x;
    if (expand_x.scalar_type() != at::kInt) {
        x = npu_preparation::apply_tensor_without_format({n, h}, expand_x.options().dtype(expand_x.scalar_type()));
    } else {
        x = npu_preparation::apply_tensor_without_format({n, h}, expand_x.options().dtype(at::kHalf));
    }
    EXEC_NPU_CMD(aclnnMoeDistributeCombineTeardown, expand_x, expert_ids, expand_idx, expert_scales, quant_expand_x, x_active_mask,
                 shared_expert_x, group_ep_ptr, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type, shared_expert_num,
                 shared_expert_rank_num, global_bs_real, comm_quant_mode, comm_type, x);
    return x;
}
}
