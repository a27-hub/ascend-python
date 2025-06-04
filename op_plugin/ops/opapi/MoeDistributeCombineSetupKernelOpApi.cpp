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
    using tensor_list = std::tuple<at::Tensor, at::Tensor>;
    const int DIM_TWO = 2;

tensor_list npu_moe_distribute_combine_setup(const at::Tensor &expand_x, const at::Tensor &ep_send_counts,
                                      c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
                                      int64_t moe_expert_num, int64_t expert_shard_type,
                                      int64_t shared_expert_num, int64_t shared_expert_rank_num,
                                      int64_t global_bs, int64_t comm_quant_mode, int64_t comm_type)
{
    TORCH_CHECK(expand_x.dim() == DIM_TWO, "The expand_x should be 2D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((expand_x.scalar_type() == at::kBFloat16) || (expand_x.scalar_type() == at::kHalf) || (expand_x.scalar_type() == at::kInt),
                "dtype of expand_x should be bfloat16, float16 or int.", OPS_ERROR(ErrCode::PARAM));

    char *group_ep_ptr = const_cast<char *>(group_ep.data());

    auto expand_x_size = expand_x.sizes();
    int64_t n = expand_x_size[0];
    int64_t h = expand_x_size[1];
    int64_t global_bs_real = (global_bs == 0) ? (n * ep_world_size) : global_bs;

    at::Tensor output_expand_x = npu_preparation::apply_tensor_without_format({n, h}, expand_x.options().dtype(expand_x.scalar_type()));
    at::Tensor quant_expand_x = npu_preparation::apply_tensor_without_format({n, h}, expand_x.options().dtype(at::kHalf));

    EXEC_NPU_CMD(aclnnMoeDistributeCombineSetup, expand_x, ep_send_counts, group_ep_ptr, ep_world_size, ep_rank_id,
        moe_expert_num, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs_real, comm_quant_mode, comm_type,
        output_expand_x, quant_expand_x);
    return std::tie(output_expand_x, quant_expand_x);
}
}