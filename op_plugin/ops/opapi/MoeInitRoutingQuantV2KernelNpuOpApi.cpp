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

static const int64_t DIM_TWO = 2;
static const int64_t EXPERT_RANGE_DIM = 2;
static const int64_t MAX_EXPERT_NUM = 10240;
static const int64_t UN_QUANT = -1;
static const int64_t STATIC_QUANT = 0;
static const int64_t DYNAMIC_QUANT = 1;
static const int64_t GATHER = 0;
static const int64_t SCATTER = 1;
static const int64_t CUMSUM = 0;
static const int64_t COUNT = 1;

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    using npu_utils = at_npu::native::NpuUtils;
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

    tensor_list npu_moe_init_routing_quantv2(const at::Tensor &x, const at::Tensor &expert_idx, 
                                             const c10::optional <at::Tensor> &scale,
                                             const c10::optional <at::Tensor> &offset,
                                             int64_t active_num, int64_t expert_capacity, int64_t expert_num,
                                             int64_t drop_pad_mode, int64_t expert_tokens_count_or_cumsum_flag,
                                            bool expert_tokens_before_capacity_flag,
                                            int64_t quant_mode)
    {
        TORCH_CHECK(x.dim() == DIM_TWO, "The x should be 2D");
        TORCH_CHECK(x.scalar_type() == at::kChar || x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat ||
                    x.scalar_type() == at::kBFloat16,
                    "an int8, float16, float32 or bfloat16 tensor is expected but got a tensor with dtype: ",
                    x.scalar_type());
        TORCH_CHECK(expert_idx.dim() == DIM_TWO, "The expert_idx should be 2D");
        TORCH_CHECK(expert_idx.scalar_type() == at::kInt,
                    "int32 tensor expected but got a tensor with dtype: ",
                    expert_idx.scalar_type());

        const at::Tensor &p_scale = c10::value_or_else(scale, [] { return at::Tensor(); });
        const at::Tensor &p_offset = c10::value_or_else(offset, [] { return at::Tensor(); });
        TORCH_CHECK(quant_mode >= UN_QUANT && quant_mode <= DYNAMIC_QUANT, "quant_mode must be in [-1, 0, 1]");

        auto x_size = x.sizes();
        auto expert_idx_size = expert_idx.sizes();
        TORCH_CHECK(x_size[0] == expert_idx_size[0], "Input rows of x and expert_idx should be same.");
        int bs = x_size[0];
        int h = x_size[1];
        int k = expert_idx_size[1];
        at::Tensor expanded_x;
        // if (quant_mode == -1) {
        //     expanded_x = npu_preparation::apply_tensor_without_format(x, {bs * k, h});
        // } else {
        //     expanded_x = npu_preparation::apply_tensor_without_format({bs * k, h}, x.options().dtype(at::kChar));
        // }
        expanded_x = npu_preparation::apply_tensor_without_format({bs * k, h}, x.options().dtype(at::kChar));
        at::Tensor expanded_row_idx = npu_preparation::apply_tensor_without_format(expert_idx, {bs * k});
        at::Tensor expert_tokens_count_or_cumsum = npu_preparation::apply_tensor_without_format({expert_num},
                                                                                                x.options().dtype(
                                                                                                    at::kInt));
        at::Tensor expert_tokens_before_capacity = npu_preparation::apply_tensor_without_format({expert_num},
                                                                                                x.options().dtype(
                                                                                                    at::kInt));
        at::Tensor dynamic_quant_scale = npu_preparation::apply_tensor_without_format({bs * k},
                                                                                 x.options().dtype(at::kFloat));
        int64_t var_args=0; 
        bool is_false = false;
        EXEC_NPU_CMD(aclnnMoeInitRoutingQuantV2, x, expert_idx, p_scale, p_offset,
                     var_args, var_args, expert_num, var_args, expert_tokens_count_or_cumsum_flag,
                     is_false, quant_mode, 
                     expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity, dynamic_quant_scale);
        return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity, dynamic_quant_scale);
    }
}
