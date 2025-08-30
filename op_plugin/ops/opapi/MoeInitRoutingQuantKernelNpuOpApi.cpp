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

static const int64_t DIM_ONE = 1;
static const int64_t DIM_TWO = 2;

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

    tensor_list npu_moe_init_routing_quant(const at::Tensor &x, const at::Tensor &expert_idx,
                                              const c10::optional <at::Tensor> &scale,
                                              const c10::optional <at::Tensor> &offset,
                                              int64_t active_num, int64_t expert_capacity, int64_t expert_num,
                                              int64_t drop_pad_mode, int64_t expert_tokens_num_mode,
                                              bool expert_tokens_before_capacity_flag, int64_t quant_mode)
    {
        TORCH_CHECK(x.dim() == DIM_TWO, "Input tensor 'x' must be 2-dimensional, but got dimension ", x.dim());
        TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
            "Expected input tensor 'x' to have dtype float16, float32, or bfloat16, but got dtype: ", x.scalar_type());
        TORCH_CHECK(expert_idx.dim() == DIM_TWO,
            "Input tensor 'expert_idx' must be 2-dimensional, but got dimension ", expert_idx.dim());
        TORCH_CHECK(expert_idx.scalar_type() == at::kInt,
            "Expected tensor 'expert_idx' to have dtype int32, but got dtype: ", expert_idx.scalar_type());

        auto x_size = x.sizes();
        auto expert_idx_size = expert_idx.sizes();
        TORCH_CHECK(x_size[0] == expert_idx_size[0],
            "The number of rows in input 'x' (", x_size[0], ") must match the number of rows in 'expert_idx' (", expert_idx_size[0], ").");

        TORCH_CHECK(drop_pad_mode == 0 || drop_pad_mode == 1,
            "Parameter 'drop_pad_mode' must be 0 or 1, but got: ", drop_pad_mode);
        TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "Parameter 'quant_mode' must be 0 or 1, but got: ", quant_mode);
        TORCH_CHECK(expert_tokens_num_mode >= 0 && expert_tokens_num_mode <= 2,
            "Parameter 'expert_tokens_num_mode' must be 0, 1, or 2, but got: ", expert_tokens_num_mode);

        if (drop_pad_mode == 0) {
            TORCH_CHECK(active_num >= 0,
                "Parameter 'active_num' must be greater than or equal to 0 in non-Drop/Pad mode, but got: ", active_num);
        } else {
            TORCH_CHECK(expert_capacity > 0 && expert_capacity < x_size[0],
                "In Drop/Pad mode, 'expert_capacity' must be in the range (0, NUM_ROWS), but got: ", expert_capacity);
            TORCH_CHECK(expert_num > 0, "In Drop/Pad mode, 'expert_num' must be greater than 0, but got: ", expert_num);
        }

        if (expert_tokens_num_mode > 0) {
            TORCH_CHECK(expert_num > 0, "When 'expert_tokens_num_mode' > 0, 'expert_num' must be greater than 0, but got: ", expert_num);
        }

        const at::Tensor &p_scale = c10::value_or_else(scale, [] { return at::Tensor(); });
        const at::Tensor &p_offset = c10::value_or_else(offset, [] { return at::Tensor(); });

        if (quant_mode == 0) {
            TORCH_CHECK(scale.has_value() && offset.has_value(), "In static quantization mode (quant_mode=0), both 'scale' and 'offset' must be provided.");
            TORCH_CHECK(p_scale.scalar_type() == at::kFloat, "Expected tensor 'scale' to have dtype float32, but got dtype: ", p_scale.scalar_type());
            TORCH_CHECK(p_scale.dim() == DIM_ONE && p_scale.size(0) == 1, "In static quantization, 'scale' must be a 1D tensor with shape (1,), but got shape: ", p_scale.sizes());

            TORCH_CHECK(p_offset.scalar_type() == at::kFloat, "Expected tensor 'offset' to have dtype float32, but got dtype: ", p_offset.scalar_type());
            TORCH_CHECK(p_offset.dim() == DIM_ONE && p_offset.size(0) == 1, "In static quantization, 'offset' must be a 1D tensor with shape (1,), but got shape: ", p_offset.sizes());
        } else {
            if (p_scale.defined()) {
                TORCH_CHECK(p_scale.scalar_type() == at::kFloat, "Expected tensor 'scale' to have dtype float32, but got dtype: ", p_scale.scalar_type());
                TORCH_CHECK(p_scale.dim() == DIM_TWO, "In dynamic quantization, optional 'scale' must be a 2D tensor, but got dimension ", p_scale.dim());
                auto scale_size = p_scale.sizes();
                TORCH_CHECK(scale_size[0] == expert_num || scale_size[0] == 1, "The first dimension of 'scale' must be 'expert_num' or 1, but got: ", scale_size[0]);
                TORCH_CHECK(scale_size[1] == x_size[1] || scale_size[1] == 1, "The second dimension of 'scale' must match 'H' of input 'x' or be 1, but got: ", scale_size[1]);
            }
        }

        int64_t bs = x_size[0];
        int64_t h = x_size[1];
        int64_t k = expert_idx_size[1];

        at::Tensor expanded_x;
        int64_t expanded_scale_len = 0;
        if (drop_pad_mode == 1) { // Drop/Pad
            expanded_x = npu_preparation::apply_tensor_without_format({expert_num, expert_capacity, h}, x.options().dtype(at::kChar));
            expanded_scale_len = expert_num * expert_capacity;
        } else { // Dropless / Active
            if (active_num > 0) { // Active
                int64_t num_out_tokens = std::min((int64_t)bs * k, active_num);
                expanded_x = npu_preparation::apply_tensor_without_format({num_out_tokens, h}, x.options().dtype(at::kChar));
                expanded_scale_len = num_out_tokens;
            } else { // Dropless
                expanded_x = npu_preparation::apply_tensor_without_format({bs * k, h}, x.options().dtype(at::kChar));
                expanded_scale_len = bs * k;
            }
        }

        at::Tensor expanded_row_idx = npu_preparation::apply_tensor_without_format({bs * k}, expert_idx.options());

        at::Tensor expert_token_cumsum_or_count;
        if (drop_pad_mode == 0 && expert_tokens_num_mode > 0) {
            expert_token_cumsum_or_count = npu_preparation::apply_tensor_without_format({expert_num}, x.options().dtype(at::kInt));
        } else {
            expert_token_cumsum_or_count = at::Tensor();
        }

        at::Tensor expert_tokens_before_capacity;
        if (drop_pad_mode == 1 && expert_tokens_before_capacity_flag) {
            expert_tokens_before_capacity = npu_preparation::apply_tensor_without_format({expert_num}, x.options().dtype(at::kInt));
        } else {
            expert_tokens_before_capacity = at::Tensor();
        }

        at::Tensor expanded_scale;
        if (quant_mode == 1) {
            expanded_scale = npu_preparation::apply_tensor_without_format({expanded_scale_len}, x.options().dtype(at::kFloat));
        } else {
            expanded_scale = at::Tensor();
        }

        EXEC_NPU_CMD(aclnnMoeInitRoutingQuantV2, x, expert_idx, p_scale, p_offset,
                     active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_mode,
                     expert_tokens_before_capacity_flag, quant_mode, 
                     expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expert_tokens_before_capacity, expanded_scale);

        return std::tie(expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expert_tokens_before_capacity, expanded_scale);
    }
}
