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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
const static int MIN_DIM = 2;
const static int X_MAX_DIM = 8;
const static int EXPERT_WEIGHT_DIM = 3;
const static int NO_EXPERT_WEIGHT_DIM = 2;
const static int MAX_EXPERT_NUM = 256;
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_ffn(const at::Tensor &x, const at::Tensor &weight1, const at::Tensor &weight2,
    c10::string_view activation, at::OptionalIntArrayRef expert_tokens,
    const c10::optional<at::Tensor> &bias1, const c10::optional<at::Tensor> &bias2,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset,
    const c10::optional<at::Tensor> &deq_scale1, const c10::optional<at::Tensor> &deq_scale2,
    const c10::optional<at::Tensor> &antiquant_scale1, const c10::optional<at::Tensor> &antiquant_scale2,
    const c10::optional<at::Tensor> &antiquant_offset1, const c10::optional<at::Tensor> &antiquant_offset2,
    int64_t inner_precise, int64_t output_dtype, bool tokens_index_flag)
{
    auto x_dim_num = x.dim();
    TORCH_CHECK(x_dim_num >= MIN_DIM && x_dim_num <= X_MAX_DIM,
                "x shape dims should be 2~8, but it is ", x_dim_num,
                OPS_ERROR(ErrCode::PARAM));

    auto x_k_dim = x.size(x.dim() - 1);
    auto weight1_k_dim = weight1.size(weight1.dim() - 2);
    TORCH_CHECK(x_k_dim == weight1_k_dim,
                "The k of x and weight1 should be equal. but x_k_dim is ",
                x_k_dim, ", weight1_k_dim is ", weight1_k_dim,
                OPS_ERROR(ErrCode::PARAM));

    // Convert string_view to const char*
    std::string activation_str(activation);
    const char *activation_ptr = activation_str.c_str();

    // Handle optional tensors
    const at::Tensor &bias1_real = bias1.value_or(at::Tensor());
    const at::Tensor &bias2_real = bias2.value_or(at::Tensor());
    const at::Tensor &scale_real = scale.value_or(at::Tensor());
    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    const at::Tensor &deq_scale1_real = deq_scale1.value_or(at::Tensor());
    const at::Tensor &deq_scale2_real = deq_scale2.value_or(at::Tensor());
    const at::Tensor &antiquant_scale1_real = antiquant_scale1.value_or(at::Tensor());
    const at::Tensor &antiquant_scale2_real = antiquant_scale2.value_or(at::Tensor());
    const at::Tensor &antiquant_offset1_real = antiquant_offset1.value_or(at::Tensor());
    const at::Tensor &antiquant_offset2_real = antiquant_offset2.value_or(at::Tensor());

    // Calculate output shape
    auto output_size = op_infer::array_to_small_vector(x.sizes());
    output_size[x.dim() - 1] = weight2.size(weight2.dim() - 1);

    // Determine output dtype
    c10::TensorOptions options = x.options();
    if (deq_scale1.has_value() || deq_scale2.has_value()) {
        // Quantization scenario
        if (output_dtype == 0) {
            options = options.dtype(at::kHalf);
        } else if (output_dtype == 1) {
            options = options.dtype(at::kBFloat16);
        } else {
            // Default to FLOAT16 for quantization
            options = options.dtype(at::kHalf);
        }
    } else if (output_dtype == 0) {
        options = options.dtype(at::kHalf);
    } else if (output_dtype == 1) {
        options = options.dtype(at::kBFloat16);
    }

    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    // Handle expert_tokens - convert OptionalIntArrayRef to c10::optional<at::IntArrayRef>
    c10::optional<at::IntArrayRef> expert_tokens_for_exec;
    if (expert_tokens.has_value()) {
        expert_tokens_for_exec = expert_tokens.value();
    }

    // Execute aclnnFFN
    EXEC_NPU_CMD(aclnnFFN, x, weight1, weight2, expert_tokens_for_exec,
                 bias1_real, bias2_real, scale_real, offset_real,
                 deq_scale1_real, deq_scale2_real,
                 antiquant_scale1_real, antiquant_scale2_real,
                 antiquant_offset1_real, antiquant_offset2_real,
                 activation_ptr, inner_precise, result);

    return result;
}

}  // namespace op_api
