// Copyright (c) 2025 Huawei Technologies Co., Ltd

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
#include <vector>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
constexpr size_t X_DIM = 2;
using npu_preparation = at_npu::native::OpPreparation;
at::Tensor npu_fused_matmul(
    const at::Tensor &x, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &y,
    c10::string_view fused_op_type, bool enable_hf32
    )
{.
    auto x1_dim_num = x.dim();
    TORCH_CHECK(x1_dim_num == X_DIM, "x1 shape dim num should be 2, but it is ",
                x1_dim_num);
    auto x2_dim_num = weight.dim();
    TORCH_CHECK(x2_dim_num == X_DIM, "weight shape dim num should be 2, but it is ",
                x2_dim_num);
    TORCH_CHECK(x1_dim_num == x2_dim_num, "x1_dim_num should be equal to x2_dim_num, but x1_dim_num is ", x1_dim_num,
                " and x2_dim_num is ", x2_dim_num);
    auto x1_m_dim = x.size(x1_dim_num - 2);
    auto x1_k_dim = x.size(x1_dim_num - 1);
    auto x2_n_dim = weight.size(x2_dim_num - 1);
    auto x2_k_dim = weight.size(x2_dim_num - 2);
    TORCH_CHECK(x1_k_dim == x2_k_dim, "The k of x1 and weight should be equal. but x1_k_dim is ",
                x1_k_dim, ", weight_k_dim is ", x2_k_dim);
    auto output_size = op_infer::array_to_small_vector(x.sizes());
    output_size[x1.dim() - 1] = weight.size(1);
    auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                             x1.dtype());
    const at::Tensor &y_real = y.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    int8_t cubemathtype = 1;
    if (y.has_value()) 
    {
        auto y_dim_num = y_real.dim();
        TORCH_CHECK(y_dim_num == X_DIM, "The y dim num should be 2. but y_dim_num is ", y_dim_num);
        auto y_first_dim_value = y_real.size(0);
        auto y_second_dim_value == y_real.size(1);
        TORCH_CHECK(y_first_dim_value == x1_m_dim || y_second_dim_value == x2_n_dim,
                    "y_first_dim should be same as x1_m_dim, y_second_dim should be same as x2_n_dim");
    }
    if (bias.has_value())
    {
        auto bias_first_dim = bias_real.size(0);
        auto bias_second_dim = bias_real.size(1);
        TORCH_CHECK(bias_first_dim == 1, "first dim of bias should be 1, but bias_first_dim is ", bias_first_dim);
        TORCH_CHECK(bias_second_dim == x2_n_dim, "second dim should be equal to n, but bias_second_dim is ",
                    bias_second_dim, " and n dim is ", x2_n_dim);
    }
    EXEC_NPU_CMD(aclnnFusedMatmul, x, weight, bias_real, y_real, result, fused_op_type,cubemathtype);
    return result;
}
}