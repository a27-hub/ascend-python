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
#include <stdio.h>
#include <string.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
constexpr size_t X_DIM = 2;
constexpr size_t FUSED_TYPE_ARRAY_SIZE = 100;
using npu_preparation = at_npu::native::OpPreparation;
at::Tensor npu_fused_matmul(
    const at::Tensor &x, const at::Tensor &x2,
    const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &x3,
    c10::string_view fused_op_type
    )

{
    auto x1_dim_num = x.dim();
    TORCH_CHECK(x1_dim_num == X_DIM, "x1 shape dim num should be 2, but it is ",
                x1_dim_num);
    auto x2_dim_num = x2.dim();
    TORCH_CHECK(x2_dim_num == X_DIM, "x2 shape dim num should be 2, but it is ",
                x2_dim_num);
    TORCH_CHECK(x1_dim_num == x2_dim_num, "x1_dim_num should be equal to x2_dim_num, but x1_dim_num is ", x1_dim_num,
                " and x2_dim_num is ", x2_dim_num);

    auto x1_m_dim = x.size(x1_dim_num - 2);
    auto x1_k_dim = x.size(x1_dim_num - 1);
    auto x2_n_dim = x2.size(x2_dim_num - 1);
    auto x2_k_dim = x2.size(x2_dim_num - 2);
    TORCH_CHECK(x1_k_dim == x2_k_dim, "The k of x1 and x2 should be equal. but x1_k_dim is ",
                x1_k_dim, ", x2_k_dim is ", x2_k_dim);
    auto output_size = op_infer::array_to_small_vector(x.sizes());
    output_size[x1.dim() - 1] = x2.size(1);
    auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                             x1.dtype());
    const at::Tensor &x3_real = x3.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    if (x3.has_value()) 
    {
        auto x3_dim_num = x3_real.dim();
        TORCH_CHECK(x3_dim_num == X_DIM, "The x3 dim num should be 2. but x3_dim_num is ", x3_dim_num);
        auto x3_first_dim_value = x3_real.size(0);
        auto x3_second_dim_value == x3_real.size(1);
        TORCH_CHECK(x3_first_dim_value == x1_m_dim || x3_second_dim_value == x2_n_dim,
                    "x3_first_dim should be same as x1_m_dim, x3_second_dim should be same as x2_n_dim");
    }
    TORCH_CHECK(!bias.has_value(), "the input of bias is not supported right now");
    int8_t cube_math_type = 0;
    char fused_type[FUSED_TYPE_ARRAY_SIZE] ={0};
    TORCH_CHECK(std::string(fused_op_type).size() <= FUSED_TYPE_ARRAY_SIZE,
                "the len of fused_op_type is bigger than the default");
    strcpy(fused_type, std::string(fused_op_type).c_str());

    EXEC_NPU_CMD(aclnnFusedMatmul, x, x2, bias_real, y_real, fused_type, cube_math_type, result);
    return result;
}
}