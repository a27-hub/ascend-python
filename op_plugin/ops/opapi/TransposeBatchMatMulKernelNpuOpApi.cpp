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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

bool is_nd_nz_format_3d(const at::Tensor &self, const at::Tensor &mat2)
{
    auto dim_tensor1 = self.dim();
    auto dim_tensor2 = mat2.dim();
    // only support 3D ND * 3D NZ
    return dim_tensor1 == 3 && dim_tensor2 == 3 && op_plugin::utils::is_nz_format(mat2) && !op_plugin::utils::is_nz_format(self);
}

c10::SmallVector<int64_t, SIZE> cal_output_size(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &scale_real,
                                                at::IntArrayRef perm_x1_real, at::IntArrayRef perm_x2_real, at::IntArrayRef perm_y_real,
                                                int32_t batch_split_factor_value)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    // auto input_dim_num = input.dim();
    // auto weight_dim_num = weight.dim();
    // constexpr int EXPECTED_DIM = 3;

    // TORCH_CHECK(input_dim_num == EXPECTED_DIM && weight_dim_num == EXPECTED_DIM,
    //             "input dim is ", input_dim_num, "but expected is ", EXPECTED_DIM,
    //             "weight dim is ", weight_dim_num, "but expected is ", EXPECTED_DIM, OPS_ERROR(ErrCode::PARAM));
    // TORCH_CHECK(((input.scalar_type() == at::ScalarType::Half) || (input.scalar_type() == at::ScalarType::BFloat16) ||
    //              (input.scalar_type() == at::ScalarType::Float)),
    //             "input's type supported for float16, float32 and bfloat16." + OPS_ERROR(ErrCode::PARAM));
    // TORCH_CHECK(((weight.scalar_type() == at::ScalarType::Half) || (weight.scalar_type() == at::ScalarType::BFloat16) ||
    //              (weight.scalar_type() == at::ScalarType::Float)),
    //             "weight's type supported for float16, float32 and bfloat16." + OPS_ERROR(ErrCode::PARAM));

    // auto check_perm_x1 = (perm_x1_real[0] == 0 && perm_x1_real[1] == 1 && perm_x1_real[2] == 2) ||
    //                      (perm_x1_real[0] == 1 && perm_x1_real[1] == 0 && perm_x1_real[2] == 2);
    // TORCH_CHECK(check_perm_x1, "perm_x1 should be [0, 1, 2] or [1, 0, 2]" + OPS_ERROR(ErrCode::PARAM));
    // if (op_plugin::utils::is_gte_cann_version_850()) {
    //     auto check_perm_x2 = (perm_x2_real[0] == 0 && perm_x2_real[1] == 1 && perm_x2_real[2] == 2) ||
    //                          (perm_x2_real[0] == 0 && perm_x2_real[1] == 2 && perm_x2_real[2] == 1);
    //     TORCH_CHECK(check_perm_x2, "perm_x2 should be [0, 1, 2] or [0, 2, 1]" + OPS_ERROR(ErrCode::PARAM));
    // } else {
    //     auto check_perm_x2 = (perm_x2_real[0] == 0 && perm_x2_real[1] == 1 && perm_x2_real[2] == 2);
    //     TORCH_CHECK(check_perm_x2, "perm_x2 should be [0, 1, 2]" + OPS_ERROR(ErrCode::PARAM));
    // }
    auto check_perm_y = perm_y_real[0] == 1 && perm_y_real[1] == 0 && perm_y_real[2] == 2;
    TORCH_CHECK(check_perm_y, "perm_y should be [1, 0, 2]" + OPS_ERROR(ErrCode::PARAM));
    auto m_dim = input.size(perm_x1_real[1]);
    auto batch_dim = input.size(perm_x1_real[0]);
    auto n_dim = weight.size(perm_x2_real[2]);

    output_size = {m_dim, batch_dim, n_dim};
    if (scale_real.defined()) {
        output_size = {m_dim, 1, batch_dim * n_dim};
    }
    if (batch_split_factor_value > 1) {
        output_size = {batch_split_factor_value, m_dim, batch_dim * n_dim / batch_split_factor_value};
    }
    return output_size;
}

at::Tensor npu_transpose_batchmatmul(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &scale_real, 
                                     at::IntArrayRef perm_x1_real, at::IntArrayRef perm_x2_real, at::IntArrayRef perm_y_real,
                                     int32_t batch_split_factor_value)
{
    const at::Tensor bias = None;
    // Use the correct function call with all required parameters
    auto output_size = cal_output_size(input, weight, scale_real, perm_x1_real, perm_x2_real, perm_y_real, batch_split_factor_value);
    
    // Construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, input.options());
    int cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    // Check format and execute the appropriate NPU command
    if (is_nd_nz_format_3d(input, weight)) {
        EXEC_NPU_CMD(aclnnTransposeBatchMatMulWeightNz, input, weight, bias, scale_real, perm_x1_real, perm_x2_real, perm_y_real, cube_math_type, batch_split_factor_value, result);
    } else {
        EXEC_NPU_CMD(aclnnTransposeBatchMatMul, input, weight, bias, scale_real, perm_x1_real, perm_x2_real, perm_y_real, cube_math_type, batch_split_factor_value, result);
    }

    return result;
}
} // namespace op_api
