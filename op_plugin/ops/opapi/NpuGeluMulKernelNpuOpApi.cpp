// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/KernelNpuNewParams.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
c10::SmallVector<int64_t, op_infer::SIZE> npu_gelu_mul_out_size(const at::Tensor& input)
{
    auto output_shape = op_infer::array_to_small_vector(input.sizes());
    TORCH_CHECK(!output_shape.empty(), "input must be at least 1-dimensional." + OPS_ERROR(ErrCode::PARAM));
    output_shape.back() = output_shape.back() / 2;
    return output_shape;
}
}  // namespace

at::Tensor npu_gelu_mul(const at::Tensor& input, c10::string_view approximate)
{
    // Constraint 1: input dtype ∈ {BFLOAT16, FLOAT16, FLOAT}
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat,
        "input dtype must be BFLOAT16, FLOAT16 or FLOAT, but got ", input.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));

    // Constraint 2: input rank 2-8
    TORCH_CHECK(input.dim() >= 2 && input.dim() <= 8,
                "input dim must be in [2, 8], but got ", input.dim(),
                OPS_ERROR(ErrCode::PARAM));

    // Constraint 3: input last dim even & ≤ 1024
    int64_t last_dim = input.size(-1);
    TORCH_CHECK(last_dim % 2 == 0,
                "input last dim must be even, but got ", last_dim,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(last_dim <= 1024,
                "input last dim must be <= 1024, but got ", last_dim,
                OPS_ERROR(ErrCode::PARAM));

    // Constraint 4: product of input other dims ≤ 200000
    int64_t other_dims_product = 1;
    for (int64_t i = 0; i < input.dim() - 1; ++i) {
        other_dims_product *= input.size(i);
    }
    TORCH_CHECK(other_dims_product <= 200000,
                "product of input dims except the last one must be <= 200000, but got ", other_dims_product,
                OPS_ERROR(ErrCode::PARAM));

    // Constraint 5: approximate ∈ {"none", "tanh"} — npu_gelu_approximate_str validates this
    std::string approximate_str = op_infer::npu_gelu_approximate_str(approximate);
    char* approximate_ptr = const_cast<char*>(approximate_str.c_str());

    // Constraint 6 & 7: output dtype == input dtype, shape: last dim = input last dim / 2
    auto output_shape = npu_gelu_mul_out_size(input);
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_shape, input.options());

    EXEC_NPU_CMD(aclnnGeluMul, input, approximate_ptr, output);
    return output;
}
}  // namespace op_api
