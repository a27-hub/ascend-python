// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_gelu_mul(const at::Tensor& input, c10::string_view approximate)
{
    // 1. Validate approximate mode
    TORCH_CHECK(approximate == "none" || approximate == "tanh",
        "approximate must be 'none' or 'tanh', but got '", approximate, "'.",
        OPS_ERROR(ErrCode::VALUE));

    // 2. Validate dtype (BFLOAT16 / FLOAT16 / FLOAT)
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kFloat,
        "input dtype must be BFLOAT16, FLOAT16 or FLOAT, but got ", dtype, ".",
        OPS_ERROR(ErrCode::TYPE));

    // 3. Validate rank (2-8) and last-dim constraints
    TORCH_CHECK(input.dim() >= 2 && input.dim() <= 8,
        "input must be 2D-8D, but got ", input.dim(), "D.",
        OPS_ERROR(ErrCode::VALUE));
    int64_t last_dim = input.size(input.dim() - 1);
    TORCH_CHECK(last_dim % 2 == 0,
        "the last dimension of input must be even, but got ", last_dim, ".",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(last_dim <= 1024,
        "the last dimension of input must be <= 1024, but got ", last_dim, ".",
        OPS_ERROR(ErrCode::VALUE));
    // product of the other dims <= 200000
    int64_t other_dims_prod = 1;
    for (int64_t i = 0; i < input.dim() - 1; i++) {
        other_dims_prod *= input.size(i);
    }
    TORCH_CHECK(other_dims_prod <= 200000,
        "the product of the input dimensions except the last one must be <= 200000, but got ",
        other_dims_prod, ".", OPS_ERROR(ErrCode::VALUE));

    // 4. Calculate output shape (last dim halved) — sync with meta
    auto output_size = op_infer::npu_gelu_mul_output_size(input);

    // 5. Create output tensor (dtype == input dtype)
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, input.options());

    // 6. Execute aclnnGeluMul
    std::string approximate_str = std::string(approximate);
    char* approximate_ptr = const_cast<char*>(approximate_str.data());
    EXEC_NPU_CMD(aclnnGeluMul, input, approximate_ptr, output);

    return output;
}
}  // namespace op_api
