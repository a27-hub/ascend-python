// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

at::Tensor npu_gelu_mul(const at::Tensor & input, c10::string_view approximate)
{
    std::string approximate_str = std::string(approximate);
    TORCH_CHECK(approximate_str == "tanh" || approximate_str == "none",
        "NPU error, approximate argument must be either none or tanh.", OPS_ERROR(ErrCode::PARAM));
    auto approximate_ptr = const_cast<char*>(approximate_str.c_str());
    auto output_size = op_infer::array_to_small_vector(input.sizes());
    int64_t last_dim = input.dim() - 1;
    output_size[last_dim] /= 2; // 2 for dim split
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size,
                                                                  input.options().dtype(input.scalar_type()));
    EXEC_NPU_CMD(aclnnGeluMul, input, approximate_ptr, out);
    return out;
}
}