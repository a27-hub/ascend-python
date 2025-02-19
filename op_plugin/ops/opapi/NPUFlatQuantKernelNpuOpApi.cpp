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
using npu_preparation = at_npu::native::OpPreparation;

const int64_t INT4_NUMS_IN_INT32_SPACE = 8;

std::tuple<at::Tensor, at::Tensor> npu_flat_quant(
    const at::Tensor &x, const at::Tensor &kronecker_p1, const at::Tensor &kronecker_p2)
{
    auto output_shape = op_infer::array_to_small_vector(x.sizes());
    auto x_dim_num = x.dim();
    TORCH_CHECK(output_shape[x_dim_num - 1] % INT4_NUMS_IN_INT32_SPACE == 0,
        "input shape last dim must be divded by 8" + OPS_ERROR(ErrCode::PARAM));
    output_shape[x_dim_num - 1] /= INT4_NUMS_IN_INT32_SPACE;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_shape, x.options().dtype(at::ScalarType::Int));

    auto scale_output_size = op_infer::flat_quant_scale_output_size(x);
    at::Tensor quant_scale =
        npu_preparation::apply_tensor_without_format(scale_output_size, x.options().dtype(at::ScalarType::Float));

    EXEC_NPU_CMD(aclnnFlatQuant, x, kronecker_p1, kronecker_p2, out, quant_scale);
    return std::tie(out, quant_scale);
}
}  // namespace op_api
