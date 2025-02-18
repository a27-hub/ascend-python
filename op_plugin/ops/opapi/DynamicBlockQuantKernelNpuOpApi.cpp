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

at::Tensor npu_dynamic_block_quant(
    const at::Tensor& x,
    float min_scale,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t row_block_size,
    int64_t col_block_size)
{
    at::Tensor y, scale;

    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    auto scale_shape = op_infer::array_to_small_vector(x.sizes());

    scale_shape[0] = op_infer::CeilDiv(scale_shape[0], row_block_size);
    scale_shape[1] = op_infer::CeilDiv(scale_shape[1], col_block_size);

    at::ScalarType dtype = c10::ScalarType::Float8_e5m2;
    if (dst_type == 36) {
        dtype = c10::ScalarType::Float8_e4m3fn;
    } else if (dst_type == 34) {
        dtype = c10::ScalarType::Int8;
    }
    y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(dtype));
    scale = npu_preparation::apply_tensor_without_format(scale_shape, c10::dtype(c10::ScalarType::Float));

    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    EXEC_NPU_CMD(aclnnDynamicBlockQuant, x, min_scale, round_mode_ptr,
                 dst_type, row_block_size, col_block_size, y, scale);

    return std::tie(y, scale);
}
}
