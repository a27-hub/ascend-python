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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr int64_t MIN_INPUT_DIM = 2;
}; // namespace

std::tuple<at::Tensor, at::Tensor> npu_grouped_dynamic_block_quant(
    const at::Tensor &x,
    const at::Tensor &group_list,
    double min_scale,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t row_block_size,
    int64_t col_block_size,
    int64_t group_list_type)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "The input should be 2D or 3D" + OPS_ERROR(ErrCode::PARAM));

    // y shape = x shape
    auto y_shape = op_infer::array_to_small_vector(x.sizes());

    // scale shape 推导
    auto scale_shape = op_infer::array_to_small_vector(x.sizes());
    int64_t dim_num = x.dim();
    if (group_list_type == 0) {
        int64_t group_list_size = group_list.sizes()[0];
        if (dim_num == 2) {
            scale_shape[0] = scale_shape[0] / row_block_size + group_list_size;
            scale_shape[1] = op_infer::CeilDiv(scale_shape[1], col_block_size);
        } else if (dim_num == 3) {
            scale_shape[1] = scale_shape[1] / row_block_size + group_list_size;
            scale_shape[2] = op_infer::CeilDiv(scale_shape[2], col_block_size);
        }
    }

    // 通过 dst_type 推导 y 的数据类型
    aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
    at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    at::Tensor y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));

    // scale 输出类型为 FLOAT32
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_shape, c10::dtype(at::ScalarType::Float));

    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    EXEC_NPU_CMD(aclnnGroupedDynamicBlockQuant, x, group_list, min_scale, round_mode_ptr, y_acltype,
                 row_block_size, col_block_size, group_list_type, y, scale);

    return std::make_tuple(y, scale);
}

} // namespace op_api
