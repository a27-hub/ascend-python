// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

const int DIMENSION_2D = 2;
const int DIMENSION_3D = 3;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DST_TYPE_HIFLOAT8 = 34;
constexpr float FLOAT_0 = 0.0;
const float FLOAT_15 = 15.0;
const float FLOAT_56 = 56.0;
const float FLOAT_224 = 224.0;
const float FLOAT_32768 = 32768.0;
const std::set<float> DST_TYPE_MAX_SUPPORT_DTYPE = {FLOAT_0, FLOAT_15, FLOAT_56, FLOAT_224, FLOAT_32768};

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_dynamic_block_quant(
    const at::Tensor& x,
    double min_scale,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t row_block_size,
    int64_t col_block_size,
    double dst_type_max)
{
    at::Tensor y;
    at::Tensor scale;
    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    auto scale_shape = op_infer::array_to_small_vector(x.sizes());
    if (scale_shape.size() == DIMENSION_2D) {
        scale_shape[DIM_0] = op_infer::CeilDiv(scale_shape[DIM_0], row_block_size);
        scale_shape[DIM_1] = op_infer::CeilDiv(scale_shape[DIM_1], col_block_size);
    } else if (scale_shape.size() == DIMENSION_3D) {
        scale_shape[DIM_1] = op_infer::CeilDiv(scale_shape[DIM_1], row_block_size);
        scale_shape[DIM_2] = op_infer::CeilDiv(scale_shape[DIM_2], col_block_size);
    } else {
        TORCH_CHECK(false, "x must be 2 or 3 dimensional.", OPS_ERROR(ErrCode::NOT_SUPPORT));
    }

    TORCH_CHECK(DST_TYPE_MAX_SUPPORT_DTYPE.find(dst_type_max) != DST_TYPE_MAX_SUPPORT_DTYPE.end(),
        "The dst_type_max is %f but should be 0 or 15 or 56 or 224 or 32768, please check.", dst_type_max, OPS_ERROR(ErrCode::NOT_SUPPORT));
    
    TORCH_CHECK(dst_type_max != 0 && dst_type != DST_TYPE_HIFLOAT8, "invalid dst_type_max: %f. dst_type_max only supported when dst_type is DT_HIFLOAT8", dst_type_max, OPS_ERROR(ErrCode::NOT_SUPPORT));

    ASCEND_LOGI("[npu_dynamic_block_quant]: Getting aclTensor y dtype by Parameter(dst_type): %ld", dst_type);
    aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
    at::ScalarType dtype = npu_preparation::convert_to_scalar_type(y_acltype);

    y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(dtype));
    scale = npu_preparation::apply_tensor_without_format(scale_shape, c10::dtype(c10::ScalarType::Float));

    char *round_mode_ptr = const_cast<char *>(round_mode.data());
    ASCEND_LOGI("[npu_dynamic_block_quant]: Setting aclTensor y dtype to: %s",
                at_npu::native::AclDataTypeToString(y_acltype).c_str());
    TensorWrapper y_wrapper = {y, y_acltype};
    if (dst_type_max) {
        EXEC_NPU_CMD(aclnnDynamicBlockQuantV2, x, min_scale, round_mode_ptr, y_acltype,
                    row_block_size, col_block_size, dst_type_max, y_wrapper, scale);
    } else {
        EXEC_NPU_CMD(aclnnDynamicBlockQuant, x, min_scale, round_mode_ptr, y_acltype,
                row_block_size, col_block_size, y_wrapper, scale);
    }

    return std::tie(y, scale);
}
}