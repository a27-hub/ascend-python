// Copyright (c) 2024-2025 Huawei Technologies Co., Ltd
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
const int64_t INT4_IN_INT32_NUM = 8;
constexpr int64_t DTYPE_NUM_FOR_QUINT4X2 = static_cast<int64_t>(at::ScalarType::QUInt4x2);

static std::initializer_list<int64_t> SUPPORT_LIST = {
    static_cast<int64_t>(at::kChar), static_cast<int64_t>(at::kFloat8_e4m3fn),
    static_cast<int64_t>(at::kFloat8_e5m2), static_cast<int64_t>(c10_npu::DType::HIFLOAT8), DTYPE_NUM_FOR_QUINT4X2};

std::tuple<at::Tensor, at::Tensor> npu_dynamic_quant_v0(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<int64_t> dst_type)
{
    bool check_dst_type = true;
    if (dst_type.has_value()) {
        check_dst_type = std::find(SUPPORT_LIST.begin(), SUPPORT_LIST.end(), dst_type.value()) != SUPPORT_LIST.end();
    }
    TORCH_CHECK(check_dst_type, "Param (dst_type) not in support list: torch.int8, "
        "torch.quint4x2, torch.float8_e5m2, torch.float8_e4m3fn, torch_npu.hifloat8" + OPS_ERROR(ErrCode::TYPE));

    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    int index = 0;
    for (; index < scale_dim; ++index) {
        scale_size.push_back(input.size(index));
    }
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    at::Tensor output;
    aclDataType y_acltype;
    if (dst_type == DTYPE_NUM_FOR_QUINT4X2) { // INT4
        TORCH_CHECK(input.size(index) % INT4_IN_INT32_NUM == 0,
                    "Input shape last dim must be divded by 8 when int4 quantization" + OPS_ERROR(ErrCode::PARAM));
        scale_size.push_back(input.size(index) / INT4_IN_INT32_NUM);
        ASCEND_LOGI("[npu_dynamic_quant]: Parameter(dst_type) is torch.quint4x2, setting aclTensor yOut dtype to: %s",
                    at_npu::native::AclDataTypeToString(aclDataType::ACL_INT32).c_str());
        output = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Int));
        y_acltype = aclDataType::ACL_INT32;
    } else if (!dst_type.has_value()) { // 默认INT8
        ASCEND_LOGI("[npu_dynamic_quant]: Parameter(dst_type) is None, setting aclTensor yOut dtype to: %s",
                    at_npu::native::AclDataTypeToString(aclDataType::ACL_INT8).c_str());
        output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
        y_acltype = aclDataType::ACL_INT8;
    } else {
        ASCEND_LOGI("[npu_dynamic_quant]: Getting aclTensor yOut dtype by Parameter(dst_type): %ld", dst_type.value());
        y_acltype = c10_npu::GetAclDataType(dst_type.value());
        ASCEND_LOGI("[npu_dynamic_quant]: Setting aclTensor yOut dtype to: %s", at_npu::native::AclDataTypeToString(y_acltype).c_str());
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(scalar_dtype));
    }
    TensorWrapper y_wrapper = {output, y_acltype};
    EXEC_NPU_CMD(aclnnDynamicQuant, input, smooth_scales, y_wrapper, scale);
    return std::make_tuple(output, scale);
}
}; // namespace

std::tuple<at::Tensor, at::Tensor> npu_dynamic_quant(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<int64_t> dst_type)
{
    // if aclnnDynamicQuantV2 is not implemented, use npu_dynamic_quant_v0(aclnnDynamicQuant)
    DO_COMPATIBILITY(aclnnDynamicQuantV2, npu_dynamic_quant_v0(input, smooth_scales, group_index, dst_type));
    bool type_is_valid = (dst_type == static_cast<int64_t>(at::ScalarType::Char) || dst_type == static_cast<int64_t>(at::ScalarType::QUInt4x2));
    TORCH_CHECK(!dst_type.has_value() || type_is_valid,
                "Param (dst_type) must be torch.int8 for int8 or torch.quint4x2 for int4" + OPS_ERROR(ErrCode::TYPE));    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    int index = 0;
    for (; index < scale_dim; ++index) {
        scale_size.push_back(input.size(index));
    }
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    int output_type;
    at::Tensor output;
    if (dst_type == static_cast<int64_t>(at::ScalarType::QUInt4x2)) { // INT4
        TORCH_CHECK(input.size(index) % INT4_IN_INT32_NUM == 0,
                    "Input shape last dim must be divded by 8 when int4 quantization" + OPS_ERROR(ErrCode::PARAM));
        output_type = ge::DataType::DT_INT32;
        scale_size.push_back(input.size(index) / INT4_IN_INT32_NUM);
        output = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Int));
    } else { // 默认INT8
        output_type = ge::DataType::DT_INT8;
        output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
    }
    c10::optional<at::Tensor> offset;

    EXEC_NPU_CMD(aclnnDynamicQuantV2, input, smooth_scales, group_index, output_type, output, scale, offset);
    return std::make_tuple(output, scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dynamic_quant_asymmetric(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<int64_t> dst_type)
{
    bool type_is_valid = (dst_type == static_cast<int64_t>(at::ScalarType::Char) || dst_type == static_cast<int64_t>(at::ScalarType::QUInt4x2));
    TORCH_CHECK(!dst_type.has_value() || type_is_valid,
                "Param (dst_type) must be torch.int8 for int8 or torch.quint4x2 for int4 when int4 quantization" + OPS_ERROR(ErrCode::TYPE));
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    int index = 0;
    for (; index < scale_dim; ++index) {
        scale_size.push_back(input.size(index));
    }

    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));
    at::Tensor offset = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));
    int output_type;
    at::Tensor output;
    if (dst_type == static_cast<int64_t>(at::ScalarType::QUInt4x2)) { // INT4
        TORCH_CHECK(input.size(index) % INT4_IN_INT32_NUM == 0,
                    "Input shape last dim must be divded by 8" + OPS_ERROR(ErrCode::PARAM));
        output_type = ge::DataType::DT_INT32;
        scale_size.push_back(input.size(index) / INT4_IN_INT32_NUM);
        output = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Int));
    } else { // 默认INT8
        output_type = ge::DataType::DT_INT8;
        output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
    }
    EXEC_NPU_CMD(aclnnDynamicQuantV2, input, smooth_scales, group_index, output_type, output, scale, offset);
    return std::make_tuple(output, scale, offset);
}
} // namespace op_api