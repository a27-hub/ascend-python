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

const int ROUND_MODE_MAX_LENGTH = 20;

at::Tensor npu_quant_scatter(
    const at::Tensor& self,
    const at::Tensor& indices,
    const at::Tensor& updates,
    const at::Tensor& quant_scales,
    const c10::optional<at::Tensor>& quant_zero_points,
    int64_t axis,
    int64_t quant_axis,
    c10::string_view reduce,
    c10::optional<int64_t> dst_type,
    c10::optional<c10::string_view> round_mode)
{
    at::Tensor result = self.clone();
    int64_t reduction = 1;
    bool isAclnnQuantScatterV2Available = check_aclnn_kernel_available("aclnnInplaceQuantScatterV2");
    if (isAclnnQuantScatterV2Available) {
        aclDataType self_acltype = dst_type.has_value() ? c10_npu::GetAclDataType(dst_type.value()) : aclDataType::ACL_INT8;
        TORCH_CHECK(self_acltype == aclDataType::ACL_INT8 || self_acltype == aclDataType::ACL_HIFLOAT8 || self_acltype == aclDataType::ACL_FLOAT8_E5M2 || self_acltype == aclDataType::ACL_FLOAT8_E4M3FN,
            "dst_type only support 'torch.int8', 'torch.float8_e5m2', 'torch.float8_e4m3fn', 'torch_npu.hifloat8', but got ", c10_npu::CustomDataTypeToString(dst_type.value()), OPS_ERROR(ErrCode::PARAM));
        char* round_mode_str = "rint";
        if (round_mode.has_value()) {
            round_mode_str = const_cast<char *>(round_mode.value().data());
        }
        TensorWrapper self_wrapper = {result, self_acltype};
        EXEC_NPU_CMD(aclnnInplaceQuantScatterV2, self_wrapper, indices, updates, quant_scales, quant_zero_points, axis, quant_axis,
                     reduction, round_mode_str);
    } else {
        EXEC_NPU_CMD(aclnnInplaceQuantScatter, result, indices, updates, quant_scales, quant_zero_points, axis, quant_axis,
                     reduction);
    }
    return result;
}

at::Tensor& npu_quant_scatter_(
    at::Tensor& self,
    const at::Tensor& indices,
    const at::Tensor& updates,
    const at::Tensor& quant_scales,
    const c10::optional<at::Tensor>& quant_zero_points,
    int64_t axis,
    int64_t quant_axis,
    c10::string_view reduce,
    c10::optional<int64_t> dst_type,
    c10::optional<c10::string_view> round_mode)
{
    int64_t reduction = 1;
    bool isAclnnQuantScatterV2Available = check_aclnn_kernel_available("aclnnInplaceQuantScatterV2");
    if (isAclnnQuantScatterV2Available) {
        aclDataType self_acltype = dst_type.has_value() ? c10_npu::GetAclDataType(dst_type.value()) : aclDataType::ACL_INT8;
        TORCH_CHECK(self_acltype == aclDataType::ACL_INT8 || self_acltype == aclDataType::ACL_HIFLOAT8 || self_acltype == aclDataType::ACL_FLOAT8_E5M2 || self_acltype == aclDataType::ACL_FLOAT8_E4M3FN,
            "dst_type only support 'torch.int8', 'torch.float8_e5m2', 'torch.float8_e4m3fn', 'torch_npu.hifloat8', but got ", c10_npu::CustomDataTypeToString(dst_type.value()), OPS_ERROR(ErrCode::PARAM));
        char* round_mode_str = "rint";
        if (round_mode.has_value()) {
            round_mode_str = const_cast<char *>(round_mode.value().data());
        }
        TensorWrapper self_wrapper = {self, self_acltype};
        EXEC_NPU_CMD(aclnnInplaceQuantScatterV2, self_wrapper, indices, updates, quant_scales, quant_zero_points, axis, quant_axis,
                     reduction, round_mode_str);
    } else {
        EXEC_NPU_CMD(aclnnInplaceQuantScatter, self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis,
                     reduction);
    }
    return self;
}

}
