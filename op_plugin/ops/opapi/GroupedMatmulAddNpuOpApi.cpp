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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor& npu_grouped_matmul_add_(at::Tensor& self, const at::Tensor& x, const at::Tensor& weight,
                                    const at::Tensor& group_list, const c10::optional<bool>& transpose_x,
                                    const c10::optional<bool>& transpose_weight, c10::optional<int64_t> group_type,
                                    c10::optional<int64_t> group_list_type)
{
    static const bool has_grouped_matmul_add_v2 = check_aclnn_kernel_available("aclnnGroupedMatmulAddV2");
    bool transpose_x_value = transpose_x.value_or(true);
    bool transpose_weight_value = transpose_weight.value_or(false);
    int64_t group_type_value = group_type.value_or(0);
    int64_t group_list_type_value = group_list_type.value_or(0);
    bool checkSoc = c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95;
    if (!has_grouped_matmul_add_v2 || checkSoc) {
        TORCH_CHECK(
            group_list_type_value == 1U,
            "can not find aclnnGroupedMatmulAddV2, group_list_type do not support count type. Please check cann version or soc version",
            OPS_ERROR(ErrCode::VALUE));
        EXEC_NPU_CMD(aclnnGroupedMatmulAdd, x, weight, group_list, self, transpose_x_value, transpose_weight_value,
                     group_type_value);
    }

    EXEC_NPU_CMD(aclnnGroupedMatmulAddV2, x, weight, group_list, self, transpose_x_value, transpose_weight_value,
                 group_type_value, group_list_type_value);

    return self;
}

} // namespace op_api