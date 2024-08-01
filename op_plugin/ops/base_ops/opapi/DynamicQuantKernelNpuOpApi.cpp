// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
// 兼容630版本CANN包接口
std::vector<at::Tensor> npu_dynamic_quant_630_version(const at::Tensor &input, const c10::optional<at::Tensor> &smooth_scales)
{
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    for (int i = 0; i < scale_dim; ++i) {
        scale_size.push_back(input.size(i));
    }

    at::Tensor output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(aclnnDynamicQuant, input, smooth_scales, output, scale);

    std::vector<at::Tensor> results;
    results.reserve(2);
    results.emplace_back(output);
    results.emplace_back(scale);
    return results;
}

std::vector<at::Tensor> npu_dynamic_quant(const at::Tensor &input, const c10::optional<at::Tensor> &smooth_scales, const c10::optional<at::Tensor> &group_index)
{
    DO_COMPATIBILITY(aclnnDynamicQuantV2, npu_dynamic_quant_630_version(input, smooth_scales));
    at::SmallVector<int64_t, op_infer::SIZE> scale_offset_size;
    int scale_offset_dim = input.dim() - 1;
    for (int i = 0; i < scale_offset_dim; ++i) {
        scale_offset_size.push_back(input.size(i));
    }
    
    // 预留参数，控制对称/非对称、INT4/INT8量化，当前仅支持对称、INT8
    bool is_symmetrical = true;
    int dst_type = ge::DataType::DT_INT8;

    at::Tensor output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_offset_size, c10::dtype(c10::ScalarType::Float));
    c10::optional<at::Tensor> offset;

    EXEC_NPU_CMD(aclnnDynamicQuantV2, input, smooth_scales, group_index, output, scale, offset, is_symmetrical, dst_type);
    std::vector<at::Tensor> results;
    results.reserve(2);
    results.emplace_back(output);
    results.emplace_back(scale);
    return results;
}
}