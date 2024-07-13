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
std::tuple<at::Tensor, at::Tensor> npu_dynamic_quant(const at::Tensor &input, const c10::optional<at::Tensor> &smooth_scales,
                                                     bool is_symmetrical, c10::optional<at::ScalarType> dst_dtype)
{
    TORCH_CHECK(input.dtype() == at::kHalf || input.dtype() == at::kBFloat16, "Not supported input dtype", input.dtype(),
                " only torch.half and torch.bfloat16 are supported now.");
    TORCH_CHECK(input.dim() >= 2, "The input should be at least 2-dim.");
    if (smooth_scales.has_value()) {
        TORCH_CHECK(input.dtype() == smooth_scales.dtype(), "The input and smooth_scale should have the same dtype.");
        TORCH_CHECK(smooth_scales.dim() == 1, "The smooth_scales should be 1-dim.");
    }
    TORCH_CHECK(!dst_dtype.has_value() ||  dst_dtype.value() != at::kChar, "Not supported dst_type", dst_dtype.value(),
                " only torch.int8 is supported now.");
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    for (int i = 0; i < scale_dim; ++i) {
        scale_size.push_back(input.size(i));
    }

    at::Tensor output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));
    at::Tensor offset = at::empty({0}, input.options());
    if (is_symmetrical) {
        offset = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));
    }
    int output_type = ge::DataType::DT_INT8;
    
    EXEC_NPU_CMD(aclnnDynamicQuant, input, smooth_scales, output, scale, offset, is_symmetrical, output_type);
    return std::make_tuple(output, scale);
}
}