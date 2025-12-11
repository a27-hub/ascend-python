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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm_dynamic_quant(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const c10::optional<at::Tensor> & smooth_scale1, const c10::optional<at::Tensor> & smooth_scale2, const c10::optional<at::Tensor> & beta, double epsilon, ::std::array<bool,2> output_mask)
{
    auto output_size_0 = x1.sizes();
    auto output_size_1 = op_infer::reduce_lastdim_output_size(x1);
    auto output_dtype_0 = at::kChar;
    auto output_dtype_1 = x1.scalar_type();
    auto output_dtype_2 = at::kFloat;
    at::Tensor y1 = npu_preparation::apply_tensor_without_format(output_size_0, x1.options().dtype(output_dtype_0));
    bool reduce_y2 = !smooth_scale2.has_value() || !output_mask[1];
    at::Tensor y2 = reduce_y2 ? npu_preparation::apply_tensor_without_format(at::IntArrayRef({1}), x1.options().dtype(output_dtype_0)) : npu_preparation::apply_tensor_without_format(output_size_0, x1.options().dtype(output_dtype_0));
    at::Tensor x_out = npu_preparation::apply_tensor_without_format(output_size_0, x1.options().dtype(output_dtype_1));
    at::Tensor scale1 = npu_preparation::apply_tensor_without_format(output_size_1, x1.options().dtype(output_dtype_2));
    at::Tensor scale2 = npu_preparation::apply_tensor_without_format(output_size_1, x1.options().dtype(output_dtype_2));
    EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuantV2, x1, x2, gamma, smooth_scale1, smooth_scale2, beta, epsilon, output_mask, y1, y2, x_out, scale1, scale2);
    return std::make_tuple(std::move(y1), std::move(y2), std::move(x_out), std::move(scale1), std::move(scale2));
}
}