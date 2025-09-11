// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm_dynamic_quant(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &gamma,
    const c10::optional<at::Tensor> &smooth1,
    const c10::optional<at::Tensor> &smooth2,
    double epsilon)
{
    const at::Tensor& smooth_local1 = c10::value_or_else(smooth1, [] {return at::Tensor();});
    const at::Tensor& smooth_local2 = c10::value_or_else(smooth2, [] {return at::Tensor();});

    DO_COMPATIBILITY(aclnnAddRmsNormDynamicQuant, acl_op::npu_add_rms_norm_dynamic_quant(x1, x2, gamma, smooth_local1, smooth_local2, epsilon));
    
    at::SmallVector<int64_t, SIZE> reduce_shape;
    for (uint64_t index = 0; index < x1.dim() - gamma.dim(); index++) {
        reduce_shape.emplace_back(x1.size(index));
    }
    reduce_shape.emplace_back(1);

    at::Tensor y1 = npu_preparation::apply_tensor(x1, x1.options().dtype(at::kChar));
    at::Tensor y2 = npu_preparation::apply_tensor(x1, x1.options().dtype(at::kChar));
    at::Tensor x = npu_preparation::apply_tensor(x1);
    at::Tensor out_scale1 = npu_preparation::apply_tensor(reduce_shape, x1.options().dtype(at::kFloat), x1);
    at::Tensor out_scale2 = npu_preparation::apply_tensor(reduce_shape, x1.options().dtype(at::kFloat), x1);

    EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuant, x1, x2, gamma, smooth_local1, smooth_local2, epsilon, y1, y2, x, out_scale1, out_scale2);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(y1, y2, x, out_scale1, out_scale2);
}
} // namespace op_api