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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_swi_glu_dynamic_quant(
    const at::Tensor &x,
    const c10::optional<at::Tensor> &smooth,
    const c10::optional<at::Tensor> &group_idx,
    int64_t dim, int64_t core_num)
{
    const at::Tensor& smooth_local = c10::value_or_else(smooth, [] {return at::Tensor();});
    const at::Tensor& group_idx_local = c10::value_or_else(group_idx, [] {return at::Tensor();});

    DO_COMPATIBILITY(aclnnSwiGluDynamicQuant, acl_op::npu_swi_glu_dynamic_quant(x, smooth_local, group_idx_local, dim, core_num));

    at::SmallVector<int64_t, SIZE> glu_shape;
    for (uint64_t index = 0; index < x.dim(); index++) {
        glu_shape.emplace_back(x.size(index));
    }
    int64_t slice_dim = dim < 0 ? x.dim() + dim : dim;
    glu_shape[slice_dim] /= 2;

    at::SmallVector<int64_t, SIZE> reduce_shape;
    for (uint64_t index = 0; index < x.dim(); index++) {
        reduce_shape.emplace_back(x.size(index));
    }
    reduce_shape[slice_dim] /= 2;
    reduce_shape[x.dim() - 1] = 1;

    at::Tensor y = npu_preparation::apply_tensor(glu_shape, x.options().dtype(at::kChar), x);
    at::Tensor scale = npu_preparation::apply_tensor(reduce_shape, x.options().dtype(at::kFloat), x);
    at::Tensor test = npu_preparation::apply_tensor(glu_shape, x.options().dtype(at::kFloat), x);

    EXEC_NPU_CMD(aclnnSwiGluDynamicQuant, x, smooth_local, group_idx_local, dim, core_num, y, scale, test);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale, test);
}
} // namespace op_api