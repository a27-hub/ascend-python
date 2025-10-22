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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_clipped_swiglu(
const at::Tensor& x, const c10::optional<at::Tensor>& group_index, int64_t dim, float alpha, float limit, float bias, bool interleaved)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.size(x.dim() - 1) % 2 == 0, "x last dim should be even", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(limit) && limit > 0.0, "limit should be positive finite",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(alpha), "alpha should be finite", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(bias),  "bias should be finite",  OPS_ERROR(ErrCode::PARAM));

    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    for (int i = 0; i < x.dim(); i++) {
        y_size.push_back(x.size(i));
    }
    y_size = y_size / 2;

    at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, x.options());

    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    EXEC_NPU_CMD(aclnnClippedSwiglu, x, group_index_opt, dim, alpha, limit, bias, interleaved, y);

    return y;

}
} // namespace op_api