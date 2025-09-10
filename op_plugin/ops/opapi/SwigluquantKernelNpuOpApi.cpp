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

namespace {
constexpr int64_t INT8 = 2;
constexpr int64_t INT4 = 29;
}  // namespace

std::tuple<at::Tensor, at::Tensor> npu_swiglu_quant(const at::Tensor& x, const c10::optional<at::Tensor>& smooth_scales,
    const c10::optional<at::Tensor>& offsets, const c10::optional<at::Tensor>& group_index, bool activate_left,
    int64_t quant_mode, int64_t group_list_type, int64_t dst_type)
{
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "quant_mode only support 0 or 1, but got", quant_mode,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1, "group_list_type only support 0 or 1, but got",
                group_list_type, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dst_type == INT8 || dst_type == INT4, "dst_type only support ", INT8, " or ", INT4, ", but got ",
                dst_type, OPS_ERROR(ErrCode::PARAM));

    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    for (int i = 0; i < x.dim() - 1; i++) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    auto last_dim = x.size(x.dim() - 1) / 2;
    y_size.push_back(last_dim);

    at::Tensor y = npu_preparation::apply_tensor_without_format(
        y_size, c10::dtype(dst_type == 2 ? c10::ScalarType::Char : c10::ScalarType::QUInt4x2)
    );
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    std::string quant_mode_str = quant_mode == 0 ? "static" : "dynamic";
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    const at::Tensor& smooth_scales_opt = c10::value_or_else(smooth_scales, [] { return at::Tensor(); });
    const at::Tensor& offsets_opt = c10::value_or_else(offsets, [] { return at::Tensor(); });
    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    EXEC_NPU_CMD(aclnnSwiGluQuantV2, x, smooth_scales_opt, offsets_opt, group_index_opt, activate_left, quant_mode_ptr,
        group_list_type, dst_type, y, scale);

    return std::tie(y, scale);
}
}  // namespace op_api
