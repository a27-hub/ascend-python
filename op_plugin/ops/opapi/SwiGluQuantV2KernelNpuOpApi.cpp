// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
constexpr int64_t SWISH_NUM = 2;
constexpr int64_t MAX_LAST_DIM = 8192;
constexpr int64_t INT4_MULTIPLE = 4;
constexpr int64_t DST_TYPE_INT8 = 2;
constexpr int64_t DST_TYPE_INT4 = 29;
constexpr int64_t GROUP_LIST_TYPE_CUMSUM = 0;
constexpr int64_t GROUP_LIST_TYPE_COUNT = 1;
}  // namespace

std::tuple<at::Tensor, at::Tensor> npu_swi_glu_quant_v2(const at::Tensor &x,
    const c10::optional<at::Tensor> &smooth_scales, const c10::optional<at::Tensor> &offsets,
    const c10::optional<at::Tensor> &group_index, bool activate_left, c10::string_view quant_mode,
    int64_t group_list_type, int64_t dst_type)
{
    TORCH_CHECK(x.dim() > 1, "x dim should be larger than 1", OPS_ERROR(ErrCode::PARAM));
    int64_t x_last_dim = x.size(x.dim() - 1);
    TORCH_CHECK(x_last_dim % SWISH_NUM == 0, "x last dim should be even", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x_last_dim <= MAX_LAST_DIM, "x last dim should not exceed 8192", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dst_type == DST_TYPE_INT8 || dst_type == DST_TYPE_INT4,
        "dst_type only supports 2 (int8) or 29 (int4), but got ", dst_type, OPS_ERROR(ErrCode::PARAM));
    if (dst_type == DST_TYPE_INT4) {
        TORCH_CHECK(x_last_dim % INT4_MULTIPLE == 0,
            "x last dim should be multiple of 4 when dst_type is int4", OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(quant_mode == "dynamic" || quant_mode == "static",
        "quant_mode only supports dynamic or static, but got ", quant_mode, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(group_list_type == GROUP_LIST_TYPE_CUMSUM || group_list_type == GROUP_LIST_TYPE_COUNT,
        "group_list_type only supports 0 (cumsum) or 1 (count), but got ", group_list_type,
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16 ||
        x.scalar_type() == at::ScalarType::Float,
        "x dtype only supports float16, bfloat16 or float32, but got ", x.scalar_type(),
        OPS_ERROR(ErrCode::PARAM));

    const at::Tensor &smooth_scales_opt = c10::value_or_else(smooth_scales, [] { return at::Tensor(); });
    const at::Tensor &offsets_opt = c10::value_or_else(offsets, [] { return at::Tensor(); });
    const at::Tensor &group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    if (smooth_scales.has_value() && smooth_scales->defined()) {
        TORCH_CHECK(smooth_scales->numel() > 0, "smooth_scales does not support empty tensor",
            OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(smooth_scales->scalar_type() == at::ScalarType::Float,
            "smooth_scales dtype only supports float32, but got ", smooth_scales->scalar_type(),
            OPS_ERROR(ErrCode::PARAM));
    }
    if (offsets.has_value() && offsets->defined()) {
        TORCH_CHECK(offsets->scalar_type() == at::ScalarType::Float,
            "offsets dtype only supports float32, but got ", offsets->scalar_type(),
            OPS_ERROR(ErrCode::PARAM));
    }
    if (group_index.has_value() && group_index->defined()) {
        TORCH_CHECK(group_index->scalar_type() == at::ScalarType::Int,
            "group_index dtype only supports int32, but got ", group_index->scalar_type(),
            OPS_ERROR(ErrCode::PARAM));
    }

    std::string quant_mode_str(quant_mode);
    char *quant_mode_ptr = const_cast<char *>(quant_mode_str.c_str());

    aclDataType y_acltype = static_cast<aclDataType>(dst_type);
    int64_t y_last_dim = x_last_dim / SWISH_NUM;
    if (y_acltype == aclDataType::ACL_INT4) {
        y_last_dim = y_last_dim / SWISH_NUM;
    }

    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    for (int64_t i = 0; i < x.dim() - 1; i++) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    y_size.push_back(y_last_dim);

    at::ScalarType y_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(y_dtype));
    at::Tensor scale =
        npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    TensorWrapper y_wrapper = {y, y_acltype};

    EXEC_NPU_CMD(aclnnSwiGluQuantV2, x, smooth_scales_opt, offsets_opt, group_index_opt,
                 activate_left, quant_mode_ptr, group_list_type, dst_type, y_wrapper, scale);

    return std::tie(y, scale);
}
}  // namespace op_api
