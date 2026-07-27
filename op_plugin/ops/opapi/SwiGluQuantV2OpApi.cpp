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

// dst_type follows the ge::DataType convention: 2 -> INT8, 29 -> INT4.
constexpr int64_t DST_TYPE_INT8 = 2;
constexpr int64_t DST_TYPE_INT4 = 29;
constexpr int64_t SWISH_NUM = 2;
constexpr int64_t MAX_LAST_DIM = 8192;

std::tuple<at::Tensor, at::Tensor> npu_swi_glu_quant_v2(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& smooth_scales,
    const c10::optional<at::Tensor>& offsets,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    c10::string_view quant_mode,
    int64_t group_list_type,
    int64_t dst_type)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1", OPS_ERROR(ErrCode::PARAM));
    int64_t last_dim = x.size(x.dim() - 1);
    TORCH_CHECK(last_dim % SWISH_NUM == 0, "x last dim should be even", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(last_dim <= MAX_LAST_DIM, "x last dim should not exceed 8192", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(quant_mode == "dynamic" || quant_mode == "static",
                "quant_mode only support dynamic or static, but got ", std::string(quant_mode),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1,
                "group_list_type only support 0 or 1, but got ", group_list_type, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dst_type == DST_TYPE_INT8 || dst_type == DST_TYPE_INT4,
                "dst_type only support 2 or 29, but got ", dst_type, OPS_ERROR(ErrCode::PARAM));
    if (dst_type == DST_TYPE_INT4) {
        TORCH_CHECK(last_dim % 4 == 0,
                    "x last dim should be multiple of 4 when dst_type is int4", OPS_ERROR(ErrCode::PARAM));
    }

    // Fail-fast dtype guards BEFORE EXEC_NPU_CMD: a bad dtype that reaches the kernel latches an
    // aclnn error state (507057) on the stream and corrupts subsequent unrelated calls in the
    // same process. Surface invalid input as a clean TYPE error instead.
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x dtype only support float32/float16/bfloat16, but got ", x.scalar_type(),
                OPS_ERROR(ErrCode::TYPE));
    if (smooth_scales.has_value() && smooth_scales.value().defined()) {
        TORCH_CHECK(smooth_scales.value().scalar_type() == at::kFloat,
                    "smooth_scales dtype only support float32, but got ", smooth_scales.value().scalar_type(),
                    OPS_ERROR(ErrCode::TYPE));
    }
    if (offsets.has_value() && offsets.value().defined()) {
        TORCH_CHECK(offsets.value().scalar_type() == at::kFloat,
                    "offsets dtype only support float32, but got ", offsets.value().scalar_type(),
                    OPS_ERROR(ErrCode::TYPE));
    }
    if (group_index.has_value() && group_index.value().defined()) {
        TORCH_CHECK(group_index.value().scalar_type() == at::kInt,
                    "group_index dtype only support int32, but got ", group_index.value().scalar_type(),
                    OPS_ERROR(ErrCode::TYPE));
    }

    const at::Tensor& smooth_scales_opt = c10::value_or_else(smooth_scales, [] { return at::Tensor(); });
    const at::Tensor& offsets_opt = c10::value_or_else(offsets, [] { return at::Tensor(); });
    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    // quant_mode is passed to aclnn as a C string (null-terminated). std::string guarantees that;
    // do NOT use string_view::data() which is not required to be null-terminated.
    std::string quant_mode_str(quant_mode);
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    // Output shapes:
    //   scale: same as x with the last dim dropped.
    //   y:     same as x, but last dim = x_last/2 (INT8) or x_last/4 (INT4, 2-per-byte packed).
    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    for (int64_t i = 0; i < x.dim() - 1; i++) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    y_size.push_back(dst_type == DST_TYPE_INT4 ? last_dim / 4 : last_dim / SWISH_NUM);

    // INT4 is stored 2-per-byte in a uint8 (Byte) tensor; the logical aclnn type is INT4, so wrap it.
    // INT8 maps directly to ScalarType::Char (ACL_INT8).
    at::Tensor y;
    aclDataType y_acltype = ACL_INT8;
    if (dst_type == DST_TYPE_INT4) {
        y = npu_preparation::apply_tensor_without_format(y_size, c10::ScalarType::Byte);
        y_acltype = ACL_INT4;
    } else {
        y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(c10::ScalarType::Char));
        y_acltype = ACL_INT8;
    }
    TensorWrapper y_wrapper = {y, y_acltype};

    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(aclnnSwiGluQuantV2, x, smooth_scales_opt, offsets_opt, group_index_opt, activate_left,
                 quant_mode_ptr, group_list_type, dst_type, y_wrapper, scale);

    return std::tie(y, scale);
}
}  // namespace op_api
