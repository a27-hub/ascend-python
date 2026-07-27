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
const int64_t SWISH_NUM = 2;
const int64_t MAX_LAST_DIM = 8192;
const int64_t DST_TYPE_INT8 = 2;
const int64_t DST_TYPE_INT4 = 29;
const int64_t INT4_LAST_DIM_ALIGN = 4;
const int64_t GROUP_LIST_TYPE_CUMSUM = 0;
const int64_t GROUP_LIST_TYPE_COUNT = 1;
}  // namespace

std::tuple<at::Tensor, at::Tensor> npu_swiglu_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& smooth_scales,
    const c10::optional<at::Tensor>& offsets,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    c10::string_view quant_mode,
    int64_t group_list_type,
    int64_t dst_type)
{
    // Input validation (constraints from aclnnSwiGluQuantV2.md)
    TORCH_CHECK(x.dim() > 1, "x dim should be larger than 1, but got ", x.dim(), OPS_ERROR(ErrCode::PARAM));
    int64_t x_last_dim = x.size(x.dim() - 1);
    TORCH_CHECK(x_last_dim % SWISH_NUM == 0, "x last dim should be even, but got ", x_last_dim,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x_last_dim <= MAX_LAST_DIM, "x last dim should not exceed 8192, but got ", x_last_dim,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(quant_mode == "dynamic" || quant_mode == "static",
                "quant_mode only supports 'dynamic' or 'static', but got ", quant_mode, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(group_list_type == GROUP_LIST_TYPE_CUMSUM || group_list_type == GROUP_LIST_TYPE_COUNT,
                "group_list_type only supports 0 (cumsum) or 1 (count), but got ", group_list_type,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dst_type == DST_TYPE_INT8 || dst_type == DST_TYPE_INT4,
                "dst_type only supports 2 (int8) or 29 (int4), but got ", dst_type, OPS_ERROR(ErrCode::PARAM));
    if (dst_type == DST_TYPE_INT4) {
        TORCH_CHECK(x_last_dim % INT4_LAST_DIM_ALIGN == 0,
                    "x last dim should be a multiple of 4 when dst_type is int4 (29), but got ", x_last_dim,
                    OPS_ERROR(ErrCode::PARAM));
    }

    // Unpack optional tensors (aclnn passes undefined tensor for absent optionals)
    const at::Tensor& smooth_scales_opt = c10::value_or_else(smooth_scales, [] { return at::Tensor(); });
    const at::Tensor& offsets_opt = c10::value_or_else(offsets, [] { return at::Tensor(); });
    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    // dtype validation (constraints from aclnnSwiGluQuantV2.md). Reject up front
    // with a clear message so an unsupported dtype does not reach the kernel and
    // leave a latched aclnn error state (107000) that would corrupt the next call.
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                    x.scalar_type() == at::kBFloat16,
                "x only supports float16, bfloat16 or float32, but got ", x.scalar_type(),
                OPS_ERROR(ErrCode::PARAM));
    if (smooth_scales_opt.defined()) {
        TORCH_CHECK(smooth_scales_opt.scalar_type() == at::kFloat,
                    "smooth_scales only supports float32, but got ", smooth_scales_opt.scalar_type(),
                    OPS_ERROR(ErrCode::PARAM));
    }
    if (offsets_opt.defined()) {
        TORCH_CHECK(offsets_opt.scalar_type() == at::kFloat,
                    "offsets only supports float32, but got ", offsets_opt.scalar_type(),
                    OPS_ERROR(ErrCode::PARAM));
    }
    if (group_index_opt.defined()) {
        TORCH_CHECK(group_index_opt.scalar_type() == at::kInt,
                    "group_index only supports int32, but got ", group_index_opt.scalar_type(),
                    OPS_ERROR(ErrCode::PARAM));
    }

    // quant_mode: string_view -> null-terminated char* (safer than string_view::data())
    std::string quant_mode_str(quant_mode);
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    // Output shapes:
    //   y     : same as x except last dim = x_last / 2
    //   scale : x with the last dim removed
    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    for (int i = 0; i < x.dim() - 1; i++) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    y_size.push_back(x_last_dim / SWISH_NUM);

    // y dtype is decided by dst_type (2->int8, 29->int4). dst_type follows the
    // ge::DataType convention, which is numerically aligned with aclDataType
    // (ACL_INT8=2, ACL_INT4=29), so cast directly. Do NOT route through
    // c10_npu::GetAclDataType, which treats its argument as an aten ScalarType
    // ordinal (2 -> Short -> ACL_INT16) and would mismatch the kernel's support list.
    aclDataType y_acltype = static_cast<aclDataType>(dst_type);
    at::Tensor y;
    if (dst_type == DST_TYPE_INT4) {
        y = npu_preparation::apply_tensor_without_format(y_size, at::ScalarType::Byte);
    } else {
        at::ScalarType y_scalar_type = npu_preparation::convert_to_scalar_type(y_acltype);
        y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(y_scalar_type));
    }
    TensorWrapper y_wrapper = {y, y_acltype};

    at::Tensor scale =
        npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(aclnnSwiGluQuantV2, x, smooth_scales_opt, offsets_opt, group_index_opt, activate_left,
                 quant_mode_ptr, group_list_type, dst_type, y_wrapper, scale);

    return std::tie(y, scale);
}
}  // namespace op_api
