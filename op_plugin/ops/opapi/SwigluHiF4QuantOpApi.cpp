// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const int64_t NUM_TWO = 2;
constexpr double DEFAULT_BLOCKSIZE = 64.0;
const int64_t SCALE_TAIL_DIM = 1;

std::tuple<at::Tensor, at::Tensor> npu_swiglu_hi_f4_quant(
    const at::Tensor& x, const c10::optional<at::Tensor>& group_index,
    int64_t activate_dim, bool activate_left, int64_t swiglu_mode,
    double clamp_limit, double glu_alpha, double glu_bias, int64_t axis)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(swiglu_mode == 0 || swiglu_mode == 1, "swiglu_mode only support 0 or 1, but got ", swiglu_mode,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit > 0.0, "clamp_limit should be positive finite",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(glu_alpha), "glu_alpha should be finite", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(glu_bias), "glu_bias should be finite", OPS_ERROR(ErrCode::PARAM));

    static const bool is_available = check_aclnn_kernel_available("aclnnSwigluHiF4Quant");
    TORCH_CHECK(is_available,
                "Current CANN version do not support this api. Please try to update the version of CANN."
                + OPS_ERROR(ErrCode::PARAM));

    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    int64_t activate_dim_value = activate_dim;
    if (activate_dim_value < 0) {
        activate_dim_value = activate_dim_value + x.dim();
    }
    TORCH_CHECK(activate_dim_value <= (x.dim() - 1) && activate_dim_value >= 0,
                "activate_dim should be in range [0, x.dim()-1]", OPS_ERROR(ErrCode::PARAM));

    int64_t quant_dim_value = axis;
    if (quant_dim_value < 0) {
        quant_dim_value = quant_dim_value + x.dim();
    }
    TORCH_CHECK(quant_dim_value >= 0 && quant_dim_value <= (x.dim() - 1),
                "quant_dim should be in range [0, x.dim()-1]", OPS_ERROR(ErrCode::PARAM));

    int64_t select_dim = activate_dim_value;

    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;

    for (int i = 0; i < x.dim(); i++) {
        if (i == select_dim) {
            y_size.push_back(x.size(i) / NUM_TWO);
            scale_size.push_back(x.size(i) / NUM_TWO);
        } else {
            y_size.push_back(x.size(i));
            scale_size.push_back(x.size(i));
        }
    }

    // HIFLOAT4: y last dim halved (2 fp4 per byte)
    y_size[x.dim() - 1] = y_size[x.dim() - 1] / NUM_TWO;

    // scale quant dimension: ceil(scale_size[quant_dim] / block_size)
    int64_t quant_size = 0;
    if (!group_index_opt.defined()) {
        quant_size = static_cast<int64_t>(std::ceil(static_cast<double>(scale_size[quant_dim_value]) / DEFAULT_BLOCKSIZE));
    } else {
        if (quant_dim_value == (x.dim() - 1)) {
            quant_size = static_cast<int64_t>(std::ceil(static_cast<double>(scale_size[quant_dim_value]) / DEFAULT_BLOCKSIZE));
        } else {
            quant_size = static_cast<int64_t>(std::floor(static_cast<double>(scale_size[quant_dim_value]) / DEFAULT_BLOCKSIZE));
            quant_size = quant_size + group_index_opt.sizes()[0];
        }
    }

    scale_size[quant_dim_value] = quant_size;
    scale_size.push_back(SCALE_TAIL_DIM);

    // y: HIFLOAT4 stored as uint8 (ACL_HIFLOAT4 = 42)
    at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, c10::ScalarType::Byte);
    TensorWrapper y_wrapper = {y, aclDataType::ACL_HIFLOAT4};

    // scale: FLOAT32
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::ScalarType::Float);
    TensorWrapper scale_wrapper = {scale, aclDataType::ACL_FLOAT};

    EXEC_NPU_CMD(aclnnSwigluHiF4Quant, x, group_index_opt, activate_dim_value, activate_left,
                 swiglu_mode, clamp_limit, glu_alpha, glu_bias, axis, y_wrapper, scale_wrapper);

    return std::tie(y, scale);
}
}  // namespace op_api