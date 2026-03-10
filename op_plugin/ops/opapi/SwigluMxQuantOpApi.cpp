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
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_swiglu_mx_quant(
    const at::Tensor& x, const c10::optional<at::Tensor>& group_index,
    c10::optional<int64_t> activate_dim, bool activate_left, int64_t swiglu_mode,
    double clamp_limit, double glu_alpha, double glu_bias,
    int64_t group_mode, int64_t axis, c10::optional<int64_t> dst_type,
    c10::optional<int64_t> round_mode, int64_t scale_alg, double max_dtype_value)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(swiglu_mode == 0 || swiglu_mode == 1, "swiglu_mode only support 0 or 1, but got ", swiglu_mode,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit > 0.0, "clamp_limit should be positive finite",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(glu_alpha), "glu_alpha should be finite", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(glu_bias), "glu_bias should be finite", OPS_ERROR(ErrCode::PARAM));

    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    int64_t dst_type_value = dst_type.has_value() ? dst_type.value() : static_cast<int64_t>(c10::ScalarType::Char);
    int64_t round_mode_value = round_mode.has_value() ? round_mode.value() : 0;
    int64_t activate_dim_value = activate_dim.has_value() ? activate_dim.value() : -1;

    TORCH_CHECK(round_mode_value >= 0 && round_mode_value <= 4,
                "round_mode only support [0, 1, 2, 3, 4], but got ", round_mode_value, OPS_ERROR(ErrCode::PARAM));

    // transform activate_dim
    if (activate_dim_value < 0) {
        activate_dim_value = activate_dim_value + x.dim();
    }
    TORCH_CHECK(activate_dim_value <= (x.dim() - 1) && activate_dim_value >= 0, "activate_dim should be in range [0, x.dim()-1]", OPS_ERROR(ErrCode::PARAM));

    // Calculate quant_dim based on axis
    int64_t quant_dim_value = axis;
    if (quant_dim_value < 0) {
        quant_dim_value = quant_dim_value + x.dim();
    }

    TORCH_CHECK(quant_dim_value >= 0 && quant_dim_value <= (x.dim() - 1), "quant_dim should be in range [0, x.dim()-1]", OPS_ERROR(ErrCode::PARAM));

    // select_dim is used to determine which dimension to divide by 2 for y and scale shapes
    int64_t select_dim = activate_dim_value;

    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;

    // Infer size of y (divide by 2 at select_dim)
    for (int i = 0; i < x.dim(); i++) {
        if (i == select_dim) {
            y_size.push_back(x.size(i) / 2);
        } else {
            y_size.push_back(x.size(i));
        }
    }

    // Infer size of scale (divide by 2 at select_dim, same as y initially)
    for (int i = 0; i < x.dim() - 1; i++) {
        if (i == select_dim) {
            scale_size.push_back(x.size(i) / 2);
        } else {
            scale_size.push_back(x.size(i));
        }
    }

    // Calculate quant_size based on group_index and quant_dim (matching meta function logic)
    int64_t quant_size = 0;
    if (!group_index_opt.defined()) {
        // group_index is None: quant_size = (ceil(scale_size[quant_dim] / 64) + 1) / 2
        quant_size = static_cast<int64_t>(std::ceil(static_cast<double>(scale_size[quant_dim_value]) / 64.0)) + 1;
        quant_size = quant_size / 2;
    } else {
        // group_index exists
        if (quant_dim_value == (x.dim() - 1)) {
            // quant_dim is last dimension: quant_size = ceil(scale_size[quant_dim] / 64) / 2
            quant_size = static_cast<int64_t>(std::ceil(static_cast<double>(scale_size[quant_dim_value]) / 64.0));
            quant_size = quant_size / 2;
        } else {
            // quant_dim is not last dimension: quant_size = ceil(scale_size[quant_dim] / 64) + group_index.shape[0]
            quant_size = static_cast<int64_t>(std::ceil(static_cast<double>(scale_size[quant_dim_value]) / 64.0));
            quant_size = quant_size + group_index_opt.sizes()[0];
        }
    }

    // Modify scale shape at quant_dim with quant_size, then append 2
    scale_size[quant_dim_value] = quant_size;
    scale_size.push_back(2);

    at::Tensor y;
    aclDataType y_acltype;

    if (dst_type_value == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
        dst_type_value == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2)) {
        int64_t last_dim_val = y_size[x.dim() - 1];
        y_size[x.dim() - 1] = last_dim_val / 2;
        TORCH_CHECK(last_dim_val % 2 == 0, "Y last dim should be even when type of y is float4_e1m2 or float4_e2m1", OPS_ERROR(ErrCode::PARAM));
    }

    if (dst_type_value == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
        dst_type_value == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2)) {
        y = npu_preparation::apply_tensor_without_format(y_size, c10::ScalarType::Byte);
        y_acltype = c10_npu::GetAclDataType(dst_type_value);
    } else {
        y_acltype = c10_npu::GetAclDataType(dst_type_value);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(scalar_dtype));
    }

    TensorWrapper y_wrapper = {y, y_acltype};

    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    static const char* round_modes[] = {"rint", "round", "floor", "ceil", "trunc"};
    const char* round_mode_ptr = round_modes[round_mode_value];

    EXEC_NPU_CMD(aclnnSwigluMxQuant, x, group_index_opt, activate_left, activate_dim_value,
                swiglu_mode, clamp_limit, glu_alpha, glu_bias, group_mode, axis,
                y_acltype, round_mode_ptr, scale_alg, max_dtype_value, y_wrapper, scale);

    return std::tie(y, scale);
}
}  // namespace op_api
