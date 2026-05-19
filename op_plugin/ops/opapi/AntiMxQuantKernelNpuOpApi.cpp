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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr int64_t X_DIM_NUM_MIN = 1;
constexpr int64_t X_DIM_NUM_MAX = 7;
constexpr int64_t MXSCALE_DIM_NUM_MIN = 2;
constexpr int64_t MXSCALE_DIM_NUM_MAX = 8;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t FP4_IN_UINT8_NUM = 2;
}; // namespace

std::tuple<at::Tensor, at::Tensor> npu_anti_mx_quant(
    const at::Tensor &x,
    const at::Tensor &mxscale,
    const int64_t axis,
    int64_t dst_type)
{
    // input x and mxscale dim check
    TORCH_CHECK(x.sizes().size() >= X_DIM_NUM_MIN && x.sizes().size() <= X_DIM_NUM_MAX,
                "X dimNum should be between 1 and 7, got ", x.sizes().size(), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(mxscale.sizes().size() >= MXSCALE_DIM_NUM_MIN && mxscale.sizes().size() <= MXSCALE_DIM_NUM_MAX,
                "Mxscale dimNum should be between 2 and 8, got ", mxscale.sizes().size(), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(axis >= -1 * x.dim() && axis < x.dim(),
        "Param (axis) is out of x dimension range" + OPS_ERROR(ErrCode::PARAM));

    int64_t axis_change = axis < 0 ? axis + x.dim() : axis;
    auto mxscale_shape = op_infer::array_to_small_vector(x.sizes());
    mxscale_shape.emplace_back(ALIGN_NUM);
    int64_t dim_size = op_infer::CeilDiv(mxscale_shape[axis_change], 32);
    dim_size = (dim_size + ALIGN_NUM - 1) / ALIGN_NUM;

    for (int64_t i = 0; i < axis_change; i++) {
        TORCH_CHECK(x[i] == mxscale[i],
        "Apart from the last two dimensions, x and mxscale should be consistent, got ", mxscale[i], OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(mxscale[axis_change + 1] == 2 && mxscale[axis_change] == dim_size,
        "The shape of mxscale does not satisfy the constraint condiditons" + OPS_ERROR(ErrCode::PARAM));
    
    at::Tensor y;
    auto y_shape = op_infer::array_to_small_vector(x.sizes());

    aclDataType y_acltype;
    ASCEND_LOGI("[npu_anti_mx_quant]: Getting aclTensor y dtype by Parameter(dst_type): %ld", dst_type);

    bool special_input_type = (x.dtype() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                               x.dtype() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    if (special_input_type) {
        int64_t last_dim_val = x[x.dim() - 1];
        TORCH_CHECK(last_dim_val % FP4_IN_UINT8_NUM == 0,
                    "The last dim input shape must be divisible by 2 if "
                    "x dtype is torch_npu.float4_e2m1 or torch_npu.float4_e1m2" + OPS_ERROR(ErrCode::PARAM));
    }

    y = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);
    y_acltype = c10_npu::GetAclDataType(dst_type);
    ASCEND_LOGI("[npu_anti_mx_quant]: Setting aclTensor y dtype to: %s", at_npu::native::AclDataTypeToString(y_acltype).c_str());

    TensorWrapper y_wrapper = {y, y_acltype};
    EXEC_NPU_CMD(aclnnAntiMxQuant, x, mxscale, axis, y_acltype, y_wrapper);
    
    return y;
}

} // namespace op_api