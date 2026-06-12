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
constexpr int64_t X_DIM_NUM = 2;
constexpr int64_t GROUP_INDEX_DIM_NUM = 1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t SPLIT_BLOCK_SIZE = 64;
constexpr int64_t RINT_ROUND_MODE = 0;
constexpr int64_t DEFAULT_SCALE_ALG = 1;
constexpr double DEFAULT_DST_TYPE_MAX = 0.0;
}; // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_grouped_dynamic_mx_quant_with_dual_axis(
    const at::Tensor &x,
    const at::Tensor &group_index,
    c10::string_view round_mode,
    int64_t scale_alg,
    int64_t dst_type,
    double dst_type_max)
{
    TORCH_CHECK(x.dim() == X_DIM_NUM, "The input x should be 2D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16,
        "The input x only supports float16 or bfloat16" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(group_index.dim() == GROUP_INDEX_DIM_NUM,
        "The input group_index should be 1D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(group_index.scalar_type() == at::ScalarType::Long,
        "The input group_index only supports int64" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(group_index.numel() > 0,
        "The input group_index should not be empty" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.size(-1) % SPLIT_BLOCK_SIZE == 0 && x.size(-1) > 0,
        "The last dim of input x must be divisible by 64 and more than 0" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(round_mode == "rint",
        "The parameter round_mode only supports rint" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(scale_alg == DEFAULT_SCALE_ALG,
        "The parameter scale_alg only supports 1" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dst_type_max == DEFAULT_DST_TYPE_MAX,
        "The parameter dst_type_max only supports 0.0" + OPS_ERROR(ErrCode::PARAM));

    bool fp8_output_type = (dst_type == 23 || dst_type == 24);
    TORCH_CHECK(fp8_output_type,
        "The parameter dst_type only supports float8_e5m2 or float8_e4m3fn" + OPS_ERROR(ErrCode::PARAM));
    aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);

    auto y_shape = op_infer::array_to_small_vector(x.sizes());

    auto mxscale1_shape = op_infer::array_to_small_vector(x.sizes());
    mxscale1_shape[mxscale1_shape.size() - 1] = x.size(-1) / SPLIT_BLOCK_SIZE;
    mxscale1_shape.emplace_back(NUM_TWO);

    auto mxscale2_shape = op_infer::array_to_small_vector(x.sizes());
    mxscale2_shape[mxscale2_shape.size() - 2] = x.size(-2) / SPLIT_BLOCK_SIZE + group_index.size(0);
    mxscale2_shape.emplace_back(NUM_TWO);

    ASCEND_LOGI("[npu_grouped_dynamic_mx_quant_with_dual_axis]: "
        "Getting aclTensor y1_out and y2_out dtype by Parameter(dst_type): %ld", dst_type);

    at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    at::Tensor y1_out = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    at::Tensor y2_out = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    at::Tensor mxscale1 = npu_preparation::apply_tensor_without_format(mxscale1_shape, c10::dtype(at::ScalarType::Byte));
    at::Tensor mxscale2 = npu_preparation::apply_tensor_without_format(mxscale2_shape, c10::dtype(at::ScalarType::Byte));

    TensorWrapper y1_out_wrapper = {y1_out, y_acltype};
    TensorWrapper y2_out_wrapper = {y2_out, y_acltype};
    TensorWrapper mxscale1_wrapper = {mxscale1, aclDataType::ACL_FLOAT8_E8M0};
    TensorWrapper mxscale2_wrapper = {mxscale2, aclDataType::ACL_FLOAT8_E8M0};

    EXEC_NPU_CMD(aclnnGroupedDynamicMxQuantWithDualAxis, x, group_index, RINT_ROUND_MODE, scale_alg, dst_type,
                 dst_type_max, y1_out_wrapper, mxscale1_wrapper, y2_out_wrapper, mxscale2_wrapper);

    return std::make_tuple(y1_out, mxscale1, y2_out, mxscale2);
}

} // namespace op_api
