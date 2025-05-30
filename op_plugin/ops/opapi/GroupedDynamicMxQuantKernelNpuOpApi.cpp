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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr int64_t BLOCKSIZE_BASE_NUM = 32;
constexpr int64_t X_DIM_NUM = 2;
constexpr int64_t NUM_TWO = 2;
}; // namespace

std::tuple<at::Tensor, at::Tensor> npu_grouped_dynamic_mx_quant(
    const at::Tensor &x,
    const at::Tensor &group_index,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t blocksize)
{
    // input x and group_index dim check
    TORCH_CHECK(x.sizes().size() == X_DIM_NUM, 
                "X dimNum should be 2, got ", x.sizes().size(), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(group_index.sizes().size() == 1, 
                "Group_index dimNum should be 1, got ", group_index.sizes().size(), OPS_ERROR(ErrCode::VALUE));
    at::Tensor y, mxscale;
    aclDataType y_acltype = torch_npu::te::GetAclDataType(dst_type);
    TORCH_CHECK((y_acltype == aclDataType::ACL_FLOAT8_E5M2 || y_acltype == aclDataType::ACL_FLOAT8_E4M3FN),
                "Parameter dst_type only supports torch.float8_e5m2(23), torch.float8_e4m3fn(24), got ", 
                dst_type, OPS_ERROR(ErrCode::PARAM));
    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    auto mxscale_shape = op_infer::array_to_small_vector(x.sizes());

    TORCH_CHECK(blocksize == BLOCKSIZE_BASE_NUM,
                "Parameter blocksize must be 32, got ", blocksize, OPS_ERROR(ErrCode::PARAM));
    // if x shape is [m, n], group_index shape is [g] (without initial 0 and ends with m), 
    // then mxscale shape is [m / (blocksize * 2) + g, n * 2]
    mxscale_shape[0] = mxscale_shape[0] / blocksize / NUM_TWO + group_index.sizes()[0];
    mxscale_shape[1] = mxscale_shape[1] * NUM_TWO;

    char *round_mode_ptr = const_cast<char *>(round_mode.data());
    // prepare for empty output tensor
    at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    mxscale = npu_preparation::apply_tensor_without_format(mxscale_shape, c10::dtype(at::ScalarType::Byte));

    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper mxscale_wrapper = {mxscale, aclDataType::ACL_FLOAT8_E8M0};
    EXEC_NPU_CMD(aclnnGroupedDynamicMxQuant, x, group_index, round_mode_ptr, y_acltype, blocksize, y_wrapper, mxscale_wrapper);
    return std::make_tuple(y, mxscale);
}

} // namespace op_api