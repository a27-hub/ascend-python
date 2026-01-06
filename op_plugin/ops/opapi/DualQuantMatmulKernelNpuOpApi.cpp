// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

#include <vector>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
constexpr int DIM_ZERO = 0;
constexpr int DIM_ONE = 1;
constexpr int DIM_TWO = 2;
constexpr int SHAPE_SIZE_TWO = 2;
const int64_t FP4_NUMS_IN_FP8 = 2;

static bool is_nz_format(const at::Tensor& x2)
{
    const torch_npu::NPUStorageDesc &tensor_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ;
}

at::Tensor npu_dual_quant_matmul(const at::Tensor &x1, const at::Tensor &x2,
                                 const at::Tensor &x1_l0_scale, const at::Tensor &x1_l1_scale,
                                 const at::Tensor &x2_l0_scale, const at::Tensor &x2_l1_scale,
                                 const c10::optional<at::Tensor> &bias, c10::optional<int64_t> output_dtype,
                                 int64_t l0_group_size, int64_t l1_group_size)
{
    TORCH_CHECK(c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910_95,
                "This interface is only supported on the Ascend910_95 platform.", OPS_ERROR(ErrCode::PARAM));

    auto x1_dim_num = x1.dim();
    TORCH_CHECK(x1_dim_num == SHAPE_SIZE_TWO, "The shape dimension of x1 is only supported to be 2, but it is ",
                x1_dim_num, OPS_ERROR(ErrCode::PARAM));

    auto x2_dim_num = x2.dim();
    TORCH_CHECK(x2_dim_num == SHAPE_SIZE_TWO, "The shape dimension of x1 is only supported to be 2, but it is ",
                x2_dim_num, OPS_ERROR(ErrCode::PARAM));
    bool trans_x2 = op_plugin::utils::is_transpose_last_two_dims(x2);
    TORCH_CHECK(trans_x2 == true,
                "x2 only supports transposed inputs.", OPS_ERROR(ErrCode::PARAM));
    bool is_weight_nz = is_nz_format(x2);
    TORCH_CHECK(is_weight_nz,
                "x2 only supports NZ format.", OPS_ERROR(ErrCode::PARAM));

    auto x1_k_dim = x1.size(DIM_ONE);
    // npu_dual_quant_matmul 只接受传入 x2 为非转置的 view shape
    // trans_x2 == true 时，x2_view_shape 为 (k/2, n)
    // trans_x2 == false 时，x2_view_shape 为 (k, n/2)（前面的校验会拦截，为后续拓展方便，仍保留相关处理逻辑）
    auto x2_k_dim = trans_x2 ? weight.size(DIM_ZERO) * FP4_NUMS_IN_FP8 : weight.size(DIM_ZERO);
    TORCH_CHECK(x1_k_dim == x2_k_dim, "The k of x1 and x2 should be equal. but x1_k_dim is ", x1_k_dim,
                ", x2_k_dim is ", x2_k_dim, OPS_ERROR(ErrCode::PARAM));

    auto out_dim_num = x1_dim_num;
    auto output_size = op_infer::array_to_small_vector(x1.sizes());
    output_size[DIM_ZERO] = x1.size(DIM_ZERO);
    output_size[DIM_ONE] = trans_x2 ? x2.size(DIM_ONE) : x2.size(DIM_ONE) * FP4_NUMS_IN_FP8;

    const at::Tensor &bias_v = bias.value_or(at::Tensor());
    int l0_group_size = static_cast<int>(l0_group_size);
    int l1_group_size = static_cast<int>(l1_group_size);
    bool is_group_size_vaild = l0_group_size == 512 && l1_group_size == 32;
    TORCH_CHECK(is_group_size_vaild,
                "l0_group_size should be 512 and l1_group_size should be 32, but now l0_group_size is", l0_group_size,
                ", l1_group_size is", l1_group_size,
                OPS_ERROR(ErrCode::PARAM));

    c10::TensorOptions options;
    if (!output_dtype.has_value()) {
        options = x1.options().dtype(at::kHalf);
    } else {
        aclDataType output_acltype = c10_npu::GetAclDataType(output_dtype.value());
        options = x1.options().dtype(npu_preparation::convert_to_scalar_type(output_acltype));
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    EXEC_NPU_CMD(aclnnDualQuantMatmulWeightNz,
                 x1, x2, x1_l0_scale, x1_l0_scale, x1_l1_scale, x2_l0_scale,
                 bias_v, l0_group_size, l1_group_size, result);

    return result;
}
}