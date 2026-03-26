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
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/KernelNpuOutputSize.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using namespace op_infer;

namespace {
constexpr int64_t BLOCK_SIZE_BASE_NUM = 32;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t FP4_IN_UINT8_NUM = 2;
constexpr int64_t DEFAULT_SCALE_ALG = 0;
constexpr double DEFAULT_EPSILON = 1e-06;
constexpr int64_t DEFAULT_DST_TYPE = 296;  // float4_e2m1fn_x2
}; // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm_dynamic_mx_quant(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &gamma,
    const c10::optional<at::Tensor> &beta,
    double epsilon,
    int64_t scale_alg,
    c10::string_view round_mode,
    int64_t dst_type)
{
    // 1. 输入校验
    TORCH_CHECK(x1.dim() >= 1 && x1.dim() <= 7,
        "The input x1 should be 1-7 dimensions" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1.sizes() == x2.sizes(),
        "The shape of x1 and x2 must be the same" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gamma.dim() == 1,
        "The gamma should be 1D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gamma.size(0) == x1.size(-1),
        "The size of gamma must match the last dimension of x1" + OPS_ERROR(ErrCode::PARAM));

    // 2. 判断是否需要输出 rstd（根据 requires_grad）
    bool output_rstd = x1.requires_grad() && x2.requires_grad();

    // 3. 推导输出 shape
    auto x_shape = op_infer::array_to_small_vector(x1.sizes());
    int64_t dim_num = x1.dim();

    // y shape: 与 x1 相同，但 FP4 时尾轴减半
    auto y_shape = x_shape;

    // x_out shape: 与 x1 相同
    auto x_out_shape = x_shape;

    // mxscale_out shape: 尾轴特殊计算，增加一维
    auto mxscale_shape = x_shape;
    // mxscale 增加一维，shape[-2] = ceil(x1.shape[-1] / 32)，然后偶数 pad
    int64_t last_dim = x_shape[dim_num - 1];
    int64_t mxscale_last_dim = (op_infer::CeilDiv(last_dim, BLOCK_SIZE_BASE_NUM) + ALIGN_NUM - 1) / ALIGN_NUM;
    mxscale_shape[dim_num - 1] = mxscale_last_dim;
    mxscale_shape.emplace_back(ALIGN_NUM);

    // rstd_out shape: 与 x1 的前几维相同，最后一维为 1
    auto rstd_shape = x_shape;
    rstd_shape[dim_num - 1] = 1;

    // 4. 准备输出 tensor
    aclDataType y_acltype;
    at::Tensor y;
    at::Tensor x_out;
    at::Tensor mxscale_out;
    at::Tensor rstd_out;

    // y 输出类型处理
    bool is_fp4_output = (dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                          dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    if (is_fp4_output) {
        TORCH_CHECK(x_shape[dim_num - 1] % FP4_IN_UINT8_NUM == 0,
            "The last dim of input must be divisible by 2 when output dtype is float4_e2m1fn_x2 or float4_e1m2fn_x2"
            + OPS_ERROR(ErrCode::PARAM));
        y_shape[dim_num - 1] = y_shape[dim_num - 1] / FP4_IN_UINT8_NUM;
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);
        y_acltype = c10_npu::GetAclDataType(dst_type);
    } else {
        y_acltype = c10_npu::GetAclDataType(dst_type);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    }

    // x_out: 与 x1 相同 dtype
    x_out = npu_preparation::apply_tensor_without_format(x_out_shape, x1.options());

    // mxscale_out: uint8 存储，但逻辑上是 float8_e8m0
    mxscale_out = npu_preparation::apply_tensor_without_format(mxscale_shape, c10::dtype(at::ScalarType::Byte));

    // rstd_out: 根据 output_rstd 决定是否分配，但 always 创建 tensor（shape 可能为 0）
    if (output_rstd) {
        rstd_out = npu_preparation::apply_tensor_without_format(rstd_shape, c10::dtype(at::ScalarType::Float));
    } else {
        // 创建空 tensor，shape 为 0
        rstd_out = npu_preparation::apply_tensor_without_format({0}, c10::dtype(at::ScalarType::Float));
    }

    // 5. 参数转换
    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    // 6. 使用 TensorWrapper 绑定实际 dtype
    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper mxscale_wrapper = {mxscale_out, aclDataType::ACL_FLOAT8_E8M0};

    // 7. 调用 aclnn 接口
    // aclnnAddRmsNormDynamicMxQuant(x1, x2, gamma, beta, epsilon, scaleAlg, roundMode, dstType, outputRstd, yOut, xOut, mxscaleOut, rstdOut)
    if (beta.has_value() && beta.value().defined()) {
        EXEC_NPU_CMD(aclnnAddRmsNormDynamicMxQuant, x1, x2, gamma, beta.value(), epsilon, scale_alg,
                     round_mode_ptr, y_acltype, output_rstd, y_wrapper, x_out, mxscale_wrapper, rstd_out);
    } else {
        // beta 为空时，使用空 tensor 或按空指针处理
        at::Tensor beta_dummy = at::Tensor();
        EXEC_NPU_CMD(aclnnAddRmsNormDynamicMxQuant, x1, x2, gamma, beta_dummy, epsilon, scale_alg,
                     round_mode_ptr, y_acltype, output_rstd, y_wrapper, x_out, mxscale_wrapper, rstd_out);
    }

    return std::make_tuple(y, x_out, mxscale_out, rstd_out);
}

} // namespace op_api
