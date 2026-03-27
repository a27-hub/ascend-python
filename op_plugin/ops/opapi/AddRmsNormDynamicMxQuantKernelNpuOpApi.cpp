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
constexpr int64_t DEFAULT_EPSILON = 1e-6;
constexpr int64_t DEFAULT_SCALE_ALG = 0;
constexpr int64_t DEFAULT_DST_TYPE = 296;  // float4_e2m1fn_x2
constexpr int64_t ALIGN_NUM = 32;
}

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
        "x1 should be 1-7D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1.dim() == x2.dim(),
        "x1 and x2 should have same dims" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gamma.dim() == 1,
        "gamma should be 1D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gamma.size(0) == x1.size(-1),
        "gamma shape should match x1 last dim" + OPS_ERROR(ErrCode::PARAM));

    // 2. 推导输出 shape
    auto x1_sizes = op_infer::array_to_small_vector(x1.sizes());
    int64_t dim_num = x1.dim();
    int64_t last_dim = x1_sizes[dim_num - 1];

    // y shape: 与 x1 相同，但 FP4 时尾轴减半
    std::vector<int64_t> y_shape = x1_sizes;
    at::ScalarType y_scalar_type;
    aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
    y_scalar_type = npu_preparation::convert_to_scalar_type(y_acltype);

    // FP4 特殊处理：uint8 承载，尾轴减半
    if (dst_type == 296 || dst_type == 297) {  // float4_e2m1fn_x2 or float4_e1m2fn_x2
        TORCH_CHECK(last_dim % 2 == 0,
            "last dim must be even for FP4" + OPS_ERROR(ErrCode::PARAM));
        y_shape[dim_num - 1] = last_dim / 2;
        y_scalar_type = at::ScalarType::Byte;
        y_acltype = ACL_UINT8;
    }

    // x_out shape: 与 x1 相同
    std::vector<int64_t> x_out_shape = x1_sizes;

    // mxscale_out shape: 尾轴特殊计算
    // mxscale_out.shape[-2] = (ceil(x1.shape[-1] / 32) + 1) / 2 向上取整到偶数
    int64_t scale_count = (last_dim + ALIGN_NUM - 1) / ALIGN_NUM;  // ceil(last_dim / 32)
    int64_t padded_scale_count = (scale_count + 1) / 2 * 2;  // 偶数 pad
    std::vector<int64_t> mxscale_shape = x1_sizes;
    mxscale_shape[dim_num - 1] = padded_scale_count / 2;  // 存储为 uint8，两个 E8M0 合一个 uint8
    // 添加最后一维 2
    mxscale_shape.push_back(2);

    // rstd_out shape: 与 x1 前几维相同，最后为1
    std::vector<int64_t> rstd_shape;
    for (int64_t i = 0; i < dim_num - 1; i++) {
        rstd_shape.push_back(x1_sizes[i]);
    }
    rstd_shape.push_back(1);

    // 3. output_rstd 由 requires_grad 决定
    bool output_rstd = x1.requires_grad() && x2.requires_grad();

    // 如果不需要输出 rstd，shape 设为0
    if (!output_rstd) {
        rstd_shape = {0};
    }

    // 4. 字符串参数转换
    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    // 5. 分配输出 tensor
    at::Tensor y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(y_scalar_type));
    at::Tensor x_out = npu_preparation::apply_tensor_without_format(x_out_shape, x1.options());
    at::Tensor mxscale_out = npu_preparation::apply_tensor_without_format(mxscale_shape, c10::dtype(at::ScalarType::Byte));
    at::Tensor rstd_out = npu_preparation::apply_tensor_without_format(rstd_shape, x1.options());

    // 使用 TensorWrapper 绑定真实 dtype
    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper mxscale_wrapper = {mxscale_out, ACL_FLOAT8_E8M0};

    // 6. 可选参数处理
    double eps_val = (epsilon == 0.0) ? DEFAULT_EPSILON : epsilon;

    // 7. EXEC_NPU_CMD 调用 aclnn 接口
    // aclnnAddRmsNormDynamicMxQuant(x1, x2, gamma, beta, epsilon, scaleAlg, roundMode, dstType, outputRstd, yOut, xOut, mxscaleOut, rstdOut)
    EXEC_NPU_CMD(aclnnAddRmsNormDynamicMxQuant,
                 x1, x2, gamma, beta, eps_val, scale_alg, round_mode_ptr,
                 static_cast<int>(dst_type - 256),  // 转为 aclDataType
                 output_rstd,
                 y_wrapper, x_out, mxscale_wrapper, rstd_out);

    return std::make_tuple(y, x_out, mxscale_out, rstd_out);
}

} // namespace op_api
