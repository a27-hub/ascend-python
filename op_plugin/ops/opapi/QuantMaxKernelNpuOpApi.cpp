// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

// 将 PyTorch 标准 dtype 枚举值映射到 torch_npu 扩展枚举值
// torch.float8_e5m2 = 23 -> 291, torch.float8_e4m3fn = 24 -> 292
static int64_t NormalizeDstDtype(int64_t dst_dtype)
{
    constexpr int64_t TORCH_FLOAT8_E5M2 = 23;
    constexpr int64_t TORCH_FLOAT8_E4M3FN = 24;
    constexpr int64_t NPU_FLOAT8_E5M2 = 291;
    constexpr int64_t NPU_FLOAT8_E4M3FN = 292;

    if (dst_dtype == TORCH_FLOAT8_E5M2) {
        return NPU_FLOAT8_E5M2;
    }
    if (dst_dtype == TORCH_FLOAT8_E4M3FN) {
        return NPU_FLOAT8_E4M3FN;
    }
    return dst_dtype;
}

std::tuple<at::Tensor, at::Tensor> npu_quant_max(
    const at::Tensor &x,
    const at::Tensor &scale,
    c10::string_view round_mode,
    c10::optional<int64_t> dst_dtype)
{
    // 处理 round_mode：如果为空则使用默认值 "rint"
    std::string round_mode_str = round_mode.empty() ? "rint" : std::string(round_mode);
    TORCH_CHECK(round_mode_str == "rint" || round_mode_str == "round" || round_mode_str == "hybrid",
        "round_mode only supports 'rint', 'round' and 'hybrid', but got '", round_mode_str,
        "'" + OPS_ERROR(ErrCode::PARAM));

    // 校验 scale 必须为一维 tensor，shape 为 [1]
    TORCH_CHECK(scale.dim() == 1,
        "scale must be a 1D tensor with shape [1], but got dim: ", scale.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(scale.size(0) == 1,
        "scale shape must be [1], but got shape: [", scale.size(0), "]",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float,
        "scale dtype must be torch.float32, but got: ", scale.scalar_type(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(scale.numel() > 0,
        "scale does not support empty tensor", OPS_ERROR(ErrCode::PARAM));

    // 处理 dst_dtype：如果为空则使用默认值，否则校验值的有效性
    int64_t dst_dtype_value = NormalizeDstDtype(dst_dtype.value_or(291));
    TORCH_CHECK(dst_dtype_value == 290 || dst_dtype_value == 291 || dst_dtype_value == 292,
        "dst_dtype only supports 290(hifloat8), 291(float8_e5m2) and 292(float8_e4m3fn), but got ",
        dst_dtype_value, OPS_ERROR(ErrCode::PARAM));

    // 校验 round_mode 与 dst_dtype 的匹配关系
    TORCH_CHECK(!((dst_dtype_value == 291 || dst_dtype_value == 292) && round_mode_str != "rint"),
        "When dst_dtype is float8_e5m2(291) or float8_e4m3fn(292), round_mode must be 'rint', but got '",
        round_mode_str, "'" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!(dst_dtype_value == 290 && round_mode_str != "round" && round_mode_str != "hybrid"),
        "When dst_dtype is hifloat8(290), round_mode must be 'round' or 'hybrid', but got '",
        round_mode_str, "'" + OPS_ERROR(ErrCode::PARAM));

    aclDataType y_acltype = c10_npu::GetAclDataType(dst_dtype_value);
    at::ScalarType y_scalar_type = npu_preparation::convert_to_scalar_type(y_acltype);

    at::Tensor y = npu_preparation::apply_tensor_without_format(x.sizes(), x.options().dtype(y_scalar_type));
    at::SmallVector<int64_t, op_infer::SIZE> amax_size = {1};
    at::Tensor amax = npu_preparation::apply_tensor_without_format(amax_size, x.options().dtype(x.scalar_type()));

    const char *round_mode_ptr = round_mode_str.data();
    TensorWrapper y_wrapper = {y, y_acltype};
    EXEC_NPU_CMD(aclnnQuantMax, x, scale, y_acltype, round_mode_ptr, y_wrapper, amax);

    return std::make_tuple(y, amax);
}
} // namespace op_api