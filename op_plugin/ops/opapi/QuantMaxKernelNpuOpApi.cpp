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

std::tuple<at::Tensor, at::Tensor> npu_quant_max(
    const at::Tensor &x,
    const at::Tensor &scale,
    c10::string_view round_mode,
    c10::optional<int64_t> dst_dtype)
{
    TORCH_CHECK(round_mode == "rint" || round_mode == "round" || round_mode == "hybrid",
        "round_mode only supports 'rint', 'round' and 'hybrid', but got '",
        std::string(round_mode), "'" + OPS_ERROR(ErrCode::PARAM));

    int64_t dst_type = dst_dtype.value_or(291);
    TORCH_CHECK(dst_type == 290 || dst_type == 291 || dst_type == 292,
        "dst_dtype only supports 290(hifloat8), 291(float8_e5m2) and 292(float8_e4m3fn), but got ",
        dst_type, OPS_ERROR(ErrCode::PARAM));

    aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
    at::ScalarType y_scalar_type = npu_preparation::convert_to_scalar_type(y_acltype);

    at::Tensor y = npu_preparation::apply_tensor_without_format(x.sizes(), x.options().dtype(y_scalar_type));
    at::Tensor amax = npu_preparation::apply_tensor_without_format({1}, x.options().dtype(x.scalar_type()));

    const char *round_mode_ptr = round_mode.data();
    TensorWrapper y_wrapper = {y, y_acltype};
    EXEC_NPU_CMD(aclnnQuantMax, x, scale, y_acltype, round_mode_ptr, y_wrapper, amax);

    return std::make_tuple(y, amax);
}
} // namespace op_api
