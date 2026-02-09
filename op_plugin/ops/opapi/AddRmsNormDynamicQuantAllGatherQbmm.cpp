// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

static c10::SmallVector<int64_t, op_infer::SIZE> get_output_size_mm(const at::Tensor &x1, const at::Tensor &x2)
{
    auto out_x = x1.size(0) * world_size;
    auto out_y = x2.size(1);
    return {out_x, out_y};
}

std::tuple<at::Tensor, at::Tensor> npu_add_rms_norm_dynamic_quant_all_gather_qbmm(const at::Tensor &x1,
    const at::Tensor &x2, const at::Tensor &residual, const at::Tensor &y, const at::Tensor &gamma, const at::Tensor &scale,
    const at::Tensor &smooth_scale, c10::string_view hcom, int64_t world_size, const c10::optional<at::Tensor> &bias, bool transpose_x2,
    c10::optional<int64_t> output_dtype, c10::optional<int64_t> residual_norm_mode)
{
    TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8 || world_size == 16 || world_size == 32,
                "world_size should be in [2, 4, 8, 16, 32], but the actual value is ", world_size, "."
                + OPS_ERROR(ErrCode::PARAM));
    // TORCH_CHECK(self.dim() == 2 && x2.dim() == 2, "Both inputs of mm are required to be 2D, but the actual inputs are ",
    //             self.dim(), "D and ", x2.dim(), "D." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.size(1) == x2.size(0),
                "The K-axis in the two inputs of Matmul must be equal, but in reality, the K-axis of x1 is ",
                self.size(1), " and the K-axis of x2 is ", x2.size(0), "." + OPS_ERROR(ErrCode::PARAM));

    const char* hcom_value = (char*)hcom.data();
    auto z_scalar_type = x1.scalar_type();
    auto output_scalar_type = x1.scalar_type();
    if (output_dtype.has_value()) {
        output_scalar_type = npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(output_dtype.value()));
    }
    c10::SmallVector<int64_t, op_infer::SIZE> output_size = get_output_size_mm(x1, x2);
    at::Tensor output =
        npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(output_scalar_type));
    auto z = npu_preparation::OpPreparation::apply_tensor_without_format(x1.sizes(), self.options().dtype(z_scalar_type));
    char *hcom_ptr = const_cast<char *>(hcom.data());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    auto residualNormMode = residual_norm_mode.value_or(0);
    TORCH_CHECK(!has_quant, "When comm_mode is ai_cpu, quantization not supported." + OPS_ERROR(ErrCode::PARAM));
    EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuantAllGatherQbmm, x1, x2, residual, y, gamma, scale, smooth_scale, bias_real,
        hcom_value, transpose_x2, residualNormMode, output, z);
    return std::tie(output, z);
}
}
