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
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

static c10::SmallVector<int64_t, op_infer::SIZE> get_output_size_mm(const at::Tensor &x1, const at::Tensor &x2, int64_t world_size)
{
    auto out_x = x1.size(0) * world_size;
    auto out_y = x2.size(1);
    return {out_x, out_y};
}

std::tuple<at::Tensor, at::Tensor> npu_add_rms_norm_dynamic_quant_all_gather_qbmm(const at::Tensor &x1,
    const at::Tensor &x2, const at::Tensor &residual, const at::Tensor &y, const at::Tensor &gamma, const at::Tensor &scale,
    c10::string_view group, const c10::optional<at::Tensor> &smooth_scale, const c10::optional<at::Tensor> &bias, int64_t world_size,
    bool transpose_x2, int64_t output_dtype, int64_t residual_norm_mode)
{
    TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8 || world_size == 16 || world_size == 32,
                "world_size should be in [2, 4, 8, 16, 32], but the actual value is ", world_size, "."
                + OPS_ERROR(ErrCode::PARAM));
    // TORCH_CHECK(x1.size(1) == x2.size(0),
    //             "The K-axis in the two inputs of Matmul must be equal, but in reality, the K-axis of x1 is ",
    //             x1.size(1), " and the K-axis of x2 is ", x2.size(0), "." + OPS_ERROR(ErrCode::PARAM));

    const char* hcom_value = (char*)group.data();
    auto z_scalar_type = x1.scalar_type();
    auto output_scalar_type = x1.scalar_type();
    if (output_dtype != -1) {
        output_scalar_type = npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(output_dtype));
    }
    c10::SmallVector<int64_t, op_infer::SIZE> output_size = get_output_size_mm(x1, x2, world_size);
    at::Tensor output =
        npu_preparation::apply_tensor_without_format(output_size, x1.options().dtype(output_scalar_type));
    at::Tensor z = npu_preparation::apply_tensor_without_format(x1.sizes(), x1.options().dtype(z_scalar_type));
    char *hcom_ptr = const_cast<char *>(group.data());
    const at::Tensor &smooth_scale_real = smooth_scale.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuantAllGatherQbmm, x1, x2, residual, y, gamma, scale, smooth_scale_real, bias_real,
        hcom_value, world_size, transpose_x2, output_dtype, residual_norm_mode, output, z);
    return std::tie(output, z);
}
}
