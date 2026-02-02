// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include <set>
#include <op_plugin/OpApiInterface.h>
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
static const int NON_GROUP = 0;
static const int TWO_DIM = 2;
static const int THREE_DIM = 3;
const int32_t MIN_SUPPORT_WORLD_SIZE = 2;
const int32_t MAX_SUPPORT_WORLD_SIZE = 256;

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_alltoallv_quant_gmm(const at::Tensor &gmm_x,
    const at::Tensor &gmm_weight, const at::Tensor &gmm_x_scale, const at::Tensor &gmm_weight_scale,
    c10::string_view hcom, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts,
    int64_t gmm_y_dtype, const c10::optional<at::Tensor> &send_counts_tensor,
    const c10::optional<at::Tensor> &recv_counts_tensor, const c10::optional<at::Tensor> &mm_x,
    const c10::optional<at::Tensor> &mm_weight, const c10::optional<at::Tensor> &mm_x_scale,
    const c10::optional<at::Tensor> &mm_weight_scale, const c10::optional<at::Tensor> &gmm_x_offset,
    const c10::optional<at::Tensor> &gmm_weight_offset, const c10::optional<at::Tensor> &mm_x_offset,
    const c10::optional<at::Tensor> &mm_weight_offset, c10::optional<int64_t> gmm_x_quant_mode,
    c10::optional<int64_t> gmm_weight_quant_mode, c10::optional<int64_t> mm_x_quant_mode,
    c10::optional<int64_t> mm_weight_quant_mode, bool permute_out_flag, c10::OptionalIntArrayRef group_size,
    c10::optional<int64_t> gmm_x_dtype, c10::optional<int64_t> gmm_weight_dtype,
    c10::optional<int64_t> gmm_x_scale_dtype, c10::optional<int64_t> gmm_weight_scale_dtype,
    c10::optional<int64_t> mm_x_dtype, c10::optional<int64_t> mm_weight_dtype, c10::optional<int64_t> mm_x_scale_dtype,
    c10::optional<int64_t> mm_weight_scale_dtype, c10::optional<int64_t> mm_y_dtype)
{
    TORCH_CHECK(gmm_x.dim() == TWO_DIM,
        "The gmm_x input of gmm is required to be 2D, but the actual gmm_x input is ",
        gmm_x.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gmm_weight.dim() == THREE_DIM,
        "The gmm_weight input of gmm is required to be 3D, but the actual gmm_weight input is ",
        gmm_weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(ep_world_size >= MIN_SUPPORT_WORLD_SIZE && ep_world_size <= MAX_SUPPORT_WORLD_SIZE &&
                (ep_world_size & (ep_world_size - 1)) == 0,
                "ep_world_size should be in [2, 4, 8, 16, 32, 64, 128, 256], but actual value is ", ep_world_size,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gmm_x_quant_mode == 1, "gmm_x_quant_mode only support 1.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gmm_weight_quant_mode == 1, "gmm_weight_quant_mode only support 1.", OPS_ERROR(ErrCode::PARAM));

    at::Tensor mm_y{nullptr};
    at::Tensor permute_out{nullptr};

    int64_t a = 0;
    for (auto &i : recv_counts) {
        a += i;
    }
    if (mm_x.has_value() && mm_weight.has_value()) {
        TORCH_CHECK(mm_x_quant_mode == 1, "mm_x_quant_mode only support 1.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(mm_weight_quant_mode == 1, "mm_weight_quant_mode only support 1.", OPS_ERROR(ErrCode::PARAM));

        const at::Tensor &mm_x_value = mm_x.value();
        const at::Tensor &mm_weight_value = mm_weight.value();
        int64_t bs = mm_x_value.size(0);
        int64_t n2 = mm_weight_value.size(1);

        // 推导输出mm_y的shape
        aclDataType mm_y_acltype = c10_npu::GetAclDataType(mm_y_dtype.value());
        at::ScalarType mm_y_scalar_dtype = at_npu::native::OpPreparation::convert_to_scalar_type(mm_y_acltype);
        mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, c10::dtype(mm_y_scalar_dtype));
    }

    if (permute_out_flag) {
        int64_t h1 = gmm_x.size(1);
        // 推导输出permute_out的shape
        aclDataType permute_out_acltype = c10_npu::GetAclDataType(gmm_x_dtype.value());
        at::ScalarType permute_out_scalar_dtype =
            at_npu::native::OpPreparation::convert_to_scalar_type(permute_out_acltype);
        permute_out =
            at_npu::native::OpPreparation::apply_tensor_without_format({a, h1}, c10::dtype(permute_out_scalar_dtype));
    }
    TensorWrapper permute_out_wrapper = {permute_out,
        (gmm_x_dtype.has_value()) ? c10_npu::GetAclDataType(gmm_x_dtype.value())
                                  : at_npu::native::OpPreparation::convert_to_acl_data_type(gmm_x.scalar_type())};

    int64_t n1 = gmm_weight.size(2);
    // 推导输出gmm_y的shape
    aclDataType gmm_y_acltype = c10_npu::GetAclDataType(gmm_y_dtype);
    at::ScalarType gmm_y_scalar_dtype = at_npu::native::OpPreparation::convert_to_scalar_type(gmm_y_acltype);
    auto gmm_y = at_npu::native::OpPreparation::apply_tensor_without_format({a, n1}, c10::dtype(gmm_y_scalar_dtype));

    const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());
    const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
    const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
    const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
    const char *hcom_ptr = const_cast<char *>(hcom.data());

    const at::Tensor &mm_x_scale_real = mm_x_scale.value_or(at::Tensor());
    const at::Tensor &mm_weight_scale_real = mm_weight_scale.value_or(at::Tensor());
    const at::Tensor &gmm_x_offset_real = gmm_x_offset.value_or(at::Tensor());
    const at::Tensor &gmm_weight_offset_real = gmm_weight_offset.value_or(at::Tensor());
    const at::Tensor &mm_x_offset_real = mm_x_offset.value_or(at::Tensor());
    const at::Tensor &mm_weight_offset_real = mm_weight_offset.value_or(at::Tensor());

    int64_t gmm_x_quant_mode_real = gmm_x_quant_mode.value();
    int64_t gmm_weight_quant_mode_real = gmm_weight_quant_mode.value();
    int64_t mm_x_quant_mode_real = mm_x_quant_mode.value();
    int64_t mm_weight_quant_mode_real = mm_weight_quant_mode.value();
    int64_t group_sizes = NON_GROUP;
    bool transposeGmmWeight = false;
    bool transposeMmWeight = false;

    TensorWrapper gmm_x_wrapper = make_wrapper(gmm_x, gmm_x_dtype);
    TensorWrapper gmm_weight_wrapper = make_wrapper(gmm_weight, gmm_weight_dtype);
    TensorWrapper mm_x_wrapper = make_wrapper(mm_x_real, mm_x_dtype);
    TensorWrapper mm_weight_wrapper = make_wrapper(mm_weight_real, mm_weight_dtype);

    EXEC_NPU_CMD(aclnnAlltoAllvQuantGroupedMatMul,
        gmm_x_wrapper,
        gmm_weight_wrapper,
        gmm_x_scale,
        gmm_weight_scale,
        gmm_x_offset_real,
        gmm_weight_offset_real,
        send_count_tensor_real,
        recv_count_tensor_real,
        mm_x_wrapper,
        mm_weight_wrapper,
        mm_x_scale_real,
        mm_weight_scale_real,
        mm_x_offset_real,
        mm_weight_offset_real,
        gmm_x_quant_mode_real,
        gmm_weight_quant_mode_real,
        mm_x_quant_mode_real,
        mm_weight_quant_mode_real,
        hcom_ptr,
        ep_world_size,
        send_counts,
        recv_counts,
        transposeGmmWeight,
        transposeMmWeight,
        group_sizes,
        permute_out_flag,
        gmm_y,
        mm_y,
        permute_out_wrapper);
    return std::tie(gmm_y, mm_y, permute_out);
}
}  // namespace op_api