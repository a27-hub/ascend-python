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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
const static int64_t IN_NOT_SPLIT_OUT_SPLIT = 2;
const static int64_t IN_SPLIT_OUT_SPLIT = 3;
const static int64_t M_SPLIT = 0;
const static int64_t K_SPLIT = 2;
const static int64_t ACT_TYPE_NONE = 0;
const static int64_t GROUP_LIST_TYPE_CUMSUM = 0;
using npu_preparation = at_npu::native::OpPreparation;

// Execute aclnnGroupedMatmulV5 (with fallback to V4 if V5 is not available).
// All optional quantization/activation parameters are passed as empty TensorLists.
static void exec_grouped_matmul_v5(
    const std::vector<at::Tensor>& x_vec,
    const std::vector<at::Tensor>& weight_vec,
    const at::Tensor& group_list,
    int64_t split_item,
    int64_t group_type,
    std::vector<at::Tensor>& result_vec)
{
    auto x = at::TensorList(x_vec);
    auto weight = at::TensorList(weight_vec);
    auto bias_real = at::TensorList();
    auto scale_real = at::TensorList();
    auto offset_real = at::TensorList();
    auto antiquant_scale_real = at::TensorList();
    auto antiquant_offset_real = at::TensorList();
    auto per_token_scale_real = at::TensorList();
    auto activation_input_real = at::TensorList();
    auto activation_quant_scale_real = at::TensorList();
    auto activation_quant_offset_real = at::TensorList();
    auto act_out = at::TensorList();
    auto dynamic_quant_scale_out = at::TensorList();
    auto result = at::TensorList(result_vec);
    int64_t group_list_type_value = GROUP_LIST_TYPE_CUMSUM;
    int64_t act_type_value = ACT_TYPE_NONE;
    auto tuning_config_real = at::IntArrayRef{};

    static const bool is_v5_available = check_aclnn_kernel_available("aclnnGroupedMatmulV5");
    if (is_v5_available) {
        EXEC_NPU_CMD(aclnnGroupedMatmulV5, x, weight, bias_real, scale_real, offset_real,
            antiquant_scale_real, antiquant_offset_real, per_token_scale_real, group_list,
            activation_input_real, activation_quant_scale_real, activation_quant_offset_real,
            split_item, group_type, group_list_type_value, act_type_value, tuning_config_real,
            result, act_out, dynamic_quant_scale_out);
    } else {
        EXEC_NPU_CMD(aclnnGroupedMatmulV4, x, weight, bias_real, scale_real, offset_real,
            antiquant_scale_real, antiquant_offset_real, per_token_scale_real, group_list,
            activation_input_real, activation_quant_scale_real, activation_quant_offset_real,
            split_item, group_type, group_list_type_value, act_type_value,
            result, act_out, dynamic_quant_scale_out);
    }
}

// Convert offs from int32 to int64, ensure contiguous.
// The NPU API requires group_list to be int64.
static at::Tensor prepare_group_list(const at::Tensor& offs)
{
    at::Tensor group_list = offs.to(at::kLong);
    if (group_list.is_contiguous()) {
        return group_list;
    }
    return group_list.contiguous();
}

// Scenario 1: 2D x 3D — mat_a(M, K) x mat_b(G, K, N), offs(G,) -> out(M, N)
// Maps to V5 groupType=0 (M-axis split), splitItem=3, single x + single 3D weight
static at::Tensor grouped_mm_2d_x_3d(const at::Tensor& mat_a, const at::Tensor& mat_b,
                                      const at::Tensor& offs, c10::ScalarType out_dtype)
{
    int64_t M = mat_a.size(0);
    int64_t N = mat_b.size(-1);
    auto options = mat_a.options().dtype(out_dtype);
    auto output_size = op_infer::array_to_small_vector({M, N});
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, options);

    auto group_list = prepare_group_list(offs);
    std::vector<at::Tensor> x_vec = {mat_a};
    std::vector<at::Tensor> weight_vec = {mat_b};
    std::vector<at::Tensor> result_vec = {out};

    exec_grouped_matmul_v5(x_vec, weight_vec, group_list, IN_SPLIT_OUT_SPLIT, M_SPLIT, result_vec);
    return out;
}

// Scenario 2: 3D x 2D — mat_a(G, M, K) x mat_b(K, N), offs(G,) -> out(M, N)
// Uses transpose trick: out^T(N, M) = mat_b^T(N, K) x mat_a^T(G, K, M)
// Maps to V5 groupType=0 (M-axis split on transposed N dim), splitItem=3
static at::Tensor grouped_mm_3d_x_2d(const at::Tensor& mat_a, const at::Tensor& mat_b,
                                      const at::Tensor& offs, c10::ScalarType out_dtype)
{
    int64_t N = mat_b.size(-1);
    int64_t M = mat_a.size(1);
    auto options = mat_a.options().dtype(out_dtype);
    auto output_size = op_infer::array_to_small_vector({N, M});
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, options);

    auto group_list = prepare_group_list(offs);
    // x = mat_b^T -> (N, K), weight = mat_a^T -> (G, K, M)
    at::Tensor x_tensor = mat_b.transpose(-2, -1).contiguous();
    at::Tensor weight_tensor = mat_a.transpose(-2, -1).contiguous();
    std::vector<at::Tensor> x_vec = {x_tensor};
    std::vector<at::Tensor> weight_vec = {weight_tensor};
    std::vector<at::Tensor> result_vec = {out};

    exec_grouped_matmul_v5(x_vec, weight_vec, group_list, IN_SPLIT_OUT_SPLIT, M_SPLIT, result_vec);
    // out is (N, M), transpose back to (M, N)
    return out.transpose(-2, -1).contiguous();
}

// Scenario 3: 2D x 2D — mat_a(M, K) x mat_b(K, N), offs(G,) -> out(G, M, N)
// Maps to V5 groupType=2 (K-axis split), splitItem=2
// V5 requires x to be transposed: (K, M), non-contiguous (transposed strides)
// weight = mat_b as-is (K, N)
static at::Tensor grouped_mm_2d_x_2d(const at::Tensor& mat_a, const at::Tensor& mat_b,
                                      const at::Tensor& offs, c10::ScalarType out_dtype)
{
    int64_t M = mat_a.size(0);
    int64_t N = mat_b.size(1);
    int64_t G = offs.size(0);
    auto options = mat_a.options().dtype(out_dtype);
    auto output_size = op_infer::array_to_small_vector({G, M, N});
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, options);

    auto group_list = prepare_group_list(offs);
    // V5 groupType=2 requires x transposed to (K, M), MUST be non-contiguous
    at::Tensor x_tensor = mat_a.t();  // (K, M), non-contiguous
    std::vector<at::Tensor> x_vec = {x_tensor};
    std::vector<at::Tensor> weight_vec = {mat_b};
    std::vector<at::Tensor> result_vec = {out};

    exec_grouped_matmul_v5(x_vec, weight_vec, group_list, IN_NOT_SPLIT_OUT_SPLIT, K_SPLIT, result_vec);
    return out;
}

// Scenario 4: 3D x 3D — mat_a(G, M, K) x mat_b(G, K, N), no offs -> out(G, M, N)
// Direct bmm — NPU already has optimized aclnnBatchMatMul dispatch
static at::Tensor grouped_mm_3d_x_3d(const at::Tensor& mat_a, const at::Tensor& mat_b,
                                     c10::ScalarType out_dtype)
{
    auto options = mat_a.options().dtype(out_dtype);
    auto output_size = op_infer::array_to_small_vector({mat_a.size(0), mat_a.size(1), mat_b.size(-1)});
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, options);
    at::bmm_out(out, mat_a, mat_b);
    return out;
}

at::Tensor _grouped_mm(const at::Tensor& mat_a, const at::Tensor& mat_b,
                       const c10::optional<at::Tensor>& offs,
                       const c10::optional<at::Tensor>& bias,
                       c10::optional<c10::ScalarType> out_dtype)
{
    // Validate inputs (mirrors native PyTorch GroupedMMUtils.h checks)
    TORCH_CHECK(mat_a.dtype() == at::kBFloat16 || mat_a.dtype() == at::kFloat || mat_a.dtype() == at::kHalf,
                "Expected mat_a to be Float32, BFloat16 or Float16 matrix, got ", mat_a.scalar_type(),
                OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mat_b.dtype() == at::kBFloat16 || mat_b.dtype() == at::kFloat || mat_b.dtype() == at::kHalf,
                "Expected mat_b to be Float32, BFloat16 or Float16 matrix, got ", mat_b.scalar_type(),
                OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mat_a.dtype() == mat_b.dtype(),
                "Expected mat_a and mat_b to have the same dtype, got ",
                mat_a.scalar_type(), " and ", mat_b.scalar_type(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3,
                "mat_a has to be 2 or 3d", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3,
                "mat_b has to be 2 or 3d", OPS_ERROR(ErrCode::PARAM));

    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;

    // Check contraction dimension matches
    TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2),
                "contraction dimension of mat_a and mat_b must match",
                OPS_ERROR(ErrCode::PARAM));

    // Check offs presence: required when any input is 2D, forbidden when both are 3D
    TORCH_CHECK(offs.has_value() == (a_is_2d || b_is_2d),
                "Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d",
                OPS_ERROR(ErrCode::PARAM));

    if (offs.has_value()) {
        TORCH_CHECK(offs->dim() == 1, "offs has to be 1D", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32",
                    OPS_ERROR(ErrCode::TYPE));
    }

    // bias not supported yet (consistent with native PyTorch)
    TORCH_CHECK(!bias.has_value(), "Bias not supported yet", OPS_ERROR(ErrCode::PARAM));

    // Resolve output dtype
    const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    TORCH_CHECK(out_dtype_ == mat_a.dtype(),
                "Grouped gemm output dtype must match mat_a dtype",
                OPS_ERROR(ErrCode::TYPE));

    // Dispatch to the appropriate scenario
    if (a_is_2d && !b_is_2d) {
        // Scenario 1: 2D x 3D — MoE forward
        return grouped_mm_2d_x_3d(mat_a, mat_b, offs.value(), out_dtype_);
    } else if (!a_is_2d && b_is_2d) {
        // Scenario 2: 3D x 2D — uses transpose trick to become 2D x 3D
        return grouped_mm_3d_x_2d(mat_a, mat_b, offs.value(), out_dtype_);
    } else if (a_is_2d && b_is_2d) {
        // Scenario 3: 2D x 2D — K-axis split, each group independent
        return grouped_mm_2d_x_2d(mat_a, mat_b, offs.value(), out_dtype_);
    } else {
        // Scenario 4: 3D x 3D — standard BMM
        return grouped_mm_3d_x_3d(mat_a, mat_b, out_dtype_);
    }
}

}  // namespace op_api
