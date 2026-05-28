// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include <functional>
#include <tuple>
#include <array>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {

using acceptance_fn = std::function<bool(c10::ScalarType, std::vector<ScalingType>&, c10::ArrayRef<at::Tensor>&, c10::ScalarType, std::vector<ScalingType>&, c10::ArrayRef<at::Tensor>&)>;

// Namespace for scaled_blas check functions
namespace scaled_blas {

// Check rowwise recipe
bool check_rowwise_recipe(c10::ScalarType a_type, std::vector<ScalingType>& a_recipe, c10::ArrayRef<at::Tensor>& a_scale,
                          c10::ScalarType b_type, std::vector<ScalingType>& b_recipe, c10::ArrayRef<at::Tensor>& b_scale) {
    if (a_type != c10::ScalarType::Float8_e4m3fn && a_type != c10::ScalarType::Float8_e5m2) {
        return false;
    }
    if (b_type != c10::ScalarType::Float8_e4m3fn && b_type != c10::ScalarType::Float8_e5m2) {
        return false;
    }
    if (a_recipe.empty() || b_recipe.empty()) {
        return false;
    }
    if (a_scale.empty() || b_scale.empty()) {
        return false;
    }
    return (a_recipe[0] == ScalingType::RowWise && b_recipe[0] == ScalingType::RowWise);
}

// Check mxfp8 recipe
bool check_mxfp8_recipe(c10::ScalarType a_type, std::vector<ScalingType>& a_recipe, c10::ArrayRef<at::Tensor>& a_scale,
                        c10::ScalarType b_type, std::vector<ScalingType>& b_recipe, c10::ArrayRef<at::Tensor>& b_scale) {
    if (a_type != c10::ScalarType::Float8_e4m3fn && a_type != c10::ScalarType::Float8_e5m2) {
        return false;
    }
    if (b_type != c10::ScalarType::Float8_e4m3fn && b_type != c10::ScalarType::Float8_e5m2) {
        return false;
    }
    if (a_recipe.empty() || b_recipe.empty()) {
        return false;
    }
    if (a_scale.empty() || b_scale.empty()) {
        return false;
    }
    return true;
}
} // namespace scaled_blas

std::array<std::tuple<std::string, acceptance_fn, ScaledGemmImplementation>, 2> scale_grouped_kernel_dispatch = {{
    {"rowwise_rowwise", scaled_blas::check_rowwise_recipe, ScaledGemmImplementation::ROWWISE_ROWWISE},
    {"mxfp8_mxfp8", scaled_blas::check_mxfp8_recipe, ScaledGemmImplementation::MXFP8_MXFP8},
}};

const static int64_t IN_NOT_SPLIT_OUT_NOT_SPLIT = 0;
const static int64_t IN_SPLIT_OUT_NOT_SPLIT = 1;
const static int64_t IN_NOT_SPLIT_OUT_SPLIT = 2;
const static int64_t IN_SPLIT_OUT_SPLIT = 3;
const static int64_t INT4_NUMS_IN_INT32 = 8;
const static int64_t DEFAULT_SPLIT = -1;
const static int64_t M_SPLIT = 0;
const static int64_t K_SPLIT = 2;
using npu_preparation = at_npu::native::OpPreparation;

static void check_dims(int64_t split_item, size_t num_x, size_t num_weight, size_t num_group_list) {
    TORCH_CHECK(num_x > 0 && num_weight > 0,
        "Invalid inputs: neither x nor weight could be empty." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(split_item == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item == IN_SPLIT_OUT_NOT_SPLIT ||
            split_item == IN_NOT_SPLIT_OUT_SPLIT || split_item == IN_SPLIT_OUT_SPLIT,
        "Invalid value of split_item [", split_item,
        "], which should only be one of 0/1/2/3." + OPS_ERROR(ErrCode::PARAM));
    if (split_item == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item == IN_SPLIT_OUT_NOT_SPLIT) {
        if (num_group_list > 0) {
            TORCH_CHECK(num_x == 1 && num_weight == num_group_list,
                "Invalid inputs. "
                "When split_item = 0 or 1 and input group_list is not None, "
                "the following two conditions are supposed to be satisfied: "
                "(1) length of x equals 1; (2) length of weight equals that of group_list. "
                "Actual lengths: x [",
                num_x, "], weight [", num_weight,
                "], "
                "group_list [",
                num_group_list, "]." + OPS_ERROR(ErrCode::PARAM));
        } else {
            TORCH_CHECK(num_x == num_weight,
                "When split_item = 0 or 1 and input group_list is None, "
                "the num of x tensors must equal the num of weight tensors."
                "Actual lengths: x [",
                num_x, "], weight [", num_weight, "]." + OPS_ERROR(ErrCode::PARAM));
        }
    }
}

static void create_new_tensor_multi_dim(
    std::vector<at::Tensor> &y, const at::Tensor &x_i, size_t n, c10::TensorOptions options) {
    auto x_sizes = x_i.sizes();
    std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
    y_sizes.at(x_sizes.size() - 1) = static_cast<int64_t>(n);

    auto output_size = op_infer::array_to_small_vector(y_sizes);
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

static void create_new_tensor(std::vector<at::Tensor> &y, size_t dim_m, size_t dim_n, c10::TensorOptions options) {
    auto output_size = op_infer::array_to_small_vector({dim_m, dim_n});
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

static void create_new_tensor_batch(
    std::vector<at::Tensor> &y, size_t batch, size_t dim_m, size_t dim_n, c10::TensorOptions options) {
    auto output_size = op_infer::array_to_small_vector({batch, dim_m, dim_n});
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

static void calculate_dim_m(size_t &dim_m, size_t num_x, const at::TensorList x) {
    for (size_t i = 0; i < num_x; i++) {
        dim_m += x[i].sizes()[0];
    }
}

static bool is_weight_trans(const at::Tensor &tensor) {
    int64_t dim1 = tensor.dim() - 1;
    int64_t dim2 = tensor.dim() - 2;
    return tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2);
}


at::Tensor _scaled_grouped_mm_v2(const at::Tensor& mat_a, const at::Tensor& mat_b,
        c10::ArrayRef<at::Tensor> scale_a,
        at::IntArrayRef scale_recipe_a,
        at::IntArrayRef swizzle_a,
        c10::ArrayRef<at::Tensor> scale_b,
        at::IntArrayRef scale_recipe_b,
        at::IntArrayRef swizzle_b,
        const c10::optional<at::Tensor> &offs, // group_list -tensor
        const c10::optional<at::Tensor> &bias, // torch not support
        c10::optional<c10::ScalarType> out_dtype,
        at::IntArrayRef contraction_dim,
        bool use_fast_accum) {

    // check A5
    TORCH_CHECK(c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
        "This interface is supported only on the Ascend950 platform and after.", OPS_ERROR(ErrCode::PARAM));

    auto scaling_a = convert_int_to_enum<ScalingType>(scale_recipe_a);
    auto scaling_b = convert_int_to_enum<ScalingType>(scale_recipe_b);

    ScaledGemmImplementation selected_impl = ScaledGemmImplementation::NONE;
    for (const auto& entry : scale_grouped_kernel_dispatch) {
        auto const& matcher = std::get<1>(entry);
        auto const& impl_val = std::get<2>(entry);

        bool valid = matcher(
            mat_a.scalar_type(),
            scaling_a,
            scale_a,
            mat_b.scalar_type(),
            scaling_b,
            scale_b
        );

        if (valid) {
            selected_impl = impl_val;
            break;
        }
    }

    TORCH_CHECK_VALUE(
        selected_impl != ScaledGemmImplementation::NONE,
        "failed to find valid scaled gemm implementation"
    );

    const bool is_allowed =
        selected_impl == ScaledGemmImplementation::ROWWISE_ROWWISE ||
        selected_impl == ScaledGemmImplementation::MXFP8_MXFP8;

    TORCH_CHECK(
        is_allowed,
        "NPU _scaled_grouped_mm_v2 only supports rowwise & mxfp8 modes; ",
        "mxfp4/nvfp4 are not yet enabled"
    );

    // 1. 基本参数校验
    auto dim_a = mat_a.dim();
    auto dim_b = mat_b.dim();
    auto dim_scale_a = scale_a[0].dim();
    auto dim_scale_b = scale_b[0].dim();

    TORCH_CHECK(dim_a == 2 || dim_a == 3,
      "Input tensor mat_a must be 2D or 3D, received ", dim_a, "D");
    TORCH_CHECK(dim_b == 2 || dim_b == 3,
      "Input tensor mat_b must be 2D or 3D, received ", dim_b, "D");
    TORCH_CHECK(dim_scale_a == 2 || dim_scale_a == 3,
      "scale_a tensor must be 2D or 3D, received ", dim_scale_a, "D");
    TORCH_CHECK(dim_scale_b == 2 || dim_scale_b == 3 || dim_scale_b == 4,
      "scale_b tensor must be 2D/3D/4D, received ", dim_scale_b, "D");

    const bool mat_a_2d = (dim_a == 2);
    const bool mat_b_2d = (dim_b == 2);
    if (!mat_a_2d || !mat_b_2d) {
      TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2),
        "Contraction dimension check failed: mat_a last dim != mat_b second last dim");
    }

    // 16 对齐校验
    TORCH_CHECK(mat_a.size(-1) % 16 == 0,
      "mat_a trailing dimension must be multiple of 16, current shape: ", mat_a.sizes());
    TORCH_CHECK(mat_b.size(-2) % 16 == 0 && mat_b.size(-1) % 16 == 0,
      "mat_b dimensions must be multiple of 16, current shape: ", mat_b.sizes());

    // 不支持的特性
    TORCH_CHECK(!bias.has_value(),
      "NPU _scaled_grouped_mm_v2 does not support bias parameter");

    // 偏移量校验
    bool require_offs = mat_a_2d || mat_b_2d;
    TORCH_CHECK(offs.has_value() == require_offs,
      "offsets are mandatory when input tensors are 2D");

    if (offs.has_value()) {
      TORCH_CHECK(offs->dim() == 1, "offsets tensor must be 1D");
      TORCH_CHECK(offs->dtype() == at::kInt, "offsets data type must be int32");
    }

    // 输出类型限制
    auto output_type = out_dtype.value_or(at::kBFloat16);
    TORCH_CHECK(output_type == at::kBFloat16,
      "NPU _scaled_grouped_mm_v2 only supports BF16 output type");
    // 2. 确定 group 数量
    int64_t num_groups = 1;
    if (offs.has_value()) {
        num_groups = offs->size(0);
    } else {
        num_groups = mat_a.size(0);
        TORCH_CHECK(mat_a.size(0) == mat_b.size(0), "mat_a and mat_b must have same number of groups when both are 3D");
    }

    // 3. mat_a -> x (TensorList)
    std::vector<at::Tensor> x_vec;
    if (a_is_2d) {
        x_vec.push_back(mat_a);
    } else {
        x_vec.push_back(mat_a.reshape({-1, mat_a.size(-1)}));
    }

    at::TensorList x = at::TensorList(x_vec);

    // 4. mat_b -> weight (TensorList)
    std::vector<at::Tensor> weight_vec;
    if (b_is_2d) {
        // mat_b 2D [K, N]: NPU singleWeight 需要 3D [G, K, N]
        auto b_expanded = mat_b[0].unsqueeze(0).expand({1, -1, -1}).contiguous();
        weight_vec.push_back(b_expanded);
    } else {
        // mat_b 3D [G, K, N]:
        weight_vec.push_back(mat_b[0]);
    }

    at::TensorList weight = at::TensorList(weight_vec);

    // 5. scale_a -> per_token_scale (TensorList)
    std::vector<at::Tensor> per_token_scale_vec;
    auto offs_int64 = offs->to(at::kLong);
    int64_t prev_offset = 0;
    for (int64_t i = 0; i < num_groups; i++) {
        int64_t curr_offset = offs_int64[i].item<int64_t>();
        per_token_scale_vec.push_back(scale_a[0].narrow(0, prev_offset, curr_offset - prev_offset));
        prev_offset = curr_offset;
    }

    c10::optional<at::TensorList> per_token_scale = c10::optional<at::TensorList>(at::TensorList(per_token_scale_vec));

    // 6. scale_b -> scale (TensorList)
    std::vector<at::Tensor> scale_vec;
    if (scale_b[0].dim() == 2 || scale_b[0].dim() == 4) {
        scale_vec.push_back(scale_b[0]);
    } else {
        scale_vec.push_back(scale_b[0].unsqueeze(0).expand({1, -1, -1, -1}).contiguous());
    }

    c10::optional<at::TensorList> scale = c10::optional<at::TensorList>(at::TensorList(scale_vec));

    // 7. offs -> group_list (c10::optional<at::Tensor>)
    c10::optional<at::Tensor> group_list = c10::nullopt;
    if (offs.has_value()) {
        group_list = offs->to(at::kLong);
    }

    // 8. split_item / group_type / group_list_type / act_type
    int64_t split_item_val = IN_NOT_SPLIT_OUT_SPLIT; // 2
    c10::optional<int64_t> split_item = split_item_val;
    c10::optional<int64_t> group_type = M_SPLIT; // -1
    c10::optional<int64_t> group_list_type = 0;
    c10::optional<int64_t> act_type = 0;

    // 9. out_dtype -> output_dtype (ACL int64 格式)
    c10::optional<int64_t> output_dtype = static_cast<int64_t>(out_dtype_);

    // 10. 其他可选参数 (默认空)
    c10::optional<at::TensorList> bias_tl = c10::nullopt;
    c10::optional<at::TensorList> offset_tl = c10::nullopt;
    c10::optional<at::TensorList> antiquant_scale_tl = c10::nullopt;
    c10::optional<at::TensorList> antiquant_offset_tl = c10::nullopt;
    c10::optional<at::TensorList> activation_input_tl = c10::nullopt;
    c10::optional<at::TensorList> activation_quant_scale_tl = c10::nullopt;
    c10::optional<at::TensorList> activation_quant_offset_tl = c10::nullopt;
    c10::OptionalIntArrayRef tuning_config = c10::OptionalIntArrayRef{};
    c10::optional<int64_t> x_dtype = c10::nullopt;
    c10::optional<int64_t> weight_dtype = c10::nullopt;
    c10::optional<int64_t> scale_dtype = c10::nullopt;
    c10::optional<int64_t> per_token_scale_dtype = c10::nullopt;

    /////////////////////////////////////////////////npu_group_matmul//////////////////////////////////////////////

    TORCH_CHECK(
        group_type.has_value(), "Requires manual passing group_type, current is None.", OPS_ERROR(ErrCode::VALUE));
    int64_t group_type_value = group_type.value();
    TORCH_CHECK(group_type_value == DEFAULT_SPLIT || group_type_value == M_SPLIT || group_type_value == K_SPLIT,
        "Use Tensor input with current cann version, "
        "The group type must be -1, 0 or 2, but now is [",
        group_type_value, "]", OPS_ERROR(ErrCode::VALUE));
    static const bool is_grouped_matmul_V4_available = check_aclnn_kernel_available("aclnnGroupedMatmulV4");
    if (C10_UNLIKELY(!is_grouped_matmul_V4_available)) {
        TORCH_CHECK(!group_list.has_value(),
            "group_list don't support Tensor input with current cann version. "
            "Please update cann version to 8.0.RC3 or higher, or use List[int] as input.",
            OPS_ERROR(ErrCode::VALUE));
        auto num_x = x.size();
        auto num_weight = weight.size();
        auto group_list_real = at::IntArrayRef{};
        size_t num_group_list = 0;
        int64_t split_item_value = split_item.value_or(0);
        check_dims(split_item_value, num_x, num_weight, num_group_list);

        std::vector<at::Tensor> y;
        c10::TensorOptions options = x[0].options().dtype(output_dtype.has_value()
                ? npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(output_dtype.value()))
                : x[0].scalar_type());

        if (split_item_value == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item_value == IN_SPLIT_OUT_NOT_SPLIT) {
            y.reserve(num_x);
            for (size_t i = 0; i < num_x; i++) {
                create_new_tensor_multi_dim(y, x[i], weight[i].size(1), options);
            }
        } else if (split_item_value == IN_NOT_SPLIT_OUT_SPLIT || split_item_value == IN_SPLIT_OUT_SPLIT) {
            if (num_x > 1) {
                size_t dim_m = 0;
                calculate_dim_m(dim_m, num_x, x);
                create_new_tensor(y, dim_m, weight[0].sizes()[1], options);
            } else if (num_x == 1) {
                create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[1], options);
            }
        }
        at::TensorList result = at::TensorList(y);

        auto bias_real = bias_tl.value_or(at::TensorList());
        auto scale_real = scale.value_or(at::TensorList());
        auto offset_real = offset_tl.value_or(at::TensorList());
        auto antiquant_scale_real = antiquant_scale_tl.value_or(at::TensorList());
        auto antiquant_offset_real = antiquant_offset_tl.value_or(at::TensorList());
        EXEC_NPU_CMD(aclnnGroupedMatmul, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real,
            antiquant_offset_real, group_list_real, split_item_value, result);

        return y[0];
    }

    auto num_x = x.size();
    bool singleWeight = weight.size() == 1 && weight[0].sizes().size() == 3;
    auto num_weight = singleWeight ? static_cast<size_t>(weight[0].size(0)) : static_cast<size_t>(weight.size());
    auto group_list_real = group_list.value_or(at::Tensor());
    auto num_group_list = group_list_real.size(0);
    int64_t split_item_value = split_item.value_or(0);
    check_dims(split_item_value, num_x, num_weight, num_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(output_dtype.has_value()
            ? npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(output_dtype.value()))
            : x[0].scalar_type());

    size_t dim_num_w = weight[0].sizes().size();
    size_t n0 = static_cast<size_t>(weight[0].size(dim_num_w - 1));
    // weight is trans or not
    bool weight_trans = is_weight_trans(weight[0]);
#if VERSION_BETWEEN(V2R1, V2R7)
    bool mxfp4_valid = x_dtype.has_value() && weight_dtype.has_value() &&
        (x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
            x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2)) &&
        (weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2) ||
            weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1));
#endif
#if VERSION_BETWEEN(V2R8, VERSION_NEWEST)
    bool mxfp4_valid = false;
    if (x_dtype.has_value()) {
        mxfp4_valid = (x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
            x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    } else {
        mxfp4_valid = x[0].scalar_type() == at::ScalarType::Float4_e2m1fn_x2;
    }
    if (weight_dtype.has_value()) {
        mxfp4_valid = mxfp4_valid &&
            (weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    } else {
        mxfp4_valid = mxfp4_valid && weight[0].scalar_type() == at::ScalarType::Float4_e2m1fn_x2;
    }
#endif
    size_t n_new = (mxfp4_valid && !weight_trans) ? (n0 * FP4_IN_INT8) : n0;
    if (mxfp4_valid) {
        TORCH_CHECK(x[0].size(1) != 1, "In mxfp4, dim K should not be 2.", OPS_ERROR(ErrCode::VALUE));
    }
    if (split_item_value == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item_value == IN_SPLIT_OUT_NOT_SPLIT) {
        if (num_group_list > 0) {
            y.reserve(num_group_list);
            int64_t glr_value_0 = group_list_real[0].item<int64_t>();
            TORCH_CHECK(glr_value_0 >= 0, "group_list[0] should be larger than or equal to 0, but now is ", glr_value_0,
                "." + OPS_ERROR(ErrCode::VALUE));
            create_new_tensor(y, glr_value_0, n0, options);
            int64_t glr_value_pre = glr_value_0;
            for (int i = 1; i < num_group_list; i++) {
                int64_t glr_value_cur = group_list_real[i].item<int64_t>();
                TORCH_CHECK(glr_value_cur - glr_value_pre >= 0, "group_list[", i, "] - group_list[", i - 1,
                    "] should be larger than or equal to 0, but now is ", glr_value_cur - glr_value_pre,
                    "." + OPS_ERROR(ErrCode::VALUE));
                size_t ni = singleWeight ? n0 : weight[i].size(dim_num_w - 1);
                create_new_tensor(y, glr_value_cur - glr_value_pre, ni, options);
                glr_value_pre = glr_value_cur;
            }
        } else {
            y.reserve(num_x);
            for (size_t i = 0; i < num_x; i++) {
                size_t ni = singleWeight ? n0 : weight[i].size(dim_num_w - 1);
                create_new_tensor_multi_dim(y, x[i], ni, options);
            }
        } // 校验NO_SPLIT时为特殊场景（groupList为空）或num_x > 1
    } else if (split_item_value == IN_NOT_SPLIT_OUT_SPLIT || split_item_value == IN_SPLIT_OUT_SPLIT) {
        if (num_x > 1) {
            size_t dim_m = 0;
            for (size_t i = 0; i < num_x; i++) {
                dim_m += static_cast<size_t>(x[i].size(0));
            }
            weight[0].dtype() == at::ScalarType::Int ? create_new_tensor(y, dim_m, n0 * INT4_NUMS_IN_INT32, options)
                                                     : create_new_tensor(y, dim_m, n_new, options);
        } else if (num_x == 1) {
            if (group_type_value == K_SPLIT) {
                TORCH_CHECK(num_weight == 1,
                    "When group_list is 2(K_SPLIT) and split_item is 2/3, the length of weight must equal x.");
                weight[0].dtype() == at::ScalarType::Int
                    ? create_new_tensor_batch(y, num_group_list, x[0].size(0), n0 * INT4_NUMS_IN_INT32, options)
                    : create_new_tensor_batch(y, num_group_list, x[0].size(0), n_new, options);
            } else {
                (weight[0].dtype() == at::ScalarType::Int ||
                    (weight[0].dtype() == at::ScalarType::Float && weight[0].dtype() != x[0].dtype())) &&
                        (!weight_trans)
                    ? create_new_tensor(y, x[0].size(0), n0 * INT4_NUMS_IN_INT32, options)
                    : create_new_tensor(y, x[0].size(0), n_new, options);
            }
        }
    }
    at::TensorList result = at::TensorList(y);

    auto bias_real = bias_tl.value_or(at::TensorList());
    auto scale_real = scale.value_or(at::TensorList());
    auto offset_real = offset_tl.value_or(at::TensorList());
    auto antiquant_scale_real = antiquant_scale_tl.value_or(at::TensorList());
    auto antiquant_offset_real = antiquant_offset_tl.value_or(at::TensorList());
    auto per_token_scale_real = per_token_scale.value_or(at::TensorList());
    auto activation_input_real = activation_input_tl.value_or(at::TensorList());
    auto activation_quant_scale_real = activation_quant_scale_tl.value_or(at::TensorList());
    auto activation_quant_offset_real = activation_quant_offset_tl.value_or(at::TensorList());
    auto act_out = at::TensorList();
    auto dynamic_quant_scale_out = at::TensorList();
    int64_t group_list_type_value = group_list_type.value_or(0);
    int64_t act_type_value = act_type.value_or(0);
    auto tuning_config_real = tuning_config.value_or(at::IntArrayRef{});

    TensorListWrapper x_wrapper = {x,
        x_dtype.has_value() ? c10_npu::GetAclDataType(x_dtype.value())
                            : npu_preparation::convert_to_acl_data_type(x[0].scalar_type())};
    TensorListWrapper weight_wrapper = {weight,
        weight_dtype.has_value() ? c10_npu::GetAclDataType(weight_dtype.value())
                                 : npu_preparation::convert_to_acl_data_type(weight[0].scalar_type())};
    TensorListWrapper scale_wrapper = {scale_real,
        scale_dtype.has_value()
            ? c10_npu::GetAclDataType(scale_dtype.value())
            : (scale_real.empty() ? aclDataType::ACL_UINT64
                                  : npu_preparation::convert_to_acl_data_type(scale_real[0].scalar_type()))};
    TensorListWrapper per_token_scale_wrapper = {per_token_scale_real,
        per_token_scale_dtype.has_value()
            ? c10_npu::GetAclDataType(per_token_scale_dtype.value())
            : (per_token_scale_real.empty()
                      ? aclDataType::ACL_FLOAT
                      : npu_preparation::convert_to_acl_data_type(per_token_scale_real[0].scalar_type()))};
    TensorListWrapper antiquant_scale_wrapper = {antiquant_scale_real,
        antiquant_scale_real.empty()
            ? aclDataType::ACL_FLOAT16
            : (antiquant_scale_real[0].scalar_type() == at::ScalarType::Byte
                      ? aclDataType::ACL_FLOAT8_E8M0
                      : npu_preparation::convert_to_acl_data_type(antiquant_scale_real[0].scalar_type()))};

    int64_t weight_format = at_npu::native::custom_ops::get_npu_format(weight[0]);
    const bool is_weight_nz = (weight_format == ACL_FORMAT_FRACTAL_NZ) ||
        (weight_format == ACL_FORMAT_FRACTAL_NZ_C0_2) || (weight_format == ACL_FORMAT_FRACTAL_NZ_C0_4) ||
        (weight_format == ACL_FORMAT_FRACTAL_NZ_C0_16);
    if (is_weight_nz) {
        static const bool is_weight_nz_available = check_aclnn_kernel_available("aclnnGroupedMatmulWeightNz");
        TORCH_CHECK(is_weight_nz_available,
            "Format of weight in npu_grouped_matmul is FRACTAL_NZ, current CANN version "
            "do not support with this format. Please try to update the version of CANN." +
                OPS_ERROR(ErrCode::PARAM));
        int64_t quant_per_group_size = 0;
        EXEC_NPU_CMD(aclnnGroupedMatmulWeightNz, x_wrapper, weight_wrapper, bias_real, scale_wrapper, offset_real,
            antiquant_scale_wrapper, antiquant_offset_real, per_token_scale_wrapper, group_list_real,
            activation_input_real, activation_quant_scale_real, activation_quant_offset_real, split_item_value,
            group_type_value, group_list_type_value, act_type_value, tuning_config_real, quant_per_group_size, result,
            act_out, dynamic_quant_scale_out);
        return y[0];
    }
    static const bool is_grouped_matmul_V5_available = check_aclnn_kernel_available("aclnnGroupedMatmulV5");
    static const bool dtypeValid = x[0].scalar_type() != at::ScalarType::Float8_e5m2 &&
        x[0].scalar_type() != at::ScalarType::Float8_e4m3fn && !x_dtype.has_value() && !weight_dtype.has_value();
    if (!is_grouped_matmul_V5_available || !dtypeValid || mxfp4_valid) {
        EXEC_NPU_CMD(aclnnGroupedMatmulV4, x_wrapper, weight_wrapper, bias_real, scale_wrapper, offset_real,
            antiquant_scale_real, antiquant_offset_real, per_token_scale_wrapper, group_list_real,
            activation_input_real, activation_quant_scale_real, activation_quant_offset_real, split_item_value,
            group_type_value, group_list_type_value, act_type_value, result, act_out, dynamic_quant_scale_out);
    } else {
        EXEC_NPU_CMD(aclnnGroupedMatmulV5, x_wrapper, weight_wrapper, bias_real, scale_wrapper, offset_real,
            antiquant_scale_real, antiquant_offset_real, per_token_scale_wrapper, group_list_real,
            activation_input_real, activation_quant_scale_real, activation_quant_offset_real, split_item_value,
            group_type_value, group_list_type_value, act_type_value, tuning_config_real, result, act_out,
            dynamic_quant_scale_out);
    }
    return y[0];
}
}