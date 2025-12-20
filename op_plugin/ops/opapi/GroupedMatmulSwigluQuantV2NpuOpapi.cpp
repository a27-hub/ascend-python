// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const static int64_t MXFP_DIVISOR_SIZE = 64;
const static int64_t MXFP_MULTI_BASE_SIZE = 2;
const static int64_t DIM_2 = 2;
constexpr int64_t WEIGHT_MAX_DIM_NUM = 3;
constexpr int64_t WEIGHT_PENULTIMATE_DIM = 1;

// const std::string GmmSwigluTypeToString(int64_t input_type)
// {
//     return c10_npu::IsCustomDType(input_type) ?
//                c10_npu::CustomDataTypeToString(input_type) : c10::toString(static_cast<at::ScalarType>(input_type));
// }

void create_new_tensor(at::Tensor &y, size_t dim_m, size_t dim_n, c10::TensorOptions options)
{
    auto output_size = op_infer::array_to_small_vector({dim_m, dim_n});
    y = npu_preparation::apply_tensor_without_format(output_size, options);
}

void create_new_tensor_batch(at::Tensor &y, size_t batch, size_t dim_m, size_t dim_n,
                             const c10::TensorOptions &options)
{
    auto output_size = op_infer::array_to_small_vector({batch, dim_m, dim_n});
    y = npu_preparation::apply_tensor_without_format(output_size, options);
}



std::tuple<at::Tensor, at::Tensor> npu_grouped_matmul_swiglu_quant_v2(
    const at::Tensor & x,
    const at::TensorList weight,
    const at::TensorList weight_scale,
    const at::Tensor & x_scale,
    const at::Tensor & group_list,
    const c10::optional<at::Tensor> & smooth_scale,
    const c10::optional<at::TensorList> weight_assist_matrix,
    const c10::optional<at::Tensor> & bias,
    c10::optional<int64_t> dequant_mode,
    c10::optional<int64_t> dequant_dtype,
    c10::optional<int64_t> quant_mode,
    c10::optional<int64_t> quant_dtype,
    c10::optional<int64_t> group_list_type,
    const c10::OptionalIntArrayRef tuning_config,
    c10::optional<int64_t> x_dtype,
    c10::optional<int64_t> weight_dtype,
    c10::optional<int64_t> weight_scale_dtype,
    c10::optional<int64_t> x_scale_dtype)
{
    TORCH_CHECK(x.dim() >= DIM_2, "x dim should greater than 2, but the actual value is ", x.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!weight_scale[0].sizes().empty(), "weight_scale[0] is empty", OPS_ERROR(ErrCode::PARAM));
    auto x_size = x.sizes();
    int n = weight[0].sizes()[2];
    int m = x_size[0];
    int k = x_size[1];

    if (x_dtype.has_value()) {
        TORCH_CHECK(x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2)
                 || x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1),
                    "The optional parameter x_dtype only supports mxfp4 or None, but now is ",
                    c10_npu::CustomDataTypeToString(x_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }

    if (weight_dtype.has_value()) {
        TORCH_CHECK(weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2)
                 || weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1),
                    "The optional parameter weight_dtype only supports mxfp4 or None, but now is ",
                    c10_npu::CustomDataTypeToString(weight_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }

    // TORCH_CHECK(!(x_dtype.has_value() || weight_dtype.has_value()),
    //             "The optional parameter x_dtype and weight_dtype should be null", "." + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK((x_dtype.has_value() && weight_dtype.has_value()) || (!x_dtype.has_value() && !weight_dtype.has_value()),
                    "The optional parameter x_dtype and weight_dtype should both be mxfp4 or None, but now x_dtype is ",
                    c10_npu::CustomDataTypeToString(x_dtype.value()), ", weight_dtype is ", 
                    c10_npu::CustomDataTypeToString(weight_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));

    if (weight_scale_dtype.has_value()) {
        TORCH_CHECK(weight_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                    "The optional parameter weight_scale_dtype only supports float8_e8m0fnu or None, but now is ",
                    c10_npu::CustomDataTypeToString(weight_scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }

    if (x_scale_dtype.has_value()) {
        TORCH_CHECK(x_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                    "The optional parameter x_scale_dtype only supports float8_e8m0fnu or None, but now is ",
                    c10_npu::CustomDataTypeToString(x_scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }

    int64_t dequant_mode_real = dequant_mode.value_or(0); // todo 改为默认值2
    int64_t dequant_dtype_real = dequant_dtype.value_or(0);
    int64_t quant_mode_real = quant_mode.value_or(0);
    int64_t group_list_type_real = group_list_type.value_or(0);
    auto weight_assist_matrix_real = weight_assist_matrix.value_or(at::TensorList());
    auto tuning_config_real = tuning_config.value_or(at::IntArrayRef{});
    auto bias_real = bias.value_or(at::Tensor());
    auto smooth_scale_real = smooth_scale.value_or(at::Tensor());

    // TensorListWrapper weight_scale_wrapper = make_wrapper(weight_scale, weight_scale_dtype);
    // TensorWrapper x_scale_wrapper = make_wrapper(x_scale, x_scale_dtype);

    // infer weight is trans or not, when wight is trans, weight_strides[1] = 1
    c10::SmallVector<int64_t, WEIGHT_MAX_DIM_NUM> weight_strides = op_infer::array_to_small_vector(weight[0].strides());
    bool weight_trans = (weight_strides[WEIGHT_PENULTIMATE_DIM] == 1);
    static const bool mxfp4_valid = x_dtype.has_value() && weight_dtype.has_value() &&
                                   (x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2) ||
                                    x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1)) &&
                                   (weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2) ||
                                    weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1));
    at::Tensor output;
    at::Tensor output_scale;
    if (!weight_scale_dtype.has_value()) {
        output = npu_preparation::apply_tensor_without_format({m, n/ MXFP_MULTI_BASE_SIZE}, c10::dtype(c10::ScalarType::Char));
        output_scale = npu_preparation::apply_tensor_without_format({m}, c10::dtype(c10::ScalarType::Float));
    } else {
        c10::TensorOptions options_output = x.options().dtype(quant_dtype.has_value()
                    ? npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(quant_dtype.value()))
                    : x[0].scalar_type());
        c10::TensorOptions options = x.options().dtype(
            npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(weight_scale_dtype.value())));
        // 原有
        // create_new_tensor(output, m, n / MXFP_MULTI_BASE_SIZE, options_output);
        // create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE, options);
    
        auto acl_data_type = c10_npu::GetAclDataType(quant_dtype.value());
        auto scalar_type = npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(quant_dtype.value()));
        // std::cout << "! [op-plugin] after infer y_options  " << std::endl;
        // std::cout << "! [op-plugin]  quant_dtype(y_options), acl_data_type: " << acl_data_type << ", scalar_type: " << scalar_type << std::endl;
   

        // std::cout << "!  [op-plugin]  x.shape: " << x.sizes();
        // std::cout << ",  [op-plugin] x_dtype: " << c10_npu::GetAclDataType(x_dtype.value()) << std::endl;
        // std::cout << "!  [op-plugin] weight.shape: " << weight[0].sizes();
        // std::cout << ",  [op-plugin] weight_dtype: " << c10_npu::GetAclDataType(weight_dtype.value()) << std::endl;
        // std::cout << "!  [op-plugin] xScale.shape: " << x_scale.sizes();
        // std::cout << ",  [op-plugin] xScale_dtype: " << c10_npu::GetAclDataType(x_scale_dtype.value()) << std::endl;
        // std::cout << "!  [op-plugin] weightScale.shape: " << weight_scale[0].sizes();
        // std::cout << ",  [op-plugin] weightScale_dtype: " << c10_npu::GetAclDataType(weight_scale_dtype.value()) << std::endl;


        if (mxfp4_valid) { // 输入是mxfp4
            if (!weight_trans) { // 非转置
                if (c10_npu::GetAclDataType(quant_dtype.value()) == 35 || c10_npu::GetAclDataType(quant_dtype.value()) == 36) { // 输出是mxfp8，扩维
                    size_t output_n_new = ((n / MXFP_MULTI_BASE_SIZE) * FP4_IN_INT8);
                    size_t output_scale_n_new = (op_infer::CeilDiv(n  * FP4_IN_INT8 / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE));
                    printf("! input is mxfp4, !weight_trans, y is mxfp8\n");

                    TORCH_CHECK(m >= 1, "m must be positive.");
                    TORCH_CHECK(output_n_new >= 1, "output_n_new must be positive.");
                    TORCH_CHECK(output_scale_n_new >= 1, "output_scale_n_new must be positive.");
                    
                    // 扩维
                    create_new_tensor(output, m, output_n_new, options_output);
                    create_new_tensor_batch(output_scale, m, output_scale_n_new, MXFP_MULTI_BASE_SIZE, options);
                } else { // 输出是mxfp4 y不扩维
                    printf("! input is mxfp4, !weight_trans, y is mxfp4 \n");
                    create_new_tensor(output, m, n / MXFP_MULTI_BASE_SIZE, options_output);
                    create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n * FP4_IN_INT8 / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE, options);
                }
            } else { // 转置
                // 输出是mxfp8，不扩维
                if (c10_npu::GetAclDataType(quant_dtype.value()) == 35 || c10_npu::GetAclDataType(quant_dtype.value()) == 36) {
                        printf("! input is mxfp4, weight_trans, y is mxfp8 \n");
                    create_new_tensor(output, m, n / MXFP_MULTI_BASE_SIZE, options_output);
                    create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE, options);
                } else { // 输出是mxfp4
                    printf("! 1651 input is mxfp4, weight_trans, y is mxfp4 \n");
                    create_new_tensor(output, m, n / MXFP_MULTI_BASE_SIZE / 2, options_output);
                    create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE, options);
                }
            }
        } else { // 输入是mxfp8
            printf("!  [op-plugin] input is mxfp8\n");
            create_new_tensor(output, m, n / MXFP_MULTI_BASE_SIZE, options_output);
            create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE, options);
        }
        std::cout << "!  [op-plugin] infer  y.shape: " << output.sizes() << std::endl;
        std::cout << "!  [op-plugin] infer y_scale.shape: " << output_scale.sizes() << std::endl;           
    }

    // todo use make_wrapper
    TensorWrapper x_wrapper = {x,
        x_dtype.has_value() ? c10_npu::GetAclDataType(x_dtype.value())
                            : npu_preparation::convert_to_acl_data_type(x.scalar_type())};
    TensorListWrapper weight_wrapper = {weight,
        weight_dtype.has_value() ? c10_npu::GetAclDataType(weight_dtype.value())
                                 : npu_preparation::convert_to_acl_data_type(weight[0].scalar_type())};
    TensorListWrapper weight_scale_wrapper = {weight_scale,
        weight_scale_dtype.has_value() ? c10_npu::GetAclDataType(weight_scale_dtype.value())
                                : (weight_scale.empty() ? aclDataType::ACL_FLOAT
                                : npu_preparation::convert_to_acl_data_type(weight_scale[0].scalar_type()))};
    TensorWrapper x_scale_wrapper = {x_scale,
        x_scale_dtype.has_value() ? c10_npu::GetAclDataType(x_scale_dtype.value())
                                : (!x_scale.numel() ? aclDataType::ACL_FLOAT
                                : npu_preparation::convert_to_acl_data_type(x_scale.scalar_type()))};
    TensorWrapper output_wrapper = {output,
        quant_dtype.has_value() ? c10_npu::GetAclDataType(quant_dtype.value()): aclDataType::ACL_FLOAT}; // todo
    TensorWrapper output_scale_wrapper = {output_scale,
        weight_scale_dtype.has_value() ? aclDataType::ACL_FLOAT8_E8M0 : aclDataType::ACL_FLOAT};
    // todo check make_wrapper
    // TensorWrapper x_wrapper = make_wrapper(x, x_dtype);
    // TensorListWrapper weight_wrapper = make_wrapper(weight, weight_dtype);
    // TensorListWrapper weight_scale_wrapper = make_wrapper(weight_scale, weight_scale_dtype);
    // TensorWrapper x_scale_wrapper = make_wrapper(x_scale, x_scale_dtype);
    // TensorWrapper output_wrapper = make_wrapper(output, quant_dtype);
    // TensorWrapper output_scale_wrapper = make_wrapper(output_scale, aclDataType::ACL_FLOAT8_E8M0);

    const bool is_weight_nz = at_npu::native::custom_ops::get_npu_format(weight[0]) == ACL_FORMAT_FRACTAL_NZ;
    if (is_weight_nz) {
        static const bool is_weight_nz_available = check_aclnn_kernel_available("aclnnGroupedMatmulSwigluQuantWeightNzV2");
        TORCH_CHECK(is_weight_nz_available,
                    "Format of weight in npu_grouped_matmul is FRACTAL_NZ, current CANN version "
                    "do not support with this format. Please try to update the version of CANN."
                    + OPS_ERROR(ErrCode::PARAM));
        EXEC_NPU_CMD(
            aclnnGroupedMatmulSwigluQuantWeightNzV2,
            x,
            weight,
            weight_scale_wrapper,
            weight_assist_matrix_real,
            bias_real,
            x_scale_wrapper,
            smooth_scale_real,
            group_list,
            dequant_mode_real,
            dequant_dtype_real,
            quant_mode_real,
            group_list_type_real,
            tuning_config_real,
            output,
            output_scale_wrapper);
    } else {
        static const bool mxfp8_valid = (x.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                                        x.scalar_type() == at::ScalarType::Float8_e5m2) &&
                                        (weight[0].scalar_type() == at::ScalarType::Float8_e5m2 ||
                                        weight[0].scalar_type() == at::ScalarType::Float8_e4m3fn);
        TORCH_CHECK(mxfp8_valid || mxfp4_valid,
                    "The dtype of x and weight only supports Float8_e4m3fn/Float8_e5m2/Float4_e2m1/Float4_e1m2, but now x_dtype is ",
                    x.scalar_type(), ", weight_dtype is ", weight[0].scalar_type(),
                    "." + OPS_ERROR(ErrCode::VALUE));
        printf("! [op-plugin] before exec EXEC_NPU_CMD.\n");
        EXEC_NPU_CMD(
            aclnnGroupedMatmulSwigluQuantV2,
            x_wrapper,
            weight_wrapper,
            weight_scale_wrapper,
            weight_assist_matrix_real,
            bias_real,
            x_scale_wrapper,
            smooth_scale_real,
            group_list,
            dequant_mode_real,
            dequant_dtype_real,
            quant_mode_real,
            group_list_type_real,
            tuning_config_real,
            output_wrapper,
            output_scale_wrapper);
        printf("! [op-plugin] after exec EXEC_NPU_CMD.\n");

    }
    std::cout << "! [op-plugin] afer aclnn  y.shape: " << output.sizes() << std::endl;
    std::cout << "! [op-plugin] afer aclnn y_scale.shape: " << output_scale.sizes() << std::endl;
    return std::tuple<at::Tensor, at::Tensor>(output, output_scale);
}
}
