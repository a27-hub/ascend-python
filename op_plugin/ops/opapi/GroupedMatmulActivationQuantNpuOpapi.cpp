// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <string>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
constexpr int64_t DIM_0 = 0LL;
constexpr int64_t DIM_1 = 1LL;
constexpr int64_t DIM_2 = 2LL;
constexpr int64_t DIM_3 = 3LL;
constexpr int64_t DIM_4 = 4LL;
constexpr int64_t DIM_5 = 5LL;
constexpr int64_t NUM_ONE = 1LL;
constexpr int64_t MXFP_DIVISOR_SIZE = 64LL;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2LL;
constexpr int64_t FLOAT8_E5M2 = 35LL;
constexpr int64_t FLOAT8_E4M3FN = 36LL;
constexpr int64_t DEFAULT_DEQUANT_DTYPE = 0LL;
constexpr int64_t SCALE_ALG_OCP = 0LL;
constexpr int64_t SCALE_ALG_CUBLAS = 1LL;
constexpr double DEFAULT_DST_TYPE_MAX = 0.0;
constexpr double FLOAT_EPS = 1e-6;
constexpr const char *GELU_TANH = "gelu_tanh";
constexpr const char *ROUND_MODE_RINT = "rint";
} // namespace

bool is_fp8_tensor(const at::Tensor &tensor)
{
    return tensor.scalar_type() == at::ScalarType::Float8_e4m3fn ||
           tensor.scalar_type() == at::ScalarType::Float8_e5m2;
}

aclDataType get_acl_fp8_dtype(int64_t dtype)
{
    return dtype == FLOAT8_E5M2 ? aclDataType::ACL_FLOAT8_E5M2 : aclDataType::ACL_FLOAT8_E4M3FN;
}

at::ScalarType get_scalar_fp8_dtype(int64_t dtype)
{
    return dtype == FLOAT8_E5M2 ? at::ScalarType::Float8_e5m2 : at::ScalarType::Float8_e4m3fn;
}

int64_t infer_logical_n(const at::Tensor &weight_scale, bool transpose_weight)
{
    TORCH_CHECK(weight_scale.dim() == DIM_4,
                "The dim of weight_scale[0] should be equal to 4 in MX quant mode, but got ",
                weight_scale.dim(), OPS_ERROR(ErrCode::PARAM));
    return transpose_weight ? weight_scale.size(DIM_1) : weight_scale.size(DIM_2);
}

void check_quant_attrs(c10::string_view activation_type, int64_t dequant_dtype, int64_t quant_dtype,
                       c10::string_view round_mode, int64_t scale_alg, double dst_type_max)
{
    TORCH_CHECK(std::string(activation_type) == GELU_TANH,
                "The activation_type only supports gelu_tanh for now, but got ",
                activation_type, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(dequant_dtype == DEFAULT_DEQUANT_DTYPE,
                "The dequant_dtype must be 0 in MX quant mode, but got ",
                dequant_dtype, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(quant_dtype == FLOAT8_E4M3FN || quant_dtype == FLOAT8_E5M2,
                "The quant_dtype only supports 36(float8_e4m3fn) or 35(float8_e5m2), but got ",
                quant_dtype, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(std::string(round_mode) == ROUND_MODE_RINT,
                "The round_mode only supports rint in MX quant mode, but got ",
                round_mode, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(scale_alg == SCALE_ALG_OCP || scale_alg == SCALE_ALG_CUBLAS,
                "The scale_alg only supports 0 or 1 in MX quant mode, but got ",
                scale_alg, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(std::abs(dst_type_max - DEFAULT_DST_TYPE_MAX) <= FLOAT_EPS,
                "The dst_type_max only supports 0.0 in current MX quant mode, but got ",
                dst_type_max, OPS_ERROR(ErrCode::VALUE));
}

void check_inputs(const at::Tensor &x, const at::Tensor &group_list, const at::Tensor &x_scale,
                  const at::TensorList weight, const at::TensorList weight_scale,
                  const c10::optional<at::TensorList> &bias)
{
    TORCH_CHECK(weight.size() == NUM_ONE, "The size of weight should be 1, current size is ",
                weight.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight_scale.size() == NUM_ONE, "The size of weight_scale should be 1, current size is ",
                weight_scale.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.dim() == DIM_2, "The dim of x should be equal to 2, but got ",
                x.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x_scale.dim() == DIM_3, "The dim of x_scale should be equal to 3, but got ",
                x_scale.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(group_list.dim() == DIM_1, "The dim of group_list should be equal to 1, but got ",
                group_list.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(is_fp8_tensor(x), "The x only supports torch.float8_e4m3fn or torch.float8_e5m2, but got ",
                x.scalar_type(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(is_fp8_tensor(weight[DIM_0]),
                "The weight only supports torch.float8_e4m3fn or torch.float8_e5m2, but got ",
                weight[DIM_0].scalar_type(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(x_scale.scalar_type() == at::ScalarType::Byte,
                "The x_scale should use uint8 storage for float8_e8m0fnu, but got ",
                x_scale.scalar_type(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(weight_scale[DIM_0].scalar_type() == at::ScalarType::Byte,
                "The weight_scale should use uint8 storage for float8_e8m0fnu, but got ",
                weight_scale[DIM_0].scalar_type(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(!bias.has_value() || bias.value().empty(),
                "Non-empty bias is not supported in current MX quant mode.", OPS_ERROR(ErrCode::PARAM));
}

at::TensorList prepare_weight_list(const at::TensorList weight, c10::SmallVector<at::Tensor, 1> &weight_vec)
{
    at::Tensor weight_for_nz = weight[DIM_0];
    const bool is_weight_nz = at_npu::native::custom_ops::get_npu_format(weight_for_nz) == ACL_FORMAT_FRACTAL_NZ ||
                              at_npu::native::custom_ops::get_npu_format(weight_for_nz) == ACL_FORMAT_FRACTAL_NZ_C0_16 ||
                              weight_for_nz.dim() == DIM_5;
    if (is_weight_nz && at_npu::native::custom_ops::get_npu_format(weight_for_nz) != ACL_FORMAT_FRACTAL_NZ) {
        weight_for_nz = weight_for_nz.clone();
        auto &desc = torch_npu::NPUBridge::GetNpuStorageImpl(weight_for_nz)->npu_desc_;
        desc.npu_format_ = ACL_FORMAT_FRACTAL_NZ;
        desc.storage_sizes_ = op_infer::array_to_small_vector(weight_for_nz.sizes());
    }
    weight_vec.clear();
    weight_vec.emplace_back(weight_for_nz);
    return at::TensorList(weight_vec);
}

std::tuple<at::Tensor, at::Tensor> npu_grouped_matmul_activation_quant(
    const at::Tensor &x,
    const at::Tensor &group_list,
    const at::Tensor &x_scale,
    const at::TensorList weight,
    const at::TensorList weight_scale,
    c10::string_view activation_type,
    const c10::optional<at::TensorList> bias,
    c10::optional<int64_t> dequant_dtype,
    bool transpose_weight,
    c10::optional<int64_t> group_list_type,
    const c10::OptionalIntArrayRef tuning_config,
    c10::optional<int64_t> quant_dtype,
    c10::string_view round_mode,
    c10::optional<int64_t> scale_alg,
    double dst_type_max)
{
    int64_t dequant_dtype_real = dequant_dtype.value_or(DEFAULT_DEQUANT_DTYPE);
    int64_t quant_dtype_real = quant_dtype.value_or(FLOAT8_E4M3FN);
    int64_t group_list_type_real = group_list_type.value_or(0);
    int64_t scale_alg_real = scale_alg.value_or(SCALE_ALG_OCP);
    check_quant_attrs(activation_type, dequant_dtype_real, quant_dtype_real, round_mode,
                      scale_alg_real, dst_type_max);
    TORCH_CHECK(group_list_type_real == 0 || group_list_type_real == 1,
                "The group_list_type only supports 0 or 1, but got ",
                group_list_type_real, OPS_ERROR(ErrCode::VALUE));
    check_inputs(x, group_list, x_scale, weight, weight_scale, bias);

    int64_t m = x.size(DIM_0);
    int64_t n = infer_logical_n(weight_scale[DIM_0], transpose_weight);
    c10::TensorOptions output_options = x.options().dtype(get_scalar_fp8_dtype(quant_dtype_real));
    at::Tensor output = npu_preparation::apply_tensor_without_format({m, n}, output_options);
    c10::TensorOptions scale_options =
        x.options().dtype(npu_preparation::convert_to_scalar_type(aclDataType::ACL_FLOAT8_E8M0));
    at::Tensor output_scale = npu_preparation::apply_tensor_without_format(
        {m, op_infer::CeilDiv(n, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE}, scale_options);

    c10::SmallVector<at::Tensor, 1> weight_vec;
    at::TensorList weight_real = prepare_weight_list(weight, weight_vec);
    auto bias_real = bias.value_or(at::TensorList());
    auto tuning_config_real = tuning_config.value_or(at::IntArrayRef{});

    TensorWrapper x_wrapper = {x, npu_preparation::convert_to_acl_data_type(x.scalar_type())};
    TensorWrapper x_scale_wrapper = {x_scale, aclDataType::ACL_FLOAT8_E8M0};
    TensorListWrapper weight_wrapper = {weight_real,
        npu_preparation::convert_to_acl_data_type(weight[DIM_0].scalar_type())};
    TensorListWrapper weight_scale_wrapper = {weight_scale, aclDataType::ACL_FLOAT8_E8M0};
    TensorWrapper output_wrapper = {output, get_acl_fp8_dtype(quant_dtype_real)};
    TensorWrapper output_scale_wrapper = {output_scale, aclDataType::ACL_FLOAT8_E8M0};
    std::string activation_type_str = std::string(activation_type);
    std::string round_mode_str = std::string(round_mode);

    EXEC_NPU_CMD(
        aclnnGroupedMatmulActivationQuant,
        x_wrapper,
        group_list,
        x_scale_wrapper,
        weight_wrapper,
        weight_scale_wrapper,
        bias_real,
        activation_type_str.c_str(),
        dequant_dtype_real,
        transpose_weight,
        group_list_type_real,
        tuning_config_real,
        quant_dtype_real,
        round_mode_str.c_str(),
        scale_alg_real,
        dst_type_max,
        output_wrapper,
        output_scale_wrapper);

    return std::tuple<at::Tensor, at::Tensor>(output, output_scale);
}
} // namespace op_api
