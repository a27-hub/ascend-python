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

#include <vector>
#include <cstring>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {

// 常量定义
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr int64_t INT4_NUMS_IN_INT32 = 8;
constexpr size_t FUSED_TYPE_ARRAY_SIZE = 100;
using npu_preparation = at_npu::native::OpPreparation;

// ==================== 辅助函数 ====================

// 检查权重是否为 NZ 格式
static bool is_nz_format(const at::Tensor& x2)
{
    const torch_npu::NPUStorageDesc &tensor_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ ||
           tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_4;
}

// 检查是否为最后两维转置
static bool is_transpose_last_two_dims(const at::Tensor &tensor)
{
    if (tensor.dim() < 2 || tensor.dim() > 6) {
        return false;
    }
    int64_t dim1 = tensor.dim() - 1;
    int64_t dim2 = tensor.dim() - 2;
    if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2)) {
        int64_t tmpNxD = tensor.size(dim1) * tensor.size(dim2);
        for (int64_t batchDim = tensor.dim() - 3; batchDim >= 0; batchDim--) {
            if (tensor.stride(batchDim) != tmpNxD) {
                return false;
            }
            tmpNxD *= tensor.size(batchDim);
        }
        if (tensor.size(dim1) == 1 && tensor.size(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

// 推导输出 batch shape
static uint64_t infer_out_batch_shape_gelu(const at::Tensor &x1, const at::Tensor &x2, std::vector<uint64_t> &batch_record)
{
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(x2) || is_nz_format(x2),
                "x2 should be in the original format or nz format, but it is ",
                npu_preparation::get_tensor_npu_format(x2), OPS_ERROR(ErrCode::PARAM));
    uint64_t batch_val = 1;
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto out_dim_num = std::max(x1_dim_num, x2_dim_num);
    auto &shape_long = x1_dim_num > x2_dim_num ? x1 : x2;
    auto &shape_short = x1_dim_num > x2_dim_num ? x2 : x1;
    int64_t valid_offset = out_dim_num - std::min(x1_dim_num, x2_dim_num);
    for (int64_t i = 0; i < out_dim_num - LAST_SECOND_DIM_INDEX; i++) {
        auto short_dim = i < valid_offset ? 1 : shape_short.size(i - valid_offset);
        auto long_dim = shape_long.size(i);
        TORCH_CHECK(!(short_dim > 1 && long_dim > 1 && short_dim != long_dim),
                    "the x1 shape and x2 shape not supported for broadcast, the short_dim is ",
                    short_dim, " and the long_dim is ", long_dim, OPS_ERROR(ErrCode::PARAM));
        uint64_t cur_batch_value = static_cast<uint64_t>(std::max(short_dim, long_dim));
        batch_val = batch_val * cur_batch_value;
        batch_record.push_back(cur_batch_value);
    }
    return batch_val;
}

// ==================== 参数校验 ====================

// 校验输入参数合法性
static void validate_inputs(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &x1_scale,
    const at::Tensor &x2_scale,
    c10::string_view fused_op_type,
    bool transpose_x1,
    bool transpose_x2,
    const c10::optional<at::Tensor> &bias)
{

    // 1. 基础参数校验
    TORCH_CHECK(x1.defined(), "x1 must be defined", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2.defined(), "x2 must be defined", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1_scale.defined(), "x1_scale must be defined", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2_scale.defined(), "x2_scale must be defined", OPS_ERROR(ErrCode::PARAM));

    // 2. 数据类型校验
    TORCH_CHECK(x1.dtype() == at::kChar || x1.dtype() == at::kInt,
                "x1 must be INT8 or INT32, but got: ", x1.dtype(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(x2.dtype() == at::kChar || x2.dtype() == at::kInt,
                "x2 must be INT8 or INT32, but got: ", x2.dtype(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(x1_scale.dtype() == at::kFloat,
                "x1_scale must be FLOAT32, but got: ", x1_scale.dtype(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(x2_scale.dtype() == at::kFloat || x2_scale.dtype() == at::kBFloat16,
                "x2_scale must be FLOAT32 or BFLOAT16, but got: ", x2_scale.dtype(), OPS_ERROR(ErrCode::TYPE));

    // 3. 维度校验
    TORCH_CHECK(x1.dim() >= 2 && x1.dim() <= 6,
                "x1 shape dim num must be in [2, 6], but it is ", x1.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1_scale.dim() == 1,
                "x1_scale must be 1-D, but got: ", x1_scale.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2_scale.dim() == 1,
                "x2_scale must be 1-D, but got: ", x2_scale.dim(), OPS_ERROR(ErrCode::PARAM));

    // 4. fused_op_type 校验
    std::string fused_type_str(fused_op_type);
    TORCH_CHECK(fused_type_str == "gelu_tanh" || fused_type_str == "gelu_erf",
                "fused_op_type must be 'gelu_tanh' or 'gelu_erf', but got: ",
                fused_type_str, OPS_ERROR(ErrCode::PARAM));

    // 5. bias 校验
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->dtype() == at::kInt || bias->dtype() == at::kFloat ||
                    bias->dtype() == at::kBFloat16 || bias->dtype() == at::kHalf,
                    "bias must be INT32, FLOAT32, BFLOAT16, or FLOAT16, but got: ",
                    bias->dtype(), OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(bias->dim() == 1 || bias->dim() == 3,
                    "bias must be 1-D or 3-D, but got: ", bias->dim(), OPS_ERROR(ErrCode::PARAM));
    }

    // 6. 特殊约束校验（A4W4 + transpose_x1 + NZ格式）
    bool is_a4w4 = (x1.dtype() == at::kInt && x2.dtype() == at::kInt);
    bool is_a4w4_int = is_a4w4;
    
    // A4W4不支持NZ格式
    if (is_a4w4 && is_nz_format(x2)) {
        TORCH_CHECK(false,
                    "A4W4 quantization does not support NZ format, x2 must be in ND format",
                    OPS_ERROR(ErrCode::PARAM));
    }
    
    if (is_a4w4 && transpose_x1) {
        TORCH_CHECK(false,
                    "A4W4 quantization only supports transpose_x1=false, but got transpose_x1=true",
                    OPS_ERROR(ErrCode::PARAM));
    }

    // 7. 计算实际维度（考虑转置和INT4打包）
    // 7.1 计算x1的m和k维度
    int64_t x1_m_dim, x1_k_dim;
    if (transpose_x1) {
        x1_m_dim = x1.size(x1.dim() - 1);
        x1_k_dim = x1.size(x1.dim() - LAST_SECOND_DIM_INDEX);
    } else {
        x1_m_dim = x1.size(x1.dim() - LAST_SECOND_DIM_INDEX);
        x1_k_dim = x1.size(x1.dim() - 1);
    }
    
    // INT4打包处理：恢复x1的实际k维度
    if (is_a4w4_int && !transpose_x1) {
        TORCH_CHECK(x1_k_dim % INT4_NUMS_IN_INT32 == 0,
                    "A4W4 quantization requires k to be a multiple of 8, but got k=",
                    x1_k_dim, OPS_ERROR(ErrCode::PARAM));
        x1_k_dim = x1_k_dim * INT4_NUMS_IN_INT32;  // 恢复实际的k维度
    }

    // 7.2 计算x2的k和n维度（考虑转置）
    int64_t x2_k_dim, x2_n_dim;
    if (transpose_x2) {
        x2_n_dim = x2.size(x2.dim() - LAST_SECOND_DIM_INDEX);
        x2_k_dim = x2.size(x2.dim() - 1);
    } else {
        x2_k_dim = x2.size(x2.dim() - LAST_SECOND_DIM_INDEX);
        x2_n_dim = x2.size(x2.dim() - 1);
    }
    
    // INT4打包处理：恢复x2的实际k和n维度
    if (is_a4w4_int) {
        if (transpose_x2) {
            TORCH_CHECK(x2_k_dim % INT4_NUMS_IN_INT32 == 0,
                        "A4W4 quantization with transpose_x2=true requires k to be a multiple of 8, but got k=",
                        x2_k_dim, OPS_ERROR(ErrCode::PARAM));
            x2_k_dim = x2_k_dim * INT4_NUMS_IN_INT32;
        } else {
            TORCH_CHECK(x2_n_dim % INT4_NUMS_IN_INT32 == 0,
                        "A4W4 quantization with transpose_x2=false requires n to be a multiple of 8, but got n=",
                        x2_n_dim, OPS_ERROR(ErrCode::PARAM));
            x2_n_dim = x2_n_dim * INT4_NUMS_IN_INT32;
        }
    }

    // 7.3 验证x1和x2的k维度一致
    TORCH_CHECK(x1_k_dim == x2_k_dim,
                "x1's k dimension must match x2's k dimension, but got x1_k_dim=", x1_k_dim,
                " and x2_k_dim=", x2_k_dim, OPS_ERROR(ErrCode::PARAM));

    // 7.4 A4W4量化约束：内轴必须为偶数
    if (is_a4w4) {
        TORCH_CHECK(x1_k_dim % 2 == 0,
                    "A4W4 quantization requires x1's inner axis (k) to be even, but got k=",
                    x1_k_dim, OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(x2_k_dim % 2 == 0,
                    "A4W4 quantization requires x2's inner axis (k) to be even, but got k=",
                    x2_k_dim, OPS_ERROR(ErrCode::PARAM));
    }

    // 8. Scale shape 详细校验（per-token 和 per-channel）
    // 8.1 x1_scale 校验：per-token scale，shape应该是 (m,)
    TORCH_CHECK(x1_scale.size(0) == x1_m_dim,
                "x1_scale size must match x1's m dimension (per-token quantization), "
                "but got x1_scale.size(0)=", x1_scale.size(0), " and x1_m_dim=", x1_m_dim,
                OPS_ERROR(ErrCode::PARAM));

    // 8.2 x2_scale 校验：per-channel scale，shape应该是 (n,) 或 (1,)
    TORCH_CHECK(x2_scale.size(0) == x2_n_dim || x2_scale.size(0) == 1,
                "x2_scale size must match x2's n dimension or be 1 (per-channel quantization), "
                "but got x2_scale.size(0)=", x2_scale.size(0), " and x2_n_dim=", x2_n_dim,
                OPS_ERROR(ErrCode::PARAM));

    // 9. NZ 格式特殊校验（仅支持A8W8）
    if (is_nz_format(x2)) {
        // A4W4不支持NZ格式，已在前面检查
        TORCH_CHECK(!is_a4w4,
                    "A4W4 quantization does not support NZ format",
                    OPS_ERROR(ErrCode::PARAM));
        
        const torch_npu::NPUStorageDesc &tensor_desc =
            torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
        auto storage_sizes = tensor_desc.storage_sizes_;
        int64_t x2_storage_dim_num = static_cast<int64_t>(storage_sizes.size());
        TORCH_CHECK(x2_storage_dim_num >= 4 && x2_storage_dim_num <= 8,
                    "For NZ format, x2 storage shape dim num must be in [4, 8], but it is ",
                    x2_storage_dim_num, OPS_ERROR(ErrCode::PARAM));

        // NZ格式约束：k、n不能为1
        TORCH_CHECK(x2_k_dim != 1 && x2_n_dim != 1,
                    "When x2 is NZ format, k and n cannot be 1, but got k=", x2_k_dim,
                    " and n=", x2_n_dim, OPS_ERROR(ErrCode::PARAM));

        // 校验 k0 和 n0
        int64_t k0_expected = transpose_x2 ? 32 : 16;
        int64_t n0_expected = transpose_x2 ? 16 : 32;
        if (storage_sizes.size() >= 5) {
            int64_t k0_idx = transpose_x2 ? (storage_sizes.size() - 1) : (storage_sizes.size() - 2);
            int64_t n0_idx = transpose_x2 ? (storage_sizes.size() - 2) : (storage_sizes.size() - 1);
            TORCH_CHECK(storage_sizes[k0_idx] == k0_expected,
                        "For NZ format with transpose_x2=", transpose_x2,
                        ", k0 must be ", k0_expected, ", but got ", storage_sizes[k0_idx],
                        OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(storage_sizes[n0_idx] == n0_expected,
                        "For NZ format with transpose_x2=", transpose_x2,
                        ", n0 must be ", n0_expected, ", but got ", storage_sizes[n0_idx],
                        OPS_ERROR(ErrCode::PARAM));
        }

        // 校验 k1 是否满足 ceil(k / k0) = k1（A8W8场景）
        int64_t k1_storage_idx = transpose_x2 ? (storage_sizes.size() - 4) : (storage_sizes.size() - 3);
        int64_t k1_storage = static_cast<int64_t>(storage_sizes[k1_storage_idx]);
        int64_t k_expected_from_k1 = (x2_k_dim + k0_expected - 1) / k0_expected;
        TORCH_CHECK(k1_storage == k_expected_from_k1,
                    "For NZ format with transpose_x2=", transpose_x2,
                    ", x2 storage shape's k1 must satisfy ceil(k / ", k0_expected, ") = k1, "
                    "where k=", x2_k_dim, " (expected k1=", k_expected_from_k1,
                    "), but got k1=", k1_storage, OPS_ERROR(ErrCode::PARAM));
    }

    // 10. ND 格式下维度校验
    if (!is_nz_format(x2)) {
        // ND格式A4W4量化约束：仅支持2维
        if (is_a4w4) {
            TORCH_CHECK(x2.dim() == 2,
                        "A4W4 quantization with ND format only supports 2-D x2, but got dim=",
                        x2.dim(), OPS_ERROR(ErrCode::PARAM));
        } else {
            TORCH_CHECK(x2.dim() >= 2 && x2.dim() <= 6,
                        "For ND format, x2 shape dim num must be in [2, 6], but it is ",
                        x2.dim(), OPS_ERROR(ErrCode::PARAM));
        }
    }

    // 11. Bias 详细校验（根据量化类型）
    if (bias.has_value() && bias->defined()) {
        const at::Tensor &bias_real = bias.value();
        auto bias_dim = bias_real.dim();
        if (is_a4w4) {
            // A4W4量化场景下只支持1维bias
            TORCH_CHECK(bias_dim == 1,
                        "A4W4 quantization only supports 1-D bias, but got dim=", bias_dim,
                        OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(bias_real.size(0) == x2_n_dim || bias_real.size(0) == 1,
                        "bias size must match x2's n dimension or be 1, but got bias.size(0)=",
                        bias_real.size(0), " and x2_n_dim=", x2_n_dim, OPS_ERROR(ErrCode::PARAM));
        } else {
            // A8W8量化场景支持1维或3维bias
            TORCH_CHECK(bias_dim == 1 || bias_dim == 3,
                        "bias must be 1-D (n,) or 3-D (batch, 1, n), but got dim=", bias_dim,
                        OPS_ERROR(ErrCode::PARAM));
            if (bias_dim == 1) {
                TORCH_CHECK(bias_real.size(0) == x2_n_dim || bias_real.size(0) == 1,
                            "bias size must match x2's n dimension or be 1, but got bias.size(0)=",
                            bias_real.size(0), " and x2_n_dim=", x2_n_dim, OPS_ERROR(ErrCode::PARAM));
            } else if (bias_dim == 3) {
                TORCH_CHECK(bias_real.size(2) == x2_n_dim || bias_real.size(2) == 1,
                            "bias size[2] must match x2's n dimension or be 1, but got bias.size(2)=",
                            bias_real.size(2), " and x2_n_dim=", x2_n_dim, OPS_ERROR(ErrCode::PARAM));
                TORCH_CHECK(bias_real.size(1) == 1,
                            "bias size[1] must be 1 for 3-D bias, but got ", bias_real.size(1),
                            OPS_ERROR(ErrCode::PARAM));
            }
        }
    }
}

// ==================== 核心计算逻辑 ====================

// 执行 ACLNN 算子调用
static void execute_aclnn_kernel(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &x1_scale,
    const at::Tensor &x2_scale,
    const c10::optional<at::Tensor> &bias,
    bool transpose_x1,
    bool transpose_x2,
    c10::string_view fused_op_type,
    at::Tensor &result)
{

    // 1. 初始化预留参数
    int64_t group_size = 0;
    at::Tensor y_scale_optional = at::Tensor();
    at::Tensor x1_offset_optional = at::Tensor();
    at::Tensor x2_offset_optional = at::Tensor();
    at::Tensor y_offset_optional = at::Tensor();
    at::Tensor x3_optional = at::Tensor();
    
    // 2. 处理bias参数：创建局部变量避免临时对象绑定问题
    at::Tensor bias_real = bias.has_value() ? bias.value() : at::Tensor();

    // 3. 处理 fused_op_type
    char fused_type[FUSED_TYPE_ARRAY_SIZE];
    memset(fused_type, 0, FUSED_TYPE_ARRAY_SIZE);
    std::string fused_type_str(fused_op_type);
    fused_type_str.copy(fused_type, FUSED_TYPE_ARRAY_SIZE);

    // 4. 处理transpose参数：A4W4时transpose_x1必须为false
    bool is_a4w4 = (x1.dtype() == at::kInt && x2.dtype() == at::kInt);
    bool transpose1_for_aclnn = is_a4w4 ? false : transpose_x1;
    bool transpose2_for_aclnn = transpose_x2;

    // 5. 根据 x2 格式选择接口
    if (is_nz_format(x2)) {
        static const bool is_fused_quant_matmul_weight_nz_available =
            check_aclnn_kernel_available("aclnnFusedQuantMatmulWeightNz");
        TORCH_CHECK(is_fused_quant_matmul_weight_nz_available,
                    "aclnnFusedQuantMatmulWeightNz is not available, please upgrade CANN.",
                    OPS_ERROR(ErrCode::NOT_SUPPORT));

        EXEC_NPU_CMD(aclnnFusedQuantMatmulWeightNz, x1, x2, x1_scale, x2_scale,
                     y_scale_optional, x1_offset_optional, x2_offset_optional,
                     y_offset_optional, bias_real, x3_optional,
                     fused_type, transpose1_for_aclnn, transpose2_for_aclnn, group_size, result);
    } else {
        static const bool is_fused_quant_matmul_available =
            check_aclnn_kernel_available("aclnnFusedQuantMatmul");
        TORCH_CHECK(is_fused_quant_matmul_available,
                    "aclnnFusedQuantMatmul is not available, please upgrade CANN.",
                    OPS_ERROR(ErrCode::NOT_SUPPORT));

        EXEC_NPU_CMD(aclnnFusedQuantMatmul, x1, x2, x1_scale, x2_scale,
                     y_scale_optional, x1_offset_optional, x2_offset_optional,
                     y_offset_optional, bias_real, x3_optional,
                     fused_type, transpose1_for_aclnn, transpose2_for_aclnn, group_size, result);
    }
}

// ==================== 主函数 ====================

// 计算实际维度（考虑转置和INT4打包）
static void compute_actual_dimensions(
    const at::Tensor &x1,
    const at::Tensor &x2,
    bool transpose_x1,
    bool transpose_x2,
    int64_t &x1_m_dim,
    int64_t &x2_n_dim)
{
    bool is_a4w4_int = (x1.dtype() == at::kInt && x2.dtype() == at::kInt);
    
    // 计算x1的m维度
    if (transpose_x1) {
        x1_m_dim = x1.size(x1.dim() - 1);
    } else {
        x1_m_dim = x1.size(x1.dim() - LAST_SECOND_DIM_INDEX);
    }
    
    // 计算x2的n维度（考虑转置和INT4打包）
    if (transpose_x2) {
        // transpose_x2=true时：维度为(batch, n, k)，n在倒数第二维
        x2_n_dim = x2.size(x2.dim() - LAST_SECOND_DIM_INDEX);
        // INT4打包：n维度不需要恢复（因为打包的是k维度）
    } else {
        // transpose_x2=false时：维度为(batch, k, n)，n在最后一维
        x2_n_dim = x2.size(x2.dim() - 1);
        // INT4打包：恢复x2的实际n维度
        if (is_a4w4_int) {
            x2_n_dim = x2_n_dim * INT4_NUMS_IN_INT32;  // 恢复实际的n维度
        }
    }
}

at::Tensor npu_quant_matmul_gelu(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &x1_scale,
    const at::Tensor &x2_scale,
    c10::string_view fused_op_type,
    bool transpose_x1,
    bool transpose_x2,
    const c10::optional<at::Tensor> &bias) {

    // 1. 参数校验
    validate_inputs(x1, x2, x1_scale, x2_scale, fused_op_type, transpose_x1, transpose_x2, bias);

    // 2. 计算实际维度（考虑INT4打包）
    int64_t x1_m_dim, x2_n_dim;
    compute_actual_dimensions(x1, x2, transpose_x1, transpose_x2, x1_m_dim, x2_n_dim);

    // 3. 推导输出形状
    std::vector<uint64_t> batch_record;
    uint64_t batch_val = infer_out_batch_shape_gelu(x1, x2, batch_record);
    const at::Tensor long_tensor = x1.dim() > x2.dim() ? x1 : x2;
    c10::SmallVector<int64_t, op_infer::SIZE> output_size =
        op_infer::array_to_small_vector(long_tensor.sizes());
    output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = x1_m_dim;
    output_size[long_tensor.dim() - 1] = x2_n_dim;
    for (int64_t i = 0; i < long_tensor.dim() - LAST_SECOND_DIM_INDEX; i++) {
        output_size[i] = static_cast<int64_t>(batch_record[i]);
    }

    // 4. 分配输出张量
    at::ScalarType output_dtype = (x2_scale.dtype() == at::kBFloat16) ? at::kBFloat16 : at::kHalf;
    c10::TensorOptions options = x1.options().dtype(output_dtype);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    // 5. 调用 ACLNN 算子
    execute_aclnn_kernel(x1, x2, x1_scale, x2_scale, bias, transpose_x1, transpose_x2, fused_op_type, result);

    return result;
}

} // namespace op_api
