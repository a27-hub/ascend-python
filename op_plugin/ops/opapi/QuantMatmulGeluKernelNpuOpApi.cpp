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
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr int64_t INT4_NUMS_IN_INT32 = 8;
constexpr size_t FUSED_TYPE_ARRAY_SIZE = 100;
using npu_preparation = at_npu::native::OpPreparation;

// 检查权重是否为 NZ 格式
static bool is_nz_format(const at::Tensor& x2)
{
    const torch_npu::NPUStorageDesc &tensor_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ ||
           tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_4;
}

// 检查是否为最后两维转置
bool static is_transpose_last_two_dims(const at::Tensor &tensor)
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
    int64_t vaild_offset = out_dim_num - std::min(x1_dim_num, x2_dim_num);
    for (int64_t i = 0; i < out_dim_num - LAST_SECOND_DIM_INDEX; i++) {
        auto short_dim = i < vaild_offset ? 1 : shape_short.size(i - vaild_offset);
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

// 中间计算结果结构体
struct QuantMatmulGeluParams {
    bool is_weight_nz;
    bool is_a8w8;
    bool is_a4w4;
    bool is_a4w4_int;
    int64_t x1_dim_num;
    int64_t x2_dim_num;
    int64_t x2_storage_dim_num;
    int64_t x1_k_dim;
    int64_t x2_k_dim;
    int64_t x2_n_dim;
    int64_t x1_m_dim;
    std::string fused_type_str;
    char fused_type[FUSED_TYPE_ARRAY_SIZE];
};

// ==================== 1. 参数预处理 ====================
static void preprocess_params(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &x1_scale,
    const at::Tensor &x2_scale,
    c10::string_view fused_op_type,
    bool transpose_x1,
    QuantMatmulGeluParams &params)
{
    // 1.1 基础参数校验：确保 tensor 已定义
    TORCH_CHECK(x1.defined(), "x1 must be defined", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2.defined(), "x2 must be defined", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1_scale.defined(), "x1_scale must be defined", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2_scale.defined(), "x2_scale must be defined", OPS_ERROR(ErrCode::PARAM));
    
    // 1.2 格式检查：x1必须是ND格式
    TORCH_CHECK(at_npu::native::FormatHelper::IsOpInputBaseFormat(x1),
                "x1 should be in base format (ND), but it is ",
                npu_preparation::get_tensor_npu_format(x1), OPS_ERROR(ErrCode::PARAM));
    
    // 1.3 判断x2格式（ND或NZ）
    params.is_weight_nz = is_nz_format(x2);
    
    // 1.4 检查量化类型：A8W8 (INT8) 或 A4W4 (INT4/INT32)
    params.is_a8w8 = (x1.dtype() == at::kChar && x2.dtype() == at::kChar);
    params.is_a4w4 = (x1.dtype() == at::kInt && x2.dtype() == at::kInt);
    params.is_a4w4_int = params.is_a4w4;
    TORCH_CHECK(params.is_a8w8 || params.is_a4w4,
                "Only support A8W8 (INT8) or A4W4 (INT4/INT32) quantization, but got x1 dtype: ",
                x1.dtype(), " and x2 dtype: ", x2.dtype(), OPS_ERROR(ErrCode::TYPE));
    
    // 1.5 A4W4量化约束：transpose_x1仅支持false
    if (params.is_a4w4 && transpose_x1) {
        TORCH_CHECK(false,
                    "A4W4 quantization only supports transpose_x1=false, but got transpose_x1=true",
                    OPS_ERROR(ErrCode::PARAM));
    }
    
    // 1.6 检查 fused_op_type 是否合法
    params.fused_type_str = std::string(fused_op_type);
    TORCH_CHECK(params.fused_type_str == "gelu_tanh" || params.fused_type_str == "gelu_erf",
                "fused_op_type must be 'gelu_tanh' or 'gelu_erf', but got: ",
                params.fused_type_str, OPS_ERROR(ErrCode::PARAM));
    
    // 1.7 转换 fused_op_type 为 C 字符串
    memset(params.fused_type, 0, FUSED_TYPE_ARRAY_SIZE);
    TORCH_CHECK(params.fused_type_str.size() <= FUSED_TYPE_ARRAY_SIZE,
                "the len of fused_op_type is bigger than the default", OPS_ERROR(ErrCode::PARAM));
    params.fused_type_str.copy(params.fused_type, FUSED_TYPE_ARRAY_SIZE);
    
    // 1.8 检查 scale 的数据类型
    TORCH_CHECK(x1_scale.dtype() == at::kFloat,
                "x1_scale must be FLOAT32, but got: ", x1_scale.dtype(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(x2_scale.dtype() == at::kFloat || x2_scale.dtype() == at::kBFloat16,
                "x2_scale must be FLOAT32 or BFLOAT16, but got: ", x2_scale.dtype(), OPS_ERROR(ErrCode::TYPE));
    
    // 1.9 检查 scale 的维度
    TORCH_CHECK(x1_scale.dim() == 1, "x1_scale must be 1-D, but got: ", x1_scale.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2_scale.dim() == 1, "x2_scale must be 1-D, but got: ", x2_scale.dim(), OPS_ERROR(ErrCode::PARAM));
    
    // 1.10 获取维度数
    params.x1_dim_num = x1.dim();
    params.x2_dim_num = x2.dim();
    params.x2_storage_dim_num = params.x2_dim_num;  // 对于ND格式，逻辑维度数等于存储维度数
}

// ==================== 2. 维度校验 ====================
static void validate_dimensions(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &x1_scale,
    const at::Tensor &x2_scale,
    bool transpose_x1,
    bool transpose_x2,
    QuantMatmulGeluParams &params)
{
    // 2.1 校验x1维度：2~6维
    TORCH_CHECK(params.x1_dim_num >= 2 && params.x1_dim_num <= 6,
                "x1 shape dim num must be in [2, 6], but it is ", params.x1_dim_num, OPS_ERROR(ErrCode::PARAM));
    
    // 2.2 校验x2维度（根据格式）
    if (params.is_weight_nz) {
        // NZ格式：从storage_sizes_获取存储维度数
        const torch_npu::NPUStorageDesc &tensor_desc =
            torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
        auto storage_sizes = tensor_desc.storage_sizes_;
        
        if (!storage_sizes.empty()) {
            params.x2_storage_dim_num = static_cast<int64_t>(storage_sizes.size());
        } else {
            TORCH_CHECK(false,
                        "For NZ format, cannot get storage shape dimension from storage_sizes_, "
                        "x2.dim() returns logical dimension: ", params.x2_dim_num,
                        OPS_ERROR(ErrCode::PARAM));
        }
        
        TORCH_CHECK(params.x2_storage_dim_num >= 4 && params.x2_storage_dim_num <= 8,
                    "For NZ format, x2 storage shape dim num must be in [4, 8], but it is ",
                    params.x2_storage_dim_num, OPS_ERROR(ErrCode::PARAM));
    } else {
        // ND格式：根据量化类型校验维度
        if (params.is_a4w4) {
            TORCH_CHECK(params.x2_dim_num == 2,
                        "A4W4 quantization with ND format only supports 2-D x2, but got dim=",
                        params.x2_dim_num, OPS_ERROR(ErrCode::PARAM));
        } else {
            TORCH_CHECK(params.x2_dim_num >= 2 && params.x2_dim_num <= 6,
                        "For ND format, x2 shape dim num must be in [2, 6], but it is ",
                        params.x2_dim_num, OPS_ERROR(ErrCode::PARAM));
        }
    }
    
    // 2.3 计算x2的k和n维度（考虑转置）
    if (transpose_x2) {
        params.x2_n_dim = x2.size(params.x2_dim_num - LAST_SECOND_DIM_INDEX);
        params.x2_k_dim = x2.size(params.x2_dim_num - 1);
    } else {
        params.x2_k_dim = x2.size(params.x2_dim_num - LAST_SECOND_DIM_INDEX);
        params.x2_n_dim = x2.size(params.x2_dim_num - 1);
    }
    
    // 2.4 NZ格式约束：k、n不能为1
    if (params.is_weight_nz) {
        TORCH_CHECK(params.x2_k_dim != 1 && params.x2_n_dim != 1,
                    "When x2 is NZ format, k and n cannot be 1, but got k=", params.x2_k_dim,
                    " and n=", params.x2_n_dim, OPS_ERROR(ErrCode::PARAM));
    }
    
    // 2.5 INT4打包处理：恢复实际维度
    if (params.is_a4w4_int) {
        if (transpose_x2) {
            TORCH_CHECK(params.x2_k_dim % INT4_NUMS_IN_INT32 == 0,
                        "A4W4 quantization with transpose_x2=true requires k to be a multiple of 8, but got k=",
                        params.x2_k_dim, OPS_ERROR(ErrCode::PARAM));
            params.x2_k_dim = params.x2_k_dim * INT4_NUMS_IN_INT32;
        } else {
            TORCH_CHECK(params.x2_n_dim % INT4_NUMS_IN_INT32 == 0,
                        "A4W4 quantization with transpose_x2=false requires n to be a multiple of 8, but got n=",
                        params.x2_n_dim, OPS_ERROR(ErrCode::PARAM));
            params.x2_n_dim = params.x2_n_dim * INT4_NUMS_IN_INT32;
        }
    }
    
    // 2.6 计算x1的k维度（考虑转置和INT4打包）
    if (transpose_x1) {
        params.x1_k_dim = x1.size(params.x1_dim_num - 1);
    } else {
        params.x1_k_dim = x1.size(params.x1_dim_num - 1);
        if (params.is_a4w4_int) {
            TORCH_CHECK(params.x1_k_dim % INT4_NUMS_IN_INT32 == 0,
                        "A4W4 quantization requires k to be a multiple of 8, but got k=",
                        params.x1_k_dim, OPS_ERROR(ErrCode::PARAM));
            params.x1_k_dim = params.x1_k_dim * INT4_NUMS_IN_INT32;
        }
    }
    
    // 2.7 验证x1和x2的k维度一致
    TORCH_CHECK(params.x1_k_dim == params.x2_k_dim,
                "x1's k dimension must match x2's k dimension, but got x1_k_dim=", params.x1_k_dim,
                " and x2_k_dim=", params.x2_k_dim, OPS_ERROR(ErrCode::PARAM));
    
    // 2.8 NZ格式存储维度关系验证
    if (params.is_weight_nz) {
        int64_t k0_expected = transpose_x2 ? 32 : 16;
        int64_t n0_expected = transpose_x2 ? 16 : 32;
        
        const torch_npu::NPUStorageDesc &tensor_desc =
            torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
        auto storage_sizes = tensor_desc.storage_sizes_;
        
        if (!storage_sizes.empty() && storage_sizes.size() == static_cast<size_t>(params.x2_storage_dim_num)) {
            int64_t k1_storage_idx = transpose_x2 ? (params.x2_storage_dim_num - 4) : (params.x2_storage_dim_num - 3);
            if (k1_storage_idx >= 0 && k1_storage_idx < static_cast<int64_t>(storage_sizes.size())) {
                int64_t k1_storage = static_cast<int64_t>(storage_sizes[k1_storage_idx]);
                int64_t k_expected_from_k1 = (params.x1_k_dim + k0_expected - 1) / k0_expected;
                
                TORCH_CHECK(k1_storage == k_expected_from_k1,
                            "For NZ format with transpose_x2=", transpose_x2,
                            ", x2 storage shape's k1 must satisfy ceil(k / ", k0_expected, ") = k1, "
                            "where k=", params.x1_k_dim, " (expected k1=", k_expected_from_k1,
                            "), but got k1=", k1_storage, OPS_ERROR(ErrCode::PARAM));
            }
            
            if (params.x2_storage_dim_num >= 5) {
                int64_t k0_idx = transpose_x2 ? (params.x2_storage_dim_num - 1) : (params.x2_storage_dim_num - 2);
                int64_t n0_idx = transpose_x2 ? (params.x2_storage_dim_num - 2) : (params.x2_storage_dim_num - 1);
                
                if (k0_idx >= 0 && k0_idx < static_cast<int64_t>(storage_sizes.size()) &&
                    n0_idx >= 0 && n0_idx < static_cast<int64_t>(storage_sizes.size())) {
                    int64_t k0_actual = static_cast<int64_t>(storage_sizes[k0_idx]);
                    int64_t n0_actual = static_cast<int64_t>(storage_sizes[n0_idx]);
                    
                    TORCH_CHECK(k0_actual == k0_expected,
                                "For NZ format with transpose_x2=", transpose_x2,
                                ", k0 must be ", k0_expected, ", but got ", k0_actual,
                                OPS_ERROR(ErrCode::PARAM));
                    TORCH_CHECK(n0_actual == n0_expected,
                                "For NZ format with transpose_x2=", transpose_x2,
                                ", n0 must be ", n0_expected, ", but got ", n0_actual,
                                OPS_ERROR(ErrCode::PARAM));
                }
            }
        }
    }
    
    // 2.9 A4W4量化约束：内轴必须为偶数
    if (params.is_a4w4) {
        TORCH_CHECK(params.x1_k_dim % 2 == 0,
                    "A4W4 quantization requires x1's inner axis (k) to be even, but got k=",
                    params.x1_k_dim, OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(params.x2_k_dim % 2 == 0,
                    "A4W4 quantization requires x2's inner axis (k) to be even, but got k=",
                    params.x2_k_dim, OPS_ERROR(ErrCode::PARAM));
    }
    
    // 2.10 计算x1的m维度
    params.x1_m_dim = transpose_x1 ? x1.size(params.x1_dim_num - 1) : x1.size(params.x1_dim_num - LAST_SECOND_DIM_INDEX);
    
    // 2.11 校验scale的shape
    TORCH_CHECK(x1_scale.size(0) == params.x1_m_dim,
                "x1_scale size must match x1's m dimension, but got x1_scale.size(0)=",
                x1_scale.size(0), " and x1_m_dim=", params.x1_m_dim, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2_scale.size(0) == params.x2_n_dim || x2_scale.size(0) == 1,
                "x2_scale size must match x2's n dimension or be 1, but got x2_scale.size(0)=",
                x2_scale.size(0), " and x2_n_dim=", params.x2_n_dim, OPS_ERROR(ErrCode::PARAM));
}

// ==================== 3. 格式转换 ====================
static at::Tensor prepare_output_and_validate_bias(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &x2_scale,
    const c10::optional<at::Tensor> &bias,
    const QuantMatmulGeluParams &params)
{
    // 3.1 推导batch shape
    std::vector<uint64_t> batch_record;
    uint64_t batch_val = infer_out_batch_shape_gelu(x1, x2, batch_record);
    
    // 3.2 构建输出shape
    const at::Tensor long_tensor = params.x1_dim_num > params.x2_dim_num ? x1 : x2;
    c10::SmallVector<int64_t, op_infer::SIZE> output_size;
    output_size = op_infer::array_to_small_vector(long_tensor.sizes());
    output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = params.x1_m_dim;
    output_size[long_tensor.dim() - 1] = params.x2_n_dim;
    for (int64_t i = 0; i < long_tensor.dim() - LAST_SECOND_DIM_INDEX; i++) {
        output_size[i] = static_cast<int64_t>(batch_record[i]);
    }
    
    // 3.3 确定输出数据类型并分配tensor
    at::ScalarType output_dtype = (x2_scale.dtype() == at::kBFloat16) ? at::kBFloat16 : at::kHalf;
    c10::TensorOptions options = x1.options().dtype(output_dtype);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);
    
    // 3.4 验证bias参数
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    if (bias.has_value() && bias_real.defined()) {
        auto bias_dim = bias_real.dim();
        if (params.is_a4w4) {
            TORCH_CHECK(bias_dim == 1,
                        "A4W4 quantization only supports 1-D bias, but got dim=", bias_dim,
                        OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(bias_real.size(0) == params.x2_n_dim || bias_real.size(0) == 1,
                        "bias size must match x2's n dimension or be 1, but got bias.size(0)=",
                        bias_real.size(0), " and x2_n_dim=", params.x2_n_dim, OPS_ERROR(ErrCode::PARAM));
        } else {
            TORCH_CHECK(bias_dim == 1 || bias_dim == 3,
                        "bias must be 1-D (n,) or 3-D (batch, 1, n), but got dim=", bias_dim,
                        OPS_ERROR(ErrCode::PARAM));
            if (bias_dim == 1) {
                TORCH_CHECK(bias_real.size(0) == params.x2_n_dim || bias_real.size(0) == 1,
                            "bias size must match x2's n dimension or be 1, but got bias.size(0)=",
                            bias_real.size(0), " and x2_n_dim=", params.x2_n_dim, OPS_ERROR(ErrCode::PARAM));
            } else if (bias_dim == 3) {
                TORCH_CHECK(bias_real.size(2) == params.x2_n_dim || bias_real.size(2) == 1,
                            "bias size[2] must match x2's n dimension or be 1, but got bias.size(2)=",
                            bias_real.size(2), " and x2_n_dim=", params.x2_n_dim, OPS_ERROR(ErrCode::PARAM));
                TORCH_CHECK(bias_real.size(1) == 1,
                            "bias size[1] must be 1 for 3-D bias, but got ", bias_real.size(1),
                            OPS_ERROR(ErrCode::PARAM));
            }
        }
    }
    
    return result;
}

// ==================== 4. 核心计算 ====================
static void execute_aclnn_kernel(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &x1_scale,
    const at::Tensor &x2_scale,
    const c10::optional<at::Tensor> &bias,
    bool transpose_x1,
    bool transpose_x2,
    const QuantMatmulGeluParams &params,
    at::Tensor &result)
{
    // 4.1 准备预留参数
    int64_t group_size = 0;
    at::Tensor y_scale_optional = at::Tensor();
    at::Tensor x1_offset_optional = at::Tensor();
    at::Tensor x2_offset_optional = at::Tensor();
    at::Tensor y_offset_optional = at::Tensor();
    at::Tensor x3_optional = at::Tensor();
    
    // 4.2 处理transpose参数
    bool transpose1_for_aclnn = params.is_a4w4 ? false : transpose_x1;
    bool transpose2_for_aclnn = transpose_x2;
    
    // 4.3 处理bias参数
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    
    // 4.4 根据x2格式选择并调用aclnn算子
    if (params.is_weight_nz) {
        static const bool is_fused_quant_matmul_weight_nz_available = 
            check_aclnn_kernel_available("aclnnFusedQuantMatmulWeightNz");
        TORCH_CHECK(is_fused_quant_matmul_weight_nz_available,
                    "aclnnFusedQuantMatmulWeightNz is not available, please upgrade CANN.",
                    OPS_ERROR(ErrCode::NOT_SUPPORT));
        
        EXEC_NPU_CMD(aclnnFusedQuantMatmulWeightNz, x1, x2, x1_scale, x2_scale,
                     y_scale_optional, x1_offset_optional, x2_offset_optional,
                     y_offset_optional, bias_real, x3_optional,
                     params.fused_type, transpose1_for_aclnn, transpose2_for_aclnn,
                     group_size, result);
    } else {
        static const bool is_fused_quant_matmul_available = 
            check_aclnn_kernel_available("aclnnFusedQuantMatmul");
        TORCH_CHECK(is_fused_quant_matmul_available,
                    "aclnnFusedQuantMatmul is not available, please upgrade CANN.",
                    OPS_ERROR(ErrCode::NOT_SUPPORT));
        
        EXEC_NPU_CMD(aclnnFusedQuantMatmul, x1, x2, x1_scale, x2_scale,
                     y_scale_optional, x1_offset_optional, x2_offset_optional,
                     y_offset_optional, bias_real, x3_optional,
                     params.fused_type, transpose1_for_aclnn, transpose2_for_aclnn,
                     group_size, result);
    }
}

// ==================== 主函数 ====================
at::Tensor npu_quant_matmul_gelu(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &x1_scale,
    const at::Tensor &x2_scale,
    c10::string_view fused_op_type,
    bool transpose_x1,
    bool transpose_x2,
    const c10::optional<at::Tensor> &bias)
{
    QuantMatmulGeluParams params;
    
    // 1. 参数预处理
    preprocess_params(x1, x2, x1_scale, x2_scale, fused_op_type, transpose_x1, params);
    
    // 2. 维度校验
    validate_dimensions(x1, x2, x1_scale, x2_scale, transpose_x1, transpose_x2, params);
    
    // 3. 格式转换：准备输出tensor并验证bias
    at::Tensor result = prepare_output_and_validate_bias(x1, x2, x2_scale, bias, params);
    
    // 4. 核心计算：调用aclnn算子
    execute_aclnn_kernel(x1, x2, x1_scale, x2_scale, bias, transpose_x1, transpose_x2, params, result);
    
    return result;
}

} // namespace op_api
