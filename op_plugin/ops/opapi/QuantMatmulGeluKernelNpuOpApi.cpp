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
    // 参数校验：确保 tensor 已定义
    TORCH_CHECK(x1.defined(), "x1 must be defined", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2.defined(), "x2 must be defined", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1_scale.defined(), "x1_scale must be defined", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2_scale.defined(), "x2_scale must be defined", OPS_ERROR(ErrCode::PARAM));
    
    // 格式检查：x1 必须是 base format（ND格式），x2 可以是 base format 或 NZ format
    // 注意：格式检查在 infer_out_batch_shape_gelu 函数中也会进行，这里提前检查以便给出更清晰的错误信息
    // 但为了兼容代码生成系统，我们使用 IsOpInputBaseFormat 来检查 x1
    TORCH_CHECK(at_npu::native::FormatHelper::IsOpInputBaseFormat(x1),
                "x1 should be in base format (ND), but it is ",
                npu_preparation::get_tensor_npu_format(x1), OPS_ERROR(ErrCode::PARAM));
    // x2 的格式检查在 infer_out_batch_shape_gelu 函数中进行，这里不重复检查
    
    auto x1_dim_num = x1.dim();
    TORCH_CHECK(x1_dim_num >= 2 && x1_dim_num <= 6,
                "x1 shape dim num must be in [2, 6], but it is ", x1_dim_num, OPS_ERROR(ErrCode::PARAM));
    
    auto x2_dim_num = x2.dim();
    TORCH_CHECK(x2_dim_num >= 2 && x2_dim_num <= 6,
                "x2 shape dim num must be in [2, 6], but it is ", x2_dim_num, OPS_ERROR(ErrCode::PARAM));

    // 检查量化类型：A8W8 (INT8) 或 A4W4 (INT4/INT32)
    bool is_a8w8 = (x1.dtype() == at::kChar && x2.dtype() == at::kChar);
    bool is_a4w4 = (x1.dtype() == at::kInt && x2.dtype() == at::kInt);
    TORCH_CHECK(is_a8w8 || is_a4w4,
                "Only support A8W8 (INT8) or A4W4 (INT4/INT32) quantization, but got x1 dtype: ",
                x1.dtype(), " and x2 dtype: ", x2.dtype(), OPS_ERROR(ErrCode::TYPE));
    
    // A4W4 量化约束：transpose_x1 仅支持 false
    if (is_a4w4 && transpose_x1) {
        TORCH_CHECK(false,
                    "A4W4 quantization only supports transpose_x1=false, but got transpose_x1=true",
                    OPS_ERROR(ErrCode::PARAM));
    }

    // 检查 fused_op_type 是否合法
    std::string fused_type_str(fused_op_type);
    TORCH_CHECK(fused_type_str == "gelu_tanh" || fused_type_str == "gelu_erf",
                "fused_op_type must be 'gelu_tanh' or 'gelu_erf', but got: ",
                fused_type_str, OPS_ERROR(ErrCode::PARAM));

    // 检查 x1_scale 和 x2_scale 的数据类型
    TORCH_CHECK(x1_scale.dtype() == at::kFloat,
                "x1_scale must be FLOAT32, but got: ", x1_scale.dtype(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(x2_scale.dtype() == at::kFloat || x2_scale.dtype() == at::kBFloat16,
                "x2_scale must be FLOAT32 or BFLOAT16, but got: ", x2_scale.dtype(), OPS_ERROR(ErrCode::TYPE));

    // 检查 scale 的维度
    TORCH_CHECK(x1_scale.dim() == 1, "x1_scale must be 1-D, but got: ", x1_scale.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2_scale.dim() == 1, "x2_scale must be 1-D, but got: ", x2_scale.dim(), OPS_ERROR(ErrCode::PARAM));

    // 计算 x2 的实际 n 维度（考虑转置和 INT4 打包）
    // 注意：PyTorch 的 size() 方法返回逻辑 shape，对于 NZ 格式也返回逻辑 shape
    bool is_a4w4_int = x1.dtype() == at::kInt && x2.dtype() == at::kInt;
    int64_t x2_n_dim;
    int64_t x2_k_dim;
    
    if (transpose_x2) {
        // 转置情况下：x2 shape 是 (n, k) 或 (batch, n, k)，n 在倒数第二维，k 在最后一维
        x2_n_dim = x2.size(x2_dim_num - LAST_SECOND_DIM_INDEX);
        x2_k_dim = x2.size(x2_dim_num - 1);
    } else {
        // 非转置情况下：x2 shape 是 (k, n) 或 (batch, k, n)，k 在倒数第二维，n 在最后一维
        x2_k_dim = x2.size(x2_dim_num - LAST_SECOND_DIM_INDEX);
        x2_n_dim = x2.size(x2_dim_num - 1);
    }
    
    // INT4 打包处理（适用于 ND 和 NZ 格式）
    if (is_a4w4_int) {
        if (transpose_x2) {
            // transposeX2为true时维度为：（n，k // 8），要求k为8的倍数
            TORCH_CHECK(x2_k_dim % INT4_NUMS_IN_INT32 == 0,
                        "A4W4 quantization with transpose_x2=true requires k to be a multiple of 8, but got k=",
                        x2_k_dim, OPS_ERROR(ErrCode::PARAM));
            x2_k_dim = x2_k_dim * INT4_NUMS_IN_INT32;  // 恢复实际的 k 维度
        } else {
            // transposeX2为false时维度为：（k，n // 8），要求n为8的倍数
            TORCH_CHECK(x2_n_dim % INT4_NUMS_IN_INT32 == 0,
                        "A4W4 quantization with transpose_x2=false requires n to be a multiple of 8, but got n=",
                        x2_n_dim, OPS_ERROR(ErrCode::PARAM));
            x2_n_dim = x2_n_dim * INT4_NUMS_IN_INT32;  // 恢复实际的 n 维度
        }
    }
    
    // 计算 x1 的实际 k 维度（考虑转置和 INT4 打包）
    int64_t x1_k_dim;
    if (transpose_x1) {
        // 转置情况下：x1 shape 是 (k, m)，k 在最后一维
        x1_k_dim = x1.size(x1_dim_num - 1);
    } else {
        // 非转置情况下：x1 shape 是 (m, k) 或 (batch, m, k)，k 在最后一维
        x1_k_dim = x1.size(x1_dim_num - 1);
        // INT4 打包：每个 INT32 存储 8 个 INT4
        if (is_a4w4_int) {
            // transposeX1为false时维度为：（batch，m，k // 8），要求k为8的倍数
            TORCH_CHECK(x1_k_dim % INT4_NUMS_IN_INT32 == 0,
                        "A4W4 quantization requires k to be a multiple of 8, but got k=",
                        x1_k_dim, OPS_ERROR(ErrCode::PARAM));
            x1_k_dim = x1_k_dim * INT4_NUMS_IN_INT32;  // 恢复实际的 k 维度
        }
    }
    
    // 验证 x1 的 k 维度和 x2 的 k 维度一致
    TORCH_CHECK(x1_k_dim == x2_k_dim,
                "x1's k dimension must match x2's k dimension, but got x1_k_dim=", x1_k_dim,
                " and x2_k_dim=", x2_k_dim, OPS_ERROR(ErrCode::PARAM));
    
    // A4W4 量化约束：x1 和 x2 的内轴（k 维度）必须为偶数
    // 注意：当数据类型为 INT32 时，每个 INT32 数据存放 8 个 INT4 数据
    // 此时 x1_k_dim 和 x2_k_dim 已经是恢复后的实际 k 维度（已乘以 8）
    if (is_a4w4) {
        TORCH_CHECK(x1_k_dim % 2 == 0,
                    "A4W4 quantization requires x1's inner axis (k) to be even, but got k=", x1_k_dim,
                    OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(x2_k_dim % 2 == 0,
                    "A4W4 quantization requires x2's inner axis (k) to be even, but got k=", x2_k_dim,
                    OPS_ERROR(ErrCode::PARAM));
    }
    
    // 检查 x1_scale 的 shape：应该是 (m,)，其中 m 是 x1 的 m 维度
    // per-token scale：每个 token（x1 的每一行）有一个 scale，所有 batch 共享相同的 scale
    auto x1_m_dim = transpose_x1 ? x1.size(x1_dim_num - 1) : x1.size(x1_dim_num - LAST_SECOND_DIM_INDEX);
    TORCH_CHECK(x1_scale.size(0) == x1_m_dim,
                "x1_scale size must match x1's m dimension, but got x1_scale.size(0)=",
                x1_scale.size(0), " and x1_m_dim=", x1_m_dim, OPS_ERROR(ErrCode::PARAM));
    
    // 检查 x2_scale 的 shape：应该是 (n,) 或 (1,)，其中 n 是 x2 的 n 维度
    TORCH_CHECK(x2_scale.size(0) == x2_n_dim || x2_scale.size(0) == 1,
                "x2_scale size must match x2's n dimension or be 1, but got x2_scale.size(0)=",
                x2_scale.size(0), " and x2_n_dim=", x2_n_dim, OPS_ERROR(ErrCode::PARAM));

    std::vector<uint64_t> batch_record;
    uint64_t batch_val = infer_out_batch_shape_gelu(x1, x2, batch_record);
    const at::Tensor long_tensor = x1_dim_num > x2_dim_num ? x1 : x2;
    auto output_size = op_infer::array_to_small_vector(long_tensor.sizes());
    // 设置输出的 m 维度（来自 x1，使用之前计算的 x1_m_dim）
    output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = x1_m_dim;
    // 设置输出的 n 维度（来自 x2，已考虑 INT4 打包）
    output_size[long_tensor.dim() - 1] = x2_n_dim;
    for (int64_t i = 0; i < long_tensor.dim() - LAST_SECOND_DIM_INDEX; i++) {
        output_size[i] = static_cast<int64_t>(batch_record[i]);
    }

    // 确定输出数据类型：根据 x2_scale 的类型，如果是 FLOAT32 则输出 FLOAT16，如果是 BFLOAT16 则输出 BFLOAT16
    at::ScalarType output_dtype = (x2_scale.dtype() == at::kBFloat16) ? at::kBFloat16 : at::kHalf;
    c10::TensorOptions options = x1.options().dtype(output_dtype);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    // 处理 bias 参数
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    if (bias.has_value() && bias_real.defined()) {
        // 验证 bias 的 shape：应该是 (n,) 或 (batch, 1, n)
        // INT4 量化场景下只支持 biasOptional 为 1 维
        auto bias_dim = bias_real.dim();
        if (is_a4w4) {
            // INT4 量化场景下只支持 1 维 bias
            TORCH_CHECK(bias_dim == 1,
                        "A4W4 quantization only supports 1-D bias, but got dim=", bias_dim,
                        OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(bias_real.size(0) == x2_n_dim || bias_real.size(0) == 1,
                        "bias size must match x2's n dimension or be 1, but got bias.size(0)=",
                        bias_real.size(0), " and x2_n_dim=", x2_n_dim, OPS_ERROR(ErrCode::PARAM));
        } else {
            // A8W8 量化场景支持 1 维或 3 维 bias
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

    // 转换 fused_op_type 为 C 字符串
    char fused_type[FUSED_TYPE_ARRAY_SIZE] = {0};
    TORCH_CHECK(fused_type_str.size() <= FUSED_TYPE_ARRAY_SIZE,
                "the len of fused_op_type is bigger than the default", OPS_ERROR(ErrCode::PARAM));
    fused_type_str.copy(fused_type, FUSED_TYPE_ARRAY_SIZE);

    // 根据权重格式选择调用的 aclnn 算子
    bool is_weight_nz = is_nz_format(x2);
    
    // NZ 格式约束：k、n 不能为 1
    if (is_weight_nz) {
        TORCH_CHECK(x2_k_dim != 1 && x2_n_dim != 1,
                    "When x2 is NZ format, k and n cannot be 1, but got k=", x2_k_dim,
                    " and n=", x2_n_dim, OPS_ERROR(ErrCode::PARAM));
    }
    
    int64_t group_size = 0;  // 当前版本不支持 group_size，传入 0

    // 预留参数：当前版本不支持，传入未定义的 tensor（ConvertType 会将其转换为 nullptr）
    at::Tensor y_scale_optional = at::Tensor();      // yScaleOptional - 预留参数，当前版本不支持
    at::Tensor x1_offset_optional = at::Tensor();    // x1OffsetOptional - 预留参数，当前版本不支持
    at::Tensor x2_offset_optional = at::Tensor();    // x2OffsetOptional - 预留参数，当前版本不支持
    at::Tensor y_offset_optional = at::Tensor();     // yOffsetOptional - 预留参数，当前版本不支持
    at::Tensor x3_optional = at::Tensor();           // x3Optional - 预留参数，当前版本不支持

    // 对于 NZ 格式，参考 QuantMatmulKernelNpuOpApi.cpp 的处理方式：
    // NZ 格式下总是传递 transpose1=false 和 transpose2=false
    // 转置信息通过 tensor 的格式本身来表达，不需要通过 transpose 参数传递
    bool transpose1_for_aclnn = is_weight_nz ? false : transpose_x1;
    bool transpose2_for_aclnn = is_weight_nz ? false : transpose_x2;

    if (is_weight_nz) {
        // 权重为 NZ 格式，调用 aclnnFusedQuantMatmulWeightNz
        static const bool is_fused_quant_matmul_weight_nz_available = 
            check_aclnn_kernel_available("aclnnFusedQuantMatmulWeightNz");
        TORCH_CHECK(is_fused_quant_matmul_weight_nz_available,
                    "aclnnFusedQuantMatmulWeightNz is not available, please upgrade CANN.",
                    OPS_ERROR(ErrCode::NOT_SUPPORT));
        
        EXEC_NPU_CMD(aclnnFusedQuantMatmulWeightNz, x1, x2, x1_scale, x2_scale,
                     y_scale_optional,      // yScaleOptional - 预留参数，传入未定义tensor会被转换为nullptr
                     x1_offset_optional,     // x1OffsetOptional - 预留参数，传入未定义tensor会被转换为nullptr
                     x2_offset_optional,     // x2OffsetOptional - 预留参数，传入未定义tensor会被转换为nullptr
                     y_offset_optional,      // yOffsetOptional - 预留参数，传入未定义tensor会被转换为nullptr
                     bias_real,              // biasOptional
                     x3_optional,           // x3Optional - 预留参数，传入未定义tensor会被转换为nullptr
                     fused_type,            // fusedOpType
                     transpose1_for_aclnn,  // transposeX1 - NZ格式下总是false
                     transpose2_for_aclnn,  // transposeX2 - NZ格式下总是false
                     group_size,            // groupSizeOptional - 预留参数，传入0
                     result);
    } else {
        // 权重为 ND 格式，调用 aclnnFusedQuantMatmul
        static const bool is_fused_quant_matmul_available = 
            check_aclnn_kernel_available("aclnnFusedQuantMatmul");
        TORCH_CHECK(is_fused_quant_matmul_available,
                    "aclnnFusedQuantMatmul is not available, please upgrade CANN.",
                    OPS_ERROR(ErrCode::NOT_SUPPORT));
        
        EXEC_NPU_CMD(aclnnFusedQuantMatmul, x1, x2, x1_scale, x2_scale,
                     y_scale_optional,      // yScaleOptional - 预留参数，传入未定义tensor会被转换为nullptr
                     x1_offset_optional,     // x1OffsetOptional - 预留参数，传入未定义tensor会被转换为nullptr
                     x2_offset_optional,    // x2OffsetOptional - 预留参数，传入未定义tensor会被转换为nullptr
                     y_offset_optional,     // yOffsetOptional - 预留参数，传入未定义tensor会被转换为nullptr
                     bias_real,             // biasOptional
                     x3_optional,           // x3Optional - 预留参数，传入未定义tensor会被转换为nullptr
                     fused_type,            // fusedOpType
                     transpose_x1,          // transposeX1
                     transpose_x2,          // transposeX2
                     group_size,            // groupSizeOptional - 预留参数，传入0
                     result);
    }

    return result;
}

} // namespace op_api
