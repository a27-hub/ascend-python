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

// 校验输入场景（A4W4或A8W8）
static void validate_quantization_scenario(
    const at::Tensor &x1,
    const at::Tensor &x2,
    bool transpose_x1)
{
    // 校验数据类型，确定量化场景
    bool is_a4w4 = (x1.dtype() == at::kInt && x2.dtype() == at::kInt);
    bool is_a8w8 = (x1.dtype() == at::kChar && x2.dtype() == at::kChar);
    
    TORCH_CHECK(is_a4w4 || is_a8w8,
                "Only A4W4 (int32) or A8W8 (int8) quantization is supported, "
                "but got x1.dtype=", x1.dtype(), ", x2.dtype=", x2.dtype(),
                OPS_ERROR(ErrCode::TYPE));
    
    // A4W4特殊约束校验
    if (is_a4w4) {
        // A4W4不支持NZ格式
        if (is_nz_format(x2)) {
            TORCH_CHECK(false,
                        "A4W4 quantization does not support NZ format, x2 must be in ND format",
                        OPS_ERROR(ErrCode::PARAM));
        }
        
        // A4W4不支持transpose_x1
        if (transpose_x1) {
            TORCH_CHECK(false,
                        "A4W4 quantization only supports transpose_x1=false, but got transpose_x1=true",
                        OPS_ERROR(ErrCode::PARAM));
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
    // 处理 fused_op_type
    char fused_type[FUSED_TYPE_ARRAY_SIZE] = {0};
    TORCH_CHECK(std::string(fused_op_type).size() <= FUSED_TYPE_ARRAY_SIZE,
                "the len of fused_op_type is bigger than the default");
    std::string(fused_op_type).copy(fused_type, FUSED_TYPE_ARRAY_SIZE);

    // 处理transpose参数：A4W4时transpose_x1必须为false
    bool is_a4w4 = (x1.dtype() == at::kInt && x2.dtype() == at::kInt);
    bool transpose1_for_aclnn = is_a4w4 ? false : transpose_x1;
    bool transpose2_for_aclnn = transpose_x2;

    // 处理bias参数
    const at::Tensor &bias_real = bias.value_or(at::Tensor());

    // 预留参数
    int64_t group_size = 0;
    const at::Tensor empty_tensor = at::Tensor();

    // 根据 x2 格式选择接口
    if (is_nz_format(x2)) {
        EXEC_NPU_CMD(aclnnFusedQuantMatmulWeightNz, x1, x2, x1_scale, x2_scale,
                     empty_tensor, empty_tensor, empty_tensor, empty_tensor, bias_real, empty_tensor,
                     fused_type, transpose1_for_aclnn, transpose2_for_aclnn, group_size, result);
    } else {
        EXEC_NPU_CMD(aclnnFusedQuantMatmul, x1, x2, x1_scale, x2_scale,
                     empty_tensor, empty_tensor, empty_tensor, empty_tensor, bias_real, empty_tensor,
                     fused_type, transpose1_for_aclnn, transpose2_for_aclnn, group_size, result);
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
    // 1. 校验输入场景（A4W4或A8W8）
    validate_quantization_scenario(x1, x2, transpose_x1);

    // 2. 推导输出size
    bool is_a4w4 = (x1.dtype() == at::kInt && x2.dtype() == at::kInt);
    
    // 计算x1的m维度
    int64_t x1_m_dim = transpose_x1 ? x1.size(x1.dim() - 1) : x1.size(x1.dim() - LAST_SECOND_DIM_INDEX);
    
    // 计算x2的n维度（考虑转置和INT4打包）
    int64_t x2_n_dim;
    if (transpose_x2) {
        x2_n_dim = x2.size(x2.dim() - LAST_SECOND_DIM_INDEX);
    } else {
        x2_n_dim = x2.size(x2.dim() - 1);
        // A4W4场景：恢复实际的n维度（INT4打包）
        if (is_a4w4) {
            x2_n_dim = x2_n_dim * INT4_NUMS_IN_INT32;
        }
    }
    
    // 推导输出形状
    std::vector<uint64_t> batch_record;
    infer_out_batch_shape_gelu(x1, x2, batch_record);
    const at::Tensor long_tensor = x1.dim() > x2.dim() ? x1 : x2;
    c10::SmallVector<int64_t, op_infer::SIZE> output_size =
        op_infer::array_to_small_vector(long_tensor.sizes());
    output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = x1_m_dim;
    output_size[long_tensor.dim() - 1] = x2_n_dim;
    for (int64_t i = 0; i < long_tensor.dim() - LAST_SECOND_DIM_INDEX; i++) {
        output_size[i] = static_cast<int64_t>(batch_record[i]);
    }

    // 3. 分配输出张量
    at::ScalarType output_dtype = (x2_scale.dtype() == at::kBFloat16) ? at::kBFloat16 : at::kHalf;
    c10::TensorOptions options = x1.options().dtype(output_dtype);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    // 4. 调用aclnn接口
    execute_aclnn_kernel(x1, x2, x1_scale, x2_scale, bias, transpose_x1, transpose_x2, fused_op_type, result);

    return result;
}

} // namespace op_api
