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
static bool is_nz_format(const at::Tensor& x2) {
    const torch_npu::NPUStorageDesc &tensor_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ ||
           tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_4;
}

// 检查是否为最后两维转置
static bool is_transpose_last_two_dims(const at::Tensor &tensor) {
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
static uint64_t infer_out_batch_shape_gelu(const at::Tensor &x1, const at::Tensor &x2, std::vector<uint64_t> &batch_record) {
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
    const c10::optional<at::Tensor> &bias) {

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

    // 6. 特殊约束校验（A4W4 + transpose_x1）
    bool is_a4w4 = (x1.dtype() == at::kInt && x2.dtype() == at::kInt);
    if (is_a4w4 && transpose_x1) {
        TORCH_CHECK(false,
                    "A4W4 quantization only supports transpose_x1=false, but got transpose_x1=true",
                    OPS_ERROR(ErrCode::PARAM));
    }

    // 7. NZ 格式特殊校验
    if (is_nz_format(x2)) {
        const torch_npu::NPUStorageDesc &tensor_desc =
            torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
        auto storage_sizes = tensor_desc.storage_sizes_;
        int64_t x2_storage_dim_num = static_cast<int64_t>(storage_sizes.size());
        TORCH_CHECK(x2_storage_dim_num >= 4 && x2_storage_dim_num <= 8,
                    "For NZ format, x2 storage shape dim num must be in [4, 8], but it is ",
                    x2_storage_dim_num, OPS_ERROR(ErrCode::PARAM));

        int64_t k_dim = x2.size(x2.dim() - LAST_SECOND_DIM_INDEX);
        int64_t n_dim = x2.size(x2.dim() - 1);
        TORCH_CHECK(k_dim != 1 && n_dim != 1,
                    "When x2 is NZ format, k and n cannot be 1, but got k=", k_dim,
                    " and n=", n_dim, OPS_ERROR(ErrCode::PARAM));
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
    at::Tensor &result) {

    // 1. 初始化预留参数
    int64_t group_size = 0;
    at::Tensor y_scale_optional = at::Tensor();
    at::Tensor x1_offset_optional = at::Tensor();
    at::Tensor x2_offset_optional = at::Tensor();
    at::Tensor y_offset_optional = at::Tensor();
    at::Tensor x3_optional = at::Tensor();

    // 2. 处理 fused_op_type
    char fused_type[FUSED_TYPE_ARRAY_SIZE];
    memset(fused_type, 0, FUSED_TYPE_ARRAY_SIZE);
    std::string fused_type_str(fused_op_type);
    fused_type_str.copy(fused_type, FUSED_TYPE_ARRAY_SIZE);

    // 3. 根据 x2 格式选择接口
    if (is_nz_format(x2)) {
        static const bool is_fused_quant_matmul_weight_nz_available =
            check_aclnn_kernel_available("aclnnFusedQuantMatmulWeightNz");
        TORCH_CHECK(is_fused_quant_matmul_weight_nz_available,
                    "aclnnFusedQuantMatmulWeightNz is not available, please upgrade CANN.",
                    OPS_ERROR(ErrCode::NOT_SUPPORT));

        EXEC_NPU_CMD(aclnnFusedQuantMatmulWeightNz, x1, x2, x1_scale, x2_scale,
                     y_scale_optional, x1_offset_optional, x2_offset_optional,
                     y_offset_optional, bias.value_or(at::Tensor()), x3_optional,
                     fused_type, transpose_x1, transpose_x2, group_size, result);
    } else {
        static const bool is_fused_quant_matmul_available =
            check_aclnn_kernel_available("aclnnFusedQuantMatmul");
        TORCH_CHECK(is_fused_quant_matmul_available,
                    "aclnnFusedQuantMatmul is not available, please upgrade CANN.",
                    OPS_ERROR(ErrCode::NOT_SUPPORT));

        EXEC_NPU_CMD(aclnnFusedQuantMatmul, x1, x2, x1_scale, x2_scale,
                     y_scale_optional, x1_offset_optional, x2_offset_optional,
                     y_offset_optional, bias.value_or(at::Tensor()), x3_optional,
                     fused_type, transpose_x1, transpose_x2, group_size, result);
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
    const c10::optional<at::Tensor> &bias) {

    // 1. 参数校验
    validate_inputs(x1, x2, x1_scale, x2_scale, fused_op_type, transpose_x1, transpose_x2, bias);

    // 2. 推导输出形状
    std::vector<uint64_t> batch_record;
    uint64_t batch_val = infer_out_batch_shape_gelu(x1, x2, batch_record);
    const at::Tensor long_tensor = x1.dim() > x2.dim() ? x1 : x2;
    c10::SmallVector<int64_t, op_infer::SIZE> output_size =
        op_infer::array_to_small_vector(long_tensor.sizes());
    output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = x1.size(x1.dim() - LAST_SECOND_DIM_INDEX);
    output_size[long_tensor.dim() - 1] = x2.size(x2.dim() - 1);
    for (int64_t i = 0; i < long_tensor.dim() - LAST_SECOND_DIM_INDEX; i++) {
        output_size[i] = static_cast<int64_t>(batch_record[i]);
    }

    // 3. 分配输出张量
    at::ScalarType output_dtype = (x2_scale.dtype() == at::kBFloat16) ? at::kBFloat16 : at::kHalf;
    c10::TensorOptions options = x1.options().dtype(output_dtype);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    // 4. 调用 ACLNN 算子
    execute_aclnn_kernel(x1, x2, x1_scale, x2_scale, bias, transpose_x1, transpose_x2, fused_op_type, result);

    return result;
}

} // namespace op_api
