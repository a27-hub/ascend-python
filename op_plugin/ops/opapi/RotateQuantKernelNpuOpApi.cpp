// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
using npu_preparation = at_npu::native::OpPreparation;

namespace {

constexpr int64_t INT4_IN_INT32_NUM = 8;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t A5_SCALE_ALIGN = 64;
constexpr int64_t AXIS_LAST_DIM = -1;
constexpr int64_t DEFAULT_Y_DTYPE_INT8 = static_cast<int64_t>(at::ScalarType::Char);
constexpr int64_t QUINT4X2_DTYPE = static_cast<int64_t>(at::ScalarType::QUInt4x2);
const std::set<std::string> VALID_ROUND_MODES = {"rint", "round", "floor"};

bool is_a5_output_dtype(int64_t y_dtype)
{
    // A5 output dtypes: FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2
    // A2/A3 output dtypes: INT8, INT32 (INT4 packed)
    return (y_dtype == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
            y_dtype == static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN) ||
            y_dtype == static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2));
}

} // namespace

std::tuple<at::Tensor, at::Tensor> npu_rotate_quant(
    const at::Tensor& x,
    const at::Tensor& rot,
    const c10::optional<at::Tensor>& alpha,
    c10::optional<int64_t> y_dtype,
    int64_t axis,
    c10::string_view round_mode,
    int64_t scale_alg,
    double dst_type_max,
    bool trans)
{
    // 1. Validate input dimensions
    TORCH_CHECK(x.dim() >= 1 && x.dim() <= 7,
        "x dim should be in range [1, 7], but got ", x.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(rot.dim() >= 2 && rot.dim() <= 3,
        "rot dim should be in range [2, 3], but got ", rot.dim(), OPS_ERROR(ErrCode::PARAM));

    // 2. Validate x and rot dtypes match
    TORCH_CHECK(x.scalar_type() == rot.scalar_type(),
        "x and rot must have same dtype, but got ", x.scalar_type(), " and ", rot.scalar_type(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 || x.scalar_type() == at::ScalarType::Half,
        "x dtype must be BFLOAT16 or FLOAT16, but got ", x.scalar_type(), OPS_ERROR(ErrCode::PARAM));

    // 3. Validate rot last two dims are square
    TORCH_CHECK(rot.size(rot.dim() - 1) == rot.size(rot.dim() - 2),
        "rot last two dims must be square, but got (", rot.size(rot.dim() - 2), ", ",
        rot.size(rot.dim() - 1), ")", OPS_ERROR(ErrCode::PARAM));

    // 4. Validate axis
    TORCH_CHECK(axis == AXIS_LAST_DIM || axis == x.dim() - 1,
        "axis only supports -1 or D-1, but got ", axis, OPS_ERROR(ErrCode::PARAM));

    // 5. Validate round_mode
    std::string round_mode_str(round_mode);
    TORCH_CHECK(VALID_ROUND_MODES.find(round_mode_str) != VALID_ROUND_MODES.end(),
        "round_mode must be one of \"rint\", \"round\", \"floor\", but got ", round_mode_str,
        OPS_ERROR(ErrCode::PARAM));

    // 6. Validate scale_alg
    TORCH_CHECK(scale_alg >= 0 && scale_alg <= 2,
        "scale_alg must be 0, 1, or 2, but got ", scale_alg, OPS_ERROR(ErrCode::PARAM));

    // 7. Validate trans
    TORCH_CHECK(!trans,
        "trans only supports false currently", OPS_ERROR(ErrCode::PARAM));

    // 8. Handle optional alpha
    const at::Tensor alpha_val = alpha.has_value() ? alpha.value() : at::Tensor();

    // 9. Determine output dtype
    int64_t y_dtype_val = y_dtype.has_value() ? y_dtype.value() : DEFAULT_Y_DTYPE_INT8;

    // 10. Calculate output shapes
    auto y_shape = op_infer::array_to_small_vector(x.sizes());

    // Handle INT4 packed output (INT4 -> int32, last_dim // 8)
    bool is_int4 = (y_dtype_val == static_cast<int64_t>(c10_npu::DType::INT4) ||
                    y_dtype_val == QUINT4X2_DTYPE);
    if (is_int4) {
        TORCH_CHECK(y_shape.back() % INT4_IN_INT32_NUM == 0,
            "For INT4 output, x last dim must be divisible by 8, but got ", y_shape.back(),
            OPS_ERROR(ErrCode::PARAM));
        y_shape.back() /= INT4_IN_INT32_NUM;
    }

    // 11. Create y tensor
    at::Tensor y;
    aclDataType y_acltype;
    bool is_a5 = is_a5_output_dtype(y_dtype_val);
    if (is_a5) {
        y_acltype = c10_npu::GetAclDataType(y_dtype_val);
        if (y_dtype_val == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1)) {
            // FLOAT4_E2M1 stored as uint8, 2 values per byte
            TORCH_CHECK(y_shape.back() % ALIGN_NUM == 0,
                "For FLOAT4 output, x last dim must be divisible by 2, but got ", y_shape.back(),
                OPS_ERROR(ErrCode::PARAM));
            y_shape.back() /= ALIGN_NUM;
            y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(at::ScalarType::Byte));
        } else {
            at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
            y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
        }
    } else {
        // A2/A3 mode: INT8 or INT32
        if (is_int4) {
            // INT4 packed as INT32
            y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(at::ScalarType::Int));
            y_acltype = aclDataType::ACL_INT32;
        } else {
            // Default INT8
            y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(at::ScalarType::Char));
            y_acltype = aclDataType::ACL_INT8;
        }
    }

    // 12. Calculate scale shape and allocate
    at::Tensor scale;
    aclDataType scale_acltype;
    if (is_a5) {
        // A5 scale shape: (*, CeilDiv(N, 64), 2), dtype: FLOAT8_E8M0
        auto scale_shape = op_infer::array_to_small_vector(x.sizes());
        int64_t n = scale_shape.back();
        int64_t ceil_div = op_infer::CeilDiv(n, A5_SCALE_ALIGN);
        scale_shape.back() = ceil_div;
        scale_shape.emplace_back(ALIGN_NUM);
        scale = npu_preparation::apply_tensor_without_format(scale_shape, c10::dtype(at::ScalarType::Byte));
        scale_acltype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        // A2/A3 scale shape: (M,), dtype: FLOAT
        at::SmallVector<int64_t, op_infer::SIZE> scale_shape;
        scale_shape.push_back(x.size(0));
        scale = npu_preparation::apply_tensor_without_format(scale_shape, c10::dtype(at::ScalarType::Float));
        scale_acltype = aclDataType::ACL_FLOAT;
    }

    // 13. Call aclnnRotateQuant
    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper scale_wrapper = {scale, scale_acltype};
    char* round_mode_ptr = const_cast<char*>(round_mode_str.c_str());

    EXEC_NPU_CMD(aclnnRotateQuant, x, rot, alpha_val, axis, round_mode_ptr, scale_alg,
                 dst_type_max, trans, y_wrapper, scale_wrapper);

    return std::make_tuple(y, scale);
}

} // namespace op_api
