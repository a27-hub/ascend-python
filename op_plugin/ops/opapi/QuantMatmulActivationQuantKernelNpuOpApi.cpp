// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DEFAULT_SCALE_ALG = 0;
}; // namespace

static bool is_nz_format(const at::Tensor& tensor)
{
    const torch_npu::NPUStorageDesc &tensor_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ;
}

std::tuple<at::Tensor, at::Tensor> npu_quant_matmul_activation_quant(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& x1scale,
    const at::Tensor& x2scale,
    c10::string_view activation_type,
    int64_t axis,
    c10::string_view round_mode,
    const c10::optional<at::Tensor>& bias,
    c10::optional<int64_t> output_dtype,
    c10::optional<int64_t> x1_dtype,
    c10::optional<int64_t> x2_dtype,
    c10::optional<int64_t> x1scale_dtype,
    c10::optional<int64_t> x2scale_dtype,
    c10::optional<int64_t> scale_alg,
    double dst_type_max)
{
    static const bool is_kernel_available = check_aclnn_kernel_available("aclnnQuantMatmulActivationQuant");
    
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto m = x1.size(x1_dim_num - 2);
    auto k = x1.size(x1_dim_num - 1);
    auto n = x2.size(x2_dim_num - 1);
    
    c10::SmallVector<int64_t, op_infer::SIZE> output_size;
    for (int64_t i = 0; i < x1_dim_num - 2; i++) {
        output_size.push_back(x1.size(i));
    }
    output_size.push_back(m);
    output_size.push_back(n);
    
    int64_t axis_change = axis < 0 ? axis + static_cast<int64_t>(output_size.size()) : axis;
    
    c10::SmallVector<int64_t, op_infer::SIZE> scale_size = op_infer::array_to_small_vector(output_size);
    int64_t dim_size = op_infer::CeilDiv(scale_size[axis_change], BLOCK_SIZE);
    dim_size = (dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
    scale_size[axis_change] = dim_size;
    scale_size.emplace_back(ALIGN_NUM);
    
    aclDataType output_acltype;
    int64_t output_dtype_val = output_dtype.value_or(0);
    bool special_output_type = false;
    
    constexpr int64_t DTYPE_FLOAT4_E2M1 = 30;
    constexpr int64_t DTYPE_FLOAT4_E1M2 = 31;
    
    if (output_dtype_val == DTYPE_FLOAT4_E2M1 || output_dtype_val == DTYPE_FLOAT4_E1M2) {
        special_output_type = true;
    }
    
    at::Tensor output;
    if (special_output_type) {
        int64_t last_dim_val = output_size[output_size.size() - 1];
        TORCH_CHECK(last_dim_val % 2 == 0,
                    "The last dim output shape must be divisible by 2 if "
                    "output dtype is FLOAT4_E2M1 or FLOAT4_E1M2" + OPS_ERROR(ErrCode::PARAM));
        output_size[output_size.size() - 1] = last_dim_val / 2;
        output = npu_preparation::apply_tensor_without_format(output_size, c10::ScalarType::Byte);
        output_acltype = c10_npu::GetAclDataType(output_dtype_val);
    } else {
        output_acltype = c10_npu::GetAclDataType(output_dtype_val);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(output_acltype);
        output = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(scalar_dtype));
    }
    
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, 
                                                                    c10::dtype(at::ScalarType::Byte));
    
    TensorWrapper output_wrapper = {output, output_acltype};
    TensorWrapper scale_wrapper = {scale, aclDataType::ACL_FLOAT8_E8M0};
    
    const at::Tensor& bias_real = bias.value_or(at::Tensor());
    
    bool transposeX1 = false;
    bool transposeX2 = false;
    int64_t scale_alg_optional = scale_alg.has_value() ? scale_alg.value() : DEFAULT_SCALE_ALG;
    
    char *activation_type_ptr = const_cast<char *>(activation_type.data());
    char *round_mode_ptr = const_cast<char *>(round_mode.data());
    printf("x1 dims: %ld, shape: ", x1.dim());
    TensorWrapper x1_wrapper = make_wrapper(x1, x1_dtype);
    TensorWrapper x2_wrapper = make_wrapper(x2, x2_dtype);
    TensorWrapper x1_scale_wrapper = make_wrapper(x1scale, x1scale_dtype);
    TensorWrapper x2_scale_wrapper = make_wrapper(x2scale, x2scale_dtype);
    EXEC_NPU_CMD(aclnnQuantMatmulActivationQuant, 
                 x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper, 
                 bias_real, output_wrapper, scale_wrapper, 
                 transposeX1, transposeX2, activation_type_ptr, axis, 
                 round_mode_ptr, scale_alg_optional, dst_type_max);
    
    return std::make_tuple(output, scale);
}
}