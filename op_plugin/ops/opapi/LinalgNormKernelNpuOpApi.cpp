// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
inline bool check_use_aclop(const at::Scalar& scalar_ord)
{
    if (c10_npu::IsAclnnOnly()) {
        return false;
    }
    if (op_plugin::utils::is_gte_cann_version_900()) {
        return false;
    }
    float val = op_plugin::utils::get_scalar_float_value(scalar_ord);
    if (op_plugin::utils::is_gte_cann_version_810rc1() && (val == INFINITY || val == -INFINITY)) {
        return false;
    }
    return val != 0.0 && val != 1.0 && val != 2.0 && val != 3.0;
}

// 转换 scalar_ord 为 float（double → float）
inline at::Scalar convert_ord_to_float(const at::Scalar& scalar_ord) {
    if (scalar_ord.type() == c10::ScalarType::Double) {
        double val = scalar_ord.toDouble();
        return at::Scalar(static_cast<float>(val));
    }
    return scalar_ord;
}

// 将 double tensor 转换为 float tensor
inline at::Tensor convert_tensor_to_float(const at::Tensor& tensor) {
    if (tensor.scalar_type() == c10::ScalarType::Double) {
        return tensor.to(c10::ScalarType::Float);
    }
    return tensor;
}

// 获取合适的 dtype（将 double 也转为 float）
inline c10::ScalarType get_valid_dtype(
    const at::Tensor& self,
    at::optional<at::ScalarType> opt_dtype) {
    auto dtype = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
    // BFloat16/Half/Float/Double → Float
    if (dtype == c10::ScalarType::BFloat16 ||
        dtype == c10::ScalarType::Half ||
        dtype == c10::ScalarType::Float ||
        dtype == c10::ScalarType::Double) {
        return c10::ScalarType::Float;
    }
    return dtype;
}
}

at::Tensor linalg_vector_norm(
    const at::Tensor& self,
    const at::Scalar& scalar_ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype)
{
    if (check_use_aclop(scalar_ord)) {
        return acl_op::linalg_vector_norm(self, scalar_ord, opt_dim, keepdim, opt_dtype);
    }

    DO_COMPATIBILITY(aclnnLinalgVectorNorm,
                     acl_op::linalg_vector_norm(self, scalar_ord, opt_dim, keepdim, opt_dtype));
    auto dim = opt_dim.value_or(at::IntArrayRef{});
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    
    // 1. 获取有效的 dtype（BFloat16/Half/Float/Double → Float）
    auto dtype = get_valid_dtype(self, opt_dtype);
    
    // 2. 转换 scalar_ord 为 float（double → float）
    at::Scalar converted_ord = convert_ord_to_float(scalar_ord);
    
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(dtype));
    EXEC_NPU_CMD(aclnnLinalgVectorNorm, self, converted_ord, dim, keepdim, dtype, out);
    return out;
}

at::Tensor& linalg_vector_norm_out(
    const at::Tensor& self,
    const at::Scalar& scalar_ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype,
    at::Tensor& result)
{
    if (check_use_aclop(scalar_ord)) {
        return acl_op::linalg_vector_norm_out(self, scalar_ord, opt_dim, keepdim, opt_dtype, result);
    }

    DO_COMPATIBILITY(aclnnLinalgVectorNorm,
                     acl_op::linalg_vector_norm_out(self, scalar_ord, opt_dim, keepdim, opt_dtype, result));
    auto dim = opt_dim.value_or(at::IntArrayRef{});
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    
    // 1. 获取有效的 dtype（BFloat16/Half/Float/Double → Float）
    auto dtype = get_valid_dtype(self, opt_dtype);
    
    // 2. 转换 scalar_ord 为 float（double → float）
    at::Scalar converted_ord = convert_ord_to_float(scalar_ord);
    
    // 3. 如果 result 是 double，也转成 float
    at::Tensor result_float = result;
    if (result.scalar_type() == c10::ScalarType::Double) {
        result_float = result.to(c10::ScalarType::Float);
    }
    
    npu_preparation::check_tensor(
        {self},
        result_float,
        dtype,
        output_size);
    EXEC_NPU_CMD(aclnnLinalgVectorNorm, self, converted_ord, dim, keepdim, dtype, result_float);
    
    // 如果 result 原本是 double，把结果转回 double
    if (result.scalar_type() == c10::ScalarType::Double) {
        result.copy_(result_float);
    }
    return result;
}
}