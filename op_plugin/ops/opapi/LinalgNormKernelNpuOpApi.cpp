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

// 获取计算用的 dtype：根据 self 的类型决定
// BFloat16/Half/Float → Float（提升精度）
// Double → Double（保留）
inline c10::ScalarType get_acc_dtype(const at::Tensor& self) {
    auto scalar_type = self.scalar_type();
    if (scalar_type == c10::ScalarType::BFloat16 ||
        scalar_type == c10::ScalarType::Half ||
        scalar_type == c10::ScalarType::Float) {
        return c10::ScalarType::Float;
    }
    return scalar_type;
}

// 将 self 转换为 acc_dtype 对应的类型
inline at::Tensor convert_to_acc_dtype(const at::Tensor& tensor, c10::ScalarType dtype) {
    if (tensor.scalar_type() != dtype) {
        return tensor.to(dtype);
    }
    return tensor;
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
    
    // 1. 根据 self 的类型获取计算用的 dtype
    auto dtype = get_acc_dtype(self);
    if (opt_dtype.has_value()) {
        dtype = opt_dtype.value();
    }
    
    // 2. 将 self 转换为计算用的 dtype
    at::Tensor self_acc = convert_to_acc_dtype(self, dtype);
    
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, self_acc.options().dtype(dtype));
    EXEC_NPU_CMD(aclnnLinalgVectorNorm, self_acc, scalar_ord, dim, keepdim, dtype, out);
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
    
    // 1. 根据 self 的类型获取计算用的 dtype
    auto dtype = get_acc_dtype(self);
    if (opt_dtype.has_value()) {
        dtype = opt_dtype.value();
    }
    
    // 2. 将 self 转换为计算用的 dtype
    at::Tensor self_acc = convert_to_acc_dtype(self, dtype);
    
    // 3. 将 result 转换为计算用的 dtype
    at::Tensor result_acc = convert_to_acc_dtype(result, dtype);
    
    npu_preparation::check_tensor(
        {self_acc},
        result_acc,
        dtype,
        output_size);
    EXEC_NPU_CMD(aclnnLinalgVectorNorm, self_acc, scalar_ord, dim, keepdim, dtype, result_acc);
    
    // 4. 如果 result 原始类型和计算 dtype 不同，拷贝回原始 dtype
    if (result.scalar_type() != dtype) {
        result.copy_(result_acc);
    }
    return result;
}
}