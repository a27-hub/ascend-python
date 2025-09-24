// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

at::Tensor& index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                        const at::Tensor& value)
{
    DO_COMPATIBILITY(aclnnInplaceIndexFill, acl_op::index_fill_(self, dim, index, value));
    TORCH_CHECK(value.dim() == 0, "Value should be a 0-dimensional tensor, but got ",
                value.dim(), OPS_ERROR(ErrCode::VALUE));
    at::Scalar value_scalar = value.item();
    EXEC_NPU_CMD(aclnnInplaceIndexFill, self, dim, index, value_scalar);
    return self;
}

at::Tensor index_fill(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                      const at::Tensor& value)
{
    DO_COMPATIBILITY(aclnnIndexFill, acl_op::index_fill(self, dim, index, value));
    TORCH_CHECK(value.dim() == 0, "Value should be a 0-dimensional tensor, but got ",
                value.dim(), OPS_ERROR(ErrCode::VALUE));
    at::Scalar value_scalar = value.item();
    // construct output tensor
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
    EXEC_NPU_CMD(aclnnIndexFill, self, dim, index, value_scalar, result);
    at::namedinference::propagate_names(result, self);
    return result;
}

at::Tensor& index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                        const at::Scalar& value)
{
    DO_COMPATIBILITY(aclnnInplaceIndexFill, acl_op::index_fill_(self, dim, index, value));
    EXEC_NPU_CMD(aclnnInplaceIndexFill, self, dim, index, value);
    return self;
}

at::Tensor index_fill(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                      const at::Scalar& value)
{
    DO_COMPATIBILITY(aclnnIndexFill, acl_op::index_fill(self, dim, index, value));
    // construct output tensor
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
    EXEC_NPU_CMD(aclnnIndexFill, self, dim, index, value, result);
    at::namedinference::propagate_names(result, self);
    return result;
}
}  // namespace op_api