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
// aclnnInplaceZero has no kernel for the unsigned 16/32/64-bit dtypes (error 161002).
// Zeroing is a pure bit pattern, so the same storage is zeroed through the signed
// dtype of equal width instead.
c10::optional<at::ScalarType> signed_alias_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kUInt16:
      return at::kShort;
    case at::kUInt32:
      return at::kInt;
    case at::kUInt64:
      return at::kLong;
    default:
      return c10::nullopt;
  }
}
} // namespace

at::Tensor& zero_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceZero, acl_op::zero_(self));
  const auto alias_dtype = signed_alias_dtype(self.scalar_type());
  if (alias_dtype.has_value()) {
    at::Tensor alias = self.view(alias_dtype.value());
    EXEC_NPU_CMD(aclnnInplaceZero, alias);
    return self;
  }
  EXEC_NPU_CMD(aclnnInplaceZero, self);
  return self;
}

} // namespace op_api
