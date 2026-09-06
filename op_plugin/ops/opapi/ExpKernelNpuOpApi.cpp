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

#include <ATen/TensorIterator.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor exp(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnExp, acl_op::exp(self));
  // Allocate the output through TensorIterator so its strides follow the
  // memory-dense layout of the input, matching ATen native semantics.
  // Pre-created empty contiguous output; see eq() for the rationale.
  // Like CPU, integral inputs (incl. bool) promote to float; aclnnExp does
  // not accept every integral input dtype, so cast them up front.
  at::Tensor self_input = self;
  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    self_input = self.to(at::kFloat);
  }
  at::Tensor result = at::empty(0, self_input.options());
  at::TensorIterator::unary_float_op(result, self_input);
  EXEC_NPU_CMD(aclnnExp, self_input, result);
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor& exp_out(const at::Tensor& self, at::Tensor& out) {
  DO_COMPATIBILITY(aclnnExp, acl_op::exp_out(self, out));
  // Build the iterator to validate/resize the output with ATen-compatible
  // memory-dense strides. Integral inputs promote to float like CPU.
  at::Tensor self_input = self;
  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    self_input = self.to(at::kFloat);
  }
  at::TensorIterator::unary_float_op(out, self_input);
  EXEC_NPU_CMD(aclnnExp, self_input, out);
  at::namedinference::propagate_names(out, self);
  return out;
}
} // namespace op_api
