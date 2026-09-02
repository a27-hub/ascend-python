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

#include <torch/library.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor to_padded_tensor(
    const at::Tensor& self,
    double padding,
    c10::optional<at::IntArrayRef> output_size) {
  // NestedTensor is not supported by NPU currently, so fall back to CPU.
  TORCH_WARN_ONCE(
      "to_padded_tensor is not supported by NPU currently. Now this kernel is running on CPU.");
  at::Tensor self_cpu = self.cpu();
  at::Tensor result_cpu = self_cpu.to_padded_tensor(padding, output_size);
  return result_cpu.to(self.device());
}

} // namespace op_api

namespace {
at::Tensor to_padded_tensor_nested_fallback(
    const at::Tensor& self,
    double padding,
    c10::OptionalArrayRef<int64_t> output_size) {
  return op_api::to_padded_tensor(self, padding, output_size);
}
} // namespace

// Register the kernel for NestedTensor on NPU. The dispatcher uses the
// NestedTensorPrivateUse1 key for nested tensors on the NPU device.
TORCH_LIBRARY_IMPL(aten, NestedTensorPrivateUse1, m) {
  m.impl("to_padded_tensor", TORCH_FN(to_padded_tensor_nested_fallback));
}
