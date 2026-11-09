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
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// aclnnInplaceOne rejects complex dtypes. fill_ already writes 1+0j.
bool complex_ones_use_fill(const at::Tensor& tensor) {
  return at::isComplexType(tensor.scalar_type());
}
} // namespace

at::Tensor ones_like(
    const at::Tensor& self,
    c10::optional<c10::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  DO_COMPATIBILITY(
      aclnnInplaceOne,
      acl_op::ones_like(self, dtype_opt, layout_opt, device_opt, pin_memory_opt, optional_memory_format));
  auto device = c10::device_or_default(device_opt);
  if (!torch_npu::utils::is_npu(device)) {
    auto result = at::empty_like(self, dtype_opt, layout_opt, device_opt, pin_memory_opt, optional_memory_format);
    return result.fill_(1);
  }
  c10::TensorOptions options_ =
      c10::TensorOptions().dtype(dtype_opt).device(device_opt).layout(layout_opt).pinned_memory(pin_memory_opt);
  c10::TensorOptions options = self.options().merge_in(options_).merge_memory_format(optional_memory_format);
  at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), options);
  if (complex_ones_use_fill(result)) {
    result.fill_(1);
  } else {
    EXEC_NPU_CMD(aclnnInplaceOne, result);
  }
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor& one_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceOne, acl_op::one_(self));
  if (complex_ones_use_fill(self)) {
    return self.fill_(1);
  }
  EXEC_NPU_CMD(aclnnInplaceOne, self);
  return self;
}
} // namespace op_api
