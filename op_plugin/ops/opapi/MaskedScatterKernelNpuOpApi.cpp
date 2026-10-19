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

namespace {
void check_source_capacity(const at::Tensor& mask, const at::Tensor& source) {
  // Fast path: the number of masked positions never exceeds mask.numel(),
  // so most valid calls return here without any device-to-host sync.
  if (source.numel() >= mask.numel()) {
    return;
  }
  // Slow path: count the masked positions exactly (one device-to-host sync).
  // Mirrors the CPU semantics of torch.masked_scatter_, which raises
  // "Number of elements of source < number of ones in mask" instead of
  // silently reading out-of-bounds device memory (see issue #602).
  int64_t masked_count = (mask != 0).sum().item<int64_t>();
  TORCH_CHECK(
      source.numel() >= masked_count,
      "Number of elements of source < number of ones in mask. Expected source to have at least ",
      masked_count,
      " elements, but got ",
      source.numel());
}
} // namespace

at::Tensor& masked_scatter_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& source) {
  DO_COMPATIBILITY(aclnnInplaceMaskedScatter, acl_op::masked_scatter_(self, mask, source));

  check_source_capacity(mask, source);

  at_npu::native::OpPreparation::check_memory({self, mask, source}, {self});

  EXEC_NPU_CMD(aclnnInplaceMaskedScatter, self, mask, source);
  return self;
}
} // namespace op_api
