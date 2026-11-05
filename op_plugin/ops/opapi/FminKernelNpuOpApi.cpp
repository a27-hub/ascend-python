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

#include <c10/core/GradMode.h>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace op_api {
namespace {
using npu_preparation = at_npu::native::OpPreparation;

at::Device fmin_device(const at::Tensor& self, const at::Tensor& other) {
  TORCH_CHECK_TYPE(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");
  const bool self_npu = torch_npu::utils::is_npu(self);
  const bool other_npu = torch_npu::utils::is_npu(other);
  TORCH_CHECK(self_npu || other_npu, "fmin expects at least one NPU tensor", OPS_ERROR(ErrCode::PARAM));
  const auto device = self_npu ? self.device() : other.device();
  for (const auto& tensor : {self, other}) {
    TORCH_CHECK(
        tensor.device() == device || (tensor.device().is_cpu() && tensor.dim() == 0),
        "Expected all tensors to be on the same device, but found ",
        tensor.device(),
        " and ",
        device,
        OPS_ERROR(ErrCode::PARAM));
  }
  return device;
}

void fmin_compute(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  if (result.numel() == 0) {
    return;
  }
  if (!c10::isFloatingType(result.scalar_type())) {
    EXEC_NPU_CMD(aclnnMinimum, self, other, result);
    return;
  }
  // fmin(a, b) = where(isnan(b) | (a < b), a, b). Self comparison
  // preserves the single-NaN rule without depending on a native Fmin API.
  auto nan_mask = npu_preparation::apply_tensor_without_format(other.sizes(), other.options().dtype(at::kBool));
  auto less_mask = npu_preparation::apply_tensor_without_format(result.sizes(), result.options().dtype(at::kBool));
  auto choose_self = npu_preparation::apply_tensor_without_format(result.sizes(), result.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnNeTensor, other, other, nan_mask);
  EXEC_NPU_CMD(aclnnLtTensor, self, other, less_mask);
  EXEC_NPU_CMD(aclnnLogicalOr, nan_mask, less_mask, choose_self);
  EXEC_NPU_CMD(aclnnSWhere, choose_self, self, other, result);
}
} // namespace

at::Tensor fmin(const at::Tensor& self, const at::Tensor& other) {
  const auto device = fmin_device(self, other);
  const c10_npu::NPUGuard device_guard(device);
  // Determine promotion before moving CPU zero-dimensional tensors: their
  // promotion priority differs from that of nonzero-dimensional tensors.
  const auto dtype = at::result_type(self, other);
  const auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  auto self_device = self.to(device, dtype);
  auto other_device = other.to(device, dtype);
  auto result = npu_preparation::apply_tensor_without_format(output_size, self_device.options());
  fmin_compute(self_device, other_device, result);
  return result;
}

at::Tensor& fmin_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  const auto device = fmin_device(self, other);
  const c10_npu::NPUGuard device_guard(device);
  const auto dtype = at::result_type(self, other);
  TORCH_CHECK(
      !at::GradMode::is_enabled() || !(self.requires_grad() || other.requires_grad() || result.requires_grad()),
      "fmin(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad.");
  TORCH_CHECK(
      result.device() == device,
      "Expected out tensor to have device ",
      device,
      ", but got ",
      result.device(),
      OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      c10::canCast(dtype, result.scalar_type()),
      "result type ",
      dtype,
      " can't be cast to the desired output type ",
      result.scalar_type(),
      OPS_ERROR(ErrCode::TYPE));
  const auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  // Check original inputs before casting: a temporary must not hide partial
  // overlap or an internally overlapping output. Exact aliases remain valid.
  npu_preparation::check_tensor({self, other}, result, output_size);
  auto value = op_api::fmin(self, other);
  result.copy_(value);
  return result;
}
} // namespace op_api
