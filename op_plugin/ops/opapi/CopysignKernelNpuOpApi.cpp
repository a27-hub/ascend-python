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

#include <limits>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// copysign is a "binary float op" (build_binary_float_op): integral (including bool)
// inputs are promoted to the default float type, consistent with the promotion rule
// used by other binary float ops in op-plugin (e.g. xlogy).
at::ScalarType copysign_result_type(const at::Tensor& self, const at::Tensor& other) {
  at::ScalarType high_type;
  // On CPU/CUDA a 0-dim CPU tensor participates in type promotion as a "weak"
  // scalar (TensorIterator demotes it), e.g. copysign(fp32_tensor, torch.tensor(2.5))
  // stays fp32. Demote it the same way here, before it is moved to the NPU and
  // would otherwise act as a full tensor operand. IsCPUScalar must be checked
  // against the original inputs, i.e. before prepare_binary_tensors.
  if (npu_preparation::IsCPUScalar(other)) {
    high_type = at::native::result_type(self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    high_type = at::native::result_type(other, self.item());
  } else {
    high_type = at::native::result_type(self, other);
  }
  if (isIntegralType(high_type, true)) {
    return at::kFloat;
  }
  return high_type;
}

at::ScalarType copysign_result_type(const at::Tensor& self, const at::Scalar& other) {
  // Same weak-scalar semantics as upstream: a floating-point Scalar must not
  // promote a lower-precision floating tensor (copysign(fp32, 2.5) stays fp32).
  at::ScalarType high_type = at::native::result_type(self, other);
  if (isIntegralType(high_type, true)) {
    return at::kFloat;
  }
  return high_type;
}

// out=/inplace accept any out dtype that CPU accepts: everything except a
// floating result cast into an integral (or bool) target, e.g. copysign_(int,
// int) promotes to float and errors exactly like CPU
// ("result type Float can't be cast to the desired output type Int").
void copysign_check_out_dtype(at::ScalarType high_type, at::ScalarType out_type) {
  TORCH_CHECK(
      !(isFloatingType(high_type) && isIntegralType(out_type, true)),
      "result type ",
      high_type,
      " can't be cast to the desired output type ",
      out_type,
      OPS_ERROR(ErrCode::TYPE));
}

// Bit-exact IEEE-754 copysign implemented with integer ops:
//     result_bits = (self_bits & 0x7FFF...) | (other_bits & 0x8000...)
// Integer bitwise ops are exact, so NaN payloads, subnormals and the signs of
// ±0 / ±NaN are all preserved bit for bit. This is NOT achievable with the
// float-level composition (abs + signbit + neg + where): measured on real
// hardware (910B, CANN 9.1.0) aclnnSignbit(-0.0) and aclnnSignbit(-NaN) both
// return false, aclnnAbs rewrites NaN payloads and aclnnNeg fails to flip NaN
// signs, while aclnnSWhere itself is bit-exact.
at::ScalarType copysign_bits_type(at::ScalarType float_type) {
  switch (float_type) {
    case at::kDouble:
      return at::kLong;
    case at::kFloat:
      return at::kInt;
    case at::kHalf:
    case at::kBFloat16:
      return at::kShort;
    default:
      TORCH_CHECK(false, "copysign: unsupported compute dtype ", float_type, OPS_ERROR(ErrCode::TYPE));
      return at::kLong; // unreachable, silences compiler warnings
  }
}

void copysign_npu_impl(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  // self/other are already cast to the promoted float dtype; out has the same
  // element size. Three kernel launches: two ANDs + one OR writing into out.
  const auto bits_type = copysign_bits_type(self.scalar_type());
  const bool is64 = bits_type == at::kLong;
  const bool is32 = bits_type == at::kInt;
  const int64_t abs_mask = is64 ? std::numeric_limits<int64_t>::max() : (is32 ? 0x7FFFFFFFLL : 0x7FFFLL);
  const int64_t sign_mask = is64 ? std::numeric_limits<int64_t>::min()
                                 : (is32 ? static_cast<int64_t>(std::numeric_limits<int32_t>::min())
                                         : static_cast<int64_t>(std::numeric_limits<int16_t>::min()));
  auto self_bits = self.view(bits_type);
  auto other_bits = other.view(bits_type);
  auto abs_bits = self_bits.bitwise_and(abs_mask);
  auto sign_bits = other_bits.bitwise_and(sign_mask);
  auto out_bits = out.view(bits_type);
  at::bitwise_or_out(out_bits, abs_bits, sign_bits);
}
} // namespace

at::Tensor& copysign_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  auto high_type = copysign_result_type(self, other);
  copysign_check_out_dtype(high_type, out.scalar_type());
  auto [self_device, other_device] = prepare_binary_tensors(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
  npu_preparation::check_tensor({self_device, other_device}, out, out.scalar_type(), output_size);

  if (out.scalar_type() == high_type) {
    copysign_npu_impl(self_device.to(high_type), other_device.to(high_type), out);
  } else {
    // out dtype differs from the promoted dtype (CPU allows widening and
    // float narrowing): compute in high_type, then cast into out.
    at::Tensor tmp = npu_preparation::apply_tensor_without_format(
        output_size, self_device.options().dtype(high_type));
    copysign_npu_impl(self_device.to(high_type), other_device.to(high_type), tmp);
    out.copy_(tmp);
  }
  return out;
}

at::Tensor copysign(const at::Tensor& self, const at::Tensor& other) {
  auto high_type = copysign_result_type(self, other);
  auto [self_device, other_device] = prepare_binary_tensors(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
  at::Tensor result =
      npu_preparation::apply_tensor_without_format(output_size, self_device.options().dtype(high_type));

  copysign_npu_impl(self_device.to(high_type), other_device.to(high_type), result);
  return result;
}

at::Tensor& copysign_(at::Tensor& self, const at::Tensor& other) {
  auto high_type = copysign_result_type(self, other);
  copysign_check_out_dtype(high_type, self.scalar_type());
  auto [self_device, other_device] = prepare_binary_tensors(self, other);

  if (self.scalar_type() == high_type) {
    copysign_npu_impl(self_device, other_device.to(high_type), self);
  } else {
    // self dtype differs from the promoted dtype (CPU allows float narrowing,
    // e.g. copysign_(fp16_tensor, fp32_tensor)): compute in high_type, then
    // cast back into self.
    at::Tensor tmp = npu_preparation::apply_tensor_without_format(
        self_device.sizes(), self_device.options().dtype(high_type));
    copysign_npu_impl(self_device.to(high_type), other_device.to(high_type), tmp);
    self.copy_(tmp);
  }
  return self;
}

at::Tensor copysign(const at::Tensor& self, const at::Scalar& other) {
  // Convert with the promoted dtype so the Tensor path re-derives the same
  // high_type instead of treating the scalar as a full fp64 tensor operand.
  at::Tensor other_tensor =
      npu_preparation::copy_scalar_to_device(other, copysign_result_type(self, other), self.device());
  return op_api::copysign(self, other_tensor);
}

at::Tensor& copysign_(at::Tensor& self, const at::Scalar& other) {
  at::Tensor other_tensor =
      npu_preparation::copy_scalar_to_device(other, copysign_result_type(self, other), self.device());
  return op_api::copysign_(self, other_tensor);
}

at::Tensor& copysign_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
  at::Tensor other_tensor =
      npu_preparation::copy_scalar_to_device(other, copysign_result_type(self, other), self.device());
  return op_api::copysign_out(self, other_tensor, out);
}
} // namespace op_api
