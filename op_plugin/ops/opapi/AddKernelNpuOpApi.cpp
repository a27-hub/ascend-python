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

inline void alpha_check_npu(const at::ScalarType dtype, at::Scalar alpha) {
  TORCH_CHECK(
      !alpha.isBoolean() || dtype == at::ScalarType::Bool,
      "Boolean alpha only supported for Boolean results." +
          OPS_ERROR(ErrCode::TYPE));
  TORCH_CHECK(
      isFloatingType(dtype) || isComplexType(dtype) || alpha.isIntegral(true),
      "For integral input tensors, argument alpha must not be a floating point number." +
          OPS_ERROR(ErrCode::TYPE));
}

static at::Tensor self_tensor_to_device(
    const at::Tensor& tensor,
    const at::ScalarType result_type,
    const c10::Device device) {
  if (npu_preparation::is_scalar_wrapped_to_tensor(tensor) ||
      (tensor.dim() == 0 && !torch_npu::utils::is_npu(tensor))) {
    at::Scalar scalar = tensor.item();
    return npu_preparation::copy_scalar_to_device(scalar, result_type, device);
  }
  return tensor;
}

// Boolean add on the CPU reference is a logical or: True + True is True
// (and alpha, which alpha_check_npu restricts to integral values for bool
// tensors, acts as self OR (alpha AND other)).  The arithmetic aclnn
// kernels instead write the integer sum into the bool bytes (True + True
// becomes 2), so boolean results must route to the logical kernels.
static inline bool add_bool_alpha_is_zero(const at::Scalar& alpha) {
  alpha_check_npu(at::kBool, alpha);
  return !alpha.to<bool>();
}

static void check_bool_add_inputs(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Tensor& result) {
  TORCH_CHECK(at::canCast(at::native::result_type(self, other), at::kBool),
      "result type cannot be cast to Bool" + OPS_ERROR(ErrCode::TYPE));
  TORCH_CHECK(
      (self.device() == result.device() || npu_preparation::IsCPUScalar(self)) &&
      (other.device() == result.device() || npu_preparation::IsCPUScalar(other)),
      "Expected all tensors to be on the same device" + OPS_ERROR(ErrCode::PARAM));
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  TORCH_CHECK(at::IntArrayRef(output_size) == result.sizes(),
      "output shape does not match the broadcast shape" + OPS_ERROR(ErrCode::PARAM));
}

static at::Tensor& add_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& result) {
  if (at::native::result_type(self, other) == at::kBool &&
      result.scalar_type() != at::kBool) {
    auto bool_result = npu_preparation::apply_tensor_without_format(
        result.sizes(), result.options().dtype(at::kBool));
    add_out_npu_nocheck(self, other, alpha, bool_result);
    result.copy_(bool_result);
    return result;
  }
  if (result.scalar_type() == at::kBool) {
    check_bool_add_inputs(self, other, result);
    if (add_bool_alpha_is_zero(alpha)) {
      // self + 0 * other == self (a zero alpha turns the right operand
      // into False and the or degenerates to a broadcast copy).
      result.copy_(self);
      return result;
    }
    at::Tensor self_b =
        (self.dim() == 0 && !torch_npu::utils::is_npu(self)) ?
        npu_preparation::copy_scalar_to_device(
            self.item(), at::kBool, result.device()) :
        self;
    at::Tensor other_b =
        (other.dim() == 0 && !torch_npu::utils::is_npu(other)) ?
        npu_preparation::copy_scalar_to_device(
            other.item(), at::kBool, result.device()) :
        other;
    EXEC_NPU_CMD(aclnnLogicalOr, self_b, other_b, result);
    return result;
  }
  // executing the NPU operator
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    c10::Scalar others = other.item();
    EXEC_NPU_CMD(aclnnAdds, self, others, alpha, result);
  } else {
    if (self.dim() == 0 && !torch_npu::utils::is_npu(self)) {
      // self is a scalar.
      static const bool is_aclnn_available =
          check_aclnn_kernel_available("aclnnAddV3");
      if (is_aclnn_available && self.dtype() != at::kBool && self.dtype() != at::kLong) {
        c10::Scalar selfs = self.item();
        EXEC_NPU_CMD(aclnnAddV3, selfs, other, alpha, result);
      } else {
        at::Tensor self_cp =
            self_tensor_to_device(self, result.scalar_type(), result.device());
        EXEC_NPU_CMD(aclnnAdd, self_cp, other, alpha, result);
      }
    } else {
      // self and other are all npu tensors.
      EXEC_NPU_CMD(aclnnAdd, self, other, alpha, result);
    }
  }
  return result;
}

static at::Tensor& inplace_add_out_npu_no_check(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  if (self.scalar_type() == at::kBool) {
    check_bool_add_inputs(self, other, self);
    // in-place boolean add is logical_or_ (True + True is True on the CPU
    // reference); the arithmetic in-place kernels write the integer sum.
    if (!add_bool_alpha_is_zero(alpha)) {
      if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        auto cp_other = npu_preparation::copy_scalar_to_device(
            other.item(), at::kBool, self.device());
        EXEC_NPU_CMD(aclnnInplaceLogicalOr, self, cp_other);
      } else {
        EXEC_NPU_CMD(aclnnInplaceLogicalOr, self, other);
      }
    }
    return self;
  }
  // check if other scalar tensor
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    c10::Scalar other_scalar = other.item();
    EXEC_NPU_CMD(aclnnInplaceAdds, self, other_scalar, alpha);
  } else {
    EXEC_NPU_CMD(aclnnInplaceAdd, self, other, alpha);
  }
  return self;
}

static at::Tensor add_dest_output(
    const at::Tensor& self,
    const at::Tensor& other) {
  bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
  return isSelfWrapped ? other : self;
}

at::Tensor add(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnAdd, acl_op::add(self, other, alpha));
  DO_COMPATIBILITY(aclnnAdds, acl_op::add(self, other, alpha));
  if (at::native::result_type(self, other) == at::kBool) {
    DO_COMPATIBILITY(aclnnLogicalOr, acl_op::add(self, other, alpha));
  }
  std::vector<at::Tensor> tensor_list = {self, other};
  auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
  // calculate the output size
  at::Tensor output_tensor = add_dest_output(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  alpha_check_npu(result_type, alpha);
  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(
      output_size, output_tensor.options().dtype(result_type));
  // calculate the output result of the NPU
  add_out_npu_nocheck(self, other, alpha, result);
  at::namedinference::propagate_names_if_nonempty(result, maybe_names);
  return result;
}

at::Tensor add(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnAdds, acl_op::add(self, other, alpha));
  if (at::native::result_type(self, other) == at::kBool) {
    DO_COMPATIBILITY(aclnnLogicalOr, acl_op::add(self, other, alpha));
  }
  // calculate the output size
  auto output_size = op_infer::input_same_output_size(self);
  at::ScalarType result_type = at::native::result_type(self, other);
  alpha_check_npu(result_type, alpha);
  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(
      output_size, self.options().dtype(result_type));
  // calculate the output result of the NPU
  if (result.scalar_type() == at::kBool) {
    // boolean add is a logical or (True + True is True); a zero alpha
    // reduces the sum to a copy of self
    if (add_bool_alpha_is_zero(alpha)) {
      result.copy_(self);
    } else {
      at::Tensor other_b = npu_preparation::copy_scalar_to_device(
          other, at::kBool, result.device());
      EXEC_NPU_CMD(aclnnLogicalOr, self, other_b, result);
    }
  } else {
    EXEC_NPU_CMD(aclnnAdds, self, other, alpha, result);
  }
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor& add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAdd, acl_op::add_out(self, other, alpha, result));
  DO_COMPATIBILITY(aclnnAdds, acl_op::add_out(self, other, alpha, result));
  if (at::native::result_type(self, other) == at::kBool) {
    DO_COMPATIBILITY(aclnnLogicalOr, acl_op::add_out(self, other, alpha, result));
  }
  std::vector<at::Tensor> tensor_list = {self, other};
  auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
  bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
  at::Tensor output_tensor = isSelfWrapped ? other : self;
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);

  npu_preparation::check_tensor({self}, result, result, output_size);
  npu_preparation::check_memory({self, other}, {result});
  add_out_npu_nocheck(self, other, alpha, result);
  at::namedinference::propagate_names_if_nonempty(result, maybe_names);
  return result;
}

at::Tensor& add_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnInplaceAdd, acl_op::add_(self, other, alpha));
  DO_COMPATIBILITY(aclnnInplaceAdds, acl_op::add_(self, other, alpha));
  if (self.scalar_type() == at::kBool) {
    DO_COMPATIBILITY(aclnnInplaceLogicalOr, acl_op::add_(self, other, alpha));
  }
  std::vector<at::Tensor> tensor_list = {self, other};
  auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
  npu_preparation::check_memory({self, other}, {self});
  inplace_add_out_npu_no_check(self, other, alpha);
  at::namedinference::propagate_names_if_nonempty(self, maybe_names);
  return self;
}

at::Tensor& add_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnInplaceAdds, acl_op::add_(self, other, alpha));
  if (self.scalar_type() == at::kBool) {
    DO_COMPATIBILITY(aclnnInplaceLogicalOr, acl_op::add_(self, other, alpha));
    TORCH_CHECK(at::canCast(at::native::result_type(self, other), at::kBool),
        "result type cannot be cast to Bool" + OPS_ERROR(ErrCode::TYPE));
    // boolean in-place add is logical_or_ (True + True is True)
    if (!add_bool_alpha_is_zero(alpha)) {
      at::Tensor other_b = npu_preparation::copy_scalar_to_device(
          other, at::kBool, self.device());
      EXEC_NPU_CMD(aclnnInplaceLogicalOr, self, other_b);
    }
    return self;
  }
  EXEC_NPU_CMD(aclnnInplaceAdds, self, other, alpha);
  return self;
}
} // namespace op_api
