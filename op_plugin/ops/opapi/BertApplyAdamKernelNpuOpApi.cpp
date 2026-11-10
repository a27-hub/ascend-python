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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_bert_apply_adam(
    const at::Scalar& lr,
    const at::Scalar& beta1,
    const at::Scalar& beta2,
    const at::Scalar& epsilon,
    const at::Tensor& grad,
    const at::Scalar& max_grad_norm,
    const at::Scalar& global_grad_norm,
    const at::Scalar& weight_decay,
    const c10::optional<at::Scalar>& step_size,
    int64_t adam_mode) {
  TORCH_CHECK(false, "npu_bert_apply_adam is not implemented for Tensor" + OPS_ERROR(ErrCode::PARAM));
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_bert_apply_adam_out(
    const at::Scalar& lr,
    const at::Scalar& beta1,
    const at::Scalar& beta2,
    const at::Scalar& epsilon,
    const at::Tensor& grad,
    const at::Scalar& max_grad_norm,
    const at::Scalar& global_grad_norm,
    const at::Scalar& weight_decay,
    const c10::optional<at::Scalar>& step_size,
    int64_t adam_mode,
    at::Tensor& var,
    at::Tensor& m,
    at::Tensor& v) {
  DO_COMPATIBILITY(
      aclnnApplyAdamV2,
      acl_op::npu_bert_apply_adam_out(
          lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm,
          weight_decay, step_size, adam_mode, var, m, v));

  if (!c10_npu::IsAclnnOnly()) {
    return acl_op::npu_bert_apply_adam_out(
        lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm,
        weight_decay, step_size, adam_mode, var, m, v);
  }

  const auto scalar_type = var.scalar_type();
  const auto device = var.device();
  at::Tensor lr_tensor = npu_preparation::copy_scalar_to_device(lr, scalar_type, device);
  at::Tensor beta1_tensor = npu_preparation::copy_scalar_to_device(beta1, scalar_type, device);
  at::Tensor beta2_tensor = npu_preparation::copy_scalar_to_device(beta2, scalar_type, device);
  at::Tensor epsilon_tensor = npu_preparation::copy_scalar_to_device(epsilon, scalar_type, device);
  at::Tensor max_grad_norm_tensor = npu_preparation::copy_scalar_to_device(max_grad_norm, scalar_type, device);
  at::Tensor global_grad_norm_tensor = npu_preparation::copy_scalar_to_device(global_grad_norm, scalar_type, device);
  at::Tensor weight_decay_tensor = npu_preparation::copy_scalar_to_device(weight_decay, scalar_type, device);
  c10::optional<at::Tensor> step_size_tensor;
  if (step_size.has_value()) {
    step_size_tensor = npu_preparation::copy_scalar_to_device(step_size.value(), scalar_type, device);
  } else {
    // mBART defaults to lr; Adam ignores this input. Older tiling requires its shape.
    step_size_tensor = lr_tensor;
  }
  std::string adam_mode_str = adam_mode == 0 ? "adam" : "mbart_adam";
  char* adam_mode_ptr = const_cast<char*>(adam_mode_str.c_str());

  EXEC_NPU_CMD(
      aclnnApplyAdamV2, var, m, v, lr_tensor, beta1_tensor, beta2_tensor, epsilon_tensor,
      grad, max_grad_norm_tensor, global_grad_norm_tensor, weight_decay_tensor,
      step_size_tensor, adam_mode_ptr);
  return std::tie(var, m, v);
}
} // namespace op_api
