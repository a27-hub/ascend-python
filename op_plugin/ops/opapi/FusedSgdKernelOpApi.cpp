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
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_api_common.h"
#include <ATen/native/ForeachUtils.h>

namespace op_api {

void _fused_sgd_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const float weight_decay,
    const float momentum,
    const float lr,
    const float dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (found_inf.has_value()) {
    at::Tensor found_inf_real = found_inf.value();
    if (found_inf_real.item().toFloat() == 1) {return;}
  }

  const at::Tensor grad_scale_real = grad_scale.value_or(at::Tensor());
  TORCH_CHECK(momentum > 0,  "momentum must be positive, but got ", momentum);
  TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads}));
  if (is_first_step) {
    TORCH_WARN_ONCE(
        "`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
  }
  EXEC_NPU_CMD(aclnnFusedSgd, params, grads, momentum_buffer_list, grad_scale_real, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step);
}

void _fused_sgd_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const float weight_decay,
    const float momentum,
    const at::Tensor& lr,
    const float dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
    TORCH_CHECK(momentum > 0,  "momentum must be positive, but got ", momentum);
    TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads}));
    if (is_first_step) {
        TORCH_WARN_ONCE(
            "`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
    }
    if (grad_scale.has_value()) {
        TORCH_CHECK(
            grad_scale->device() == params[0].device(),
            "grad_scale must be on the same NPU device as the params");
    }
    if (found_inf.has_value()) {
        TORCH_CHECK(
            found_inf->device() == params[0].device(),
            "found_inf must be on the same NPU device as the params");
    }
    TORCH_CHECK(
        lr.device() == params[0].device(),
        "lr must be on the same NPU device as the params");
    if (found_inf.has_value()) {
        at::Tensor found_inf_real = found_inf.value();
        if (found_inf_real.item().toFloat() == 1) {return;}
    }
    const float lr_value = lr.item().toFloat();
    const at::Tensor grad_scale_real = grad_scale.value_or(at::Tensor());
    EXEC_NPU_CMD(aclnnFusedSgd, params, grads, momentum_buffer_list, grad_scale_real, weight_decay, momentum, lr_value, dampening, nesterov, maximize, is_first_step);
}

} // namespace op_api