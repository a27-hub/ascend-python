// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

static bool use_half_backward_order(const at::Tensor& grad_output, const at::Tensor& output) {
  return grad_output.scalar_type() == at::kHalf && output.scalar_type() == at::kHalf;
}

static at::Tensor sigmoid_backward_half_order(const at::Tensor& grad_output, const at::Tensor& output) {
  // PyTorch CPU/CUDA evaluates the Half expression a * (Half(1) - b) * b
  // with Half intermediate results. The fused aclnnSigmoidBackward path can
  // evaluate the whole expression in float and round only the final result.
  // For b=0.00024437904357910156 and a=1, these give 0.000244140625 and
  // 0.00024437904357910156 respectively. Keep each intermediate as a Half
  // tensor, and preserve the native subtraction/multiplication order.
  // Restrict this decomposition to Half: other dtypes retain the fused path.
  auto complement = at::sub(at::ones_like(output), output);
  auto first_product = at::mul(grad_output, complement);
  return at::mul(first_product, output);
}

at::Tensor sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& output) {
  if (use_half_backward_order(grad_output, output)) {
    return sigmoid_backward_half_order(grad_output, output);
  }
  DO_COMPATIBILITY(aclnnSigmoidBackward, acl_op::sigmoid_backward(grad_output, output));
  auto shape = op_infer::broadcast_ops_npu_output_size(grad_output, output);
  auto dtype = at::native::result_type(grad_output, output);
  auto result = npu_preparation::apply_tensor_without_format(shape, grad_output.options().dtype(dtype));
  EXEC_NPU_CMD(aclnnSigmoidBackward, grad_output, output, result);
  return result;
}

at::Tensor& sigmoid_backward_out(
    const at::Tensor& grad_output, const at::Tensor& output, at::Tensor& grad_input) {
  auto shape = op_infer::broadcast_ops_npu_output_size(grad_output, output);
  npu_preparation::check_tensor({grad_output, output}, grad_input, grad_input.scalar_type(), shape);
  if (use_half_backward_order(grad_output, output) && grad_input.scalar_type() == at::kHalf) {
    grad_input.copy_(sigmoid_backward_half_order(grad_output, output));
    return grad_input;
  }
  DO_COMPATIBILITY(aclnnSigmoidBackward, acl_op::sigmoid_backward_out(grad_output, output, grad_input));
  EXEC_NPU_CMD(aclnnSigmoidBackward, grad_output, output, grad_input);
  return grad_input;
}
} // namespace op_api
