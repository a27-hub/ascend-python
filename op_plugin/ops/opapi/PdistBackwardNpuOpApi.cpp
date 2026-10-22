// Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
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

namespace op_api {

// _pdist_backward has no dedicated aclnn interface. pdist is the cdist of
// self with itself packed over the strict upper triangle, so scatter the
// packed grad/pdist into symmetric (n, n) matrices and reuse the cdist
// backward with x1 == x2 == self: the row sums of
// _cdist_backward(G, self, self, p, D) are exactly the pdist gradient,
// because G is symmetric with a zero diagonal.
at::Tensor _pdist_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    double p,
    const at::Tensor& pdist) {
  TORCH_CHECK(p >= 0, "pdist only supports non-negative p values", OPS_ERROR(ErrCode::VALUE));

  // The gradient vanishes when there is no pair to accumulate (non-2D input
  // or n <= 1), when p == 0 (piecewise-constant forward) or when the feature
  // dimension is empty.
  if (self.dim() != 2 || self.size(0) <= 1 || p == 0.0 || self.size(1) == 0) {
    return at::zeros_like(self);
  }
  const int64_t n = self.size(0);

  // triu_indices has no NPU kernel; build the small index metadata on the
  // CPU and copy it to the device.
  auto cpu_options = at::TensorOptions().dtype(at::kLong).device(at::Device(at::kCPU));
  auto idx = at::triu_indices(n, n, 1, cpu_options).to(self.device());

  // Make self contiguous once: the acl_op fallback of _cdist_backward (used
  // on CANN < 9.2.0) only accepts contiguous inputs.
  auto self_contiguous = self.contiguous();

  auto grad_upper = at::zeros({n, n}, self_contiguous.options());
  grad_upper.index_put_({idx.select(0, 0), idx.select(0, 1)}, grad_output);
  auto grad_matrix = grad_upper + grad_upper.mT();

  auto dist_upper = at::zeros({n, n}, self_contiguous.options());
  dist_upper.index_put_({idx.select(0, 0), idx.select(0, 1)}, pdist);
  auto dist_matrix = dist_upper + dist_upper.mT();

  return op_api::_cdist_backward(grad_matrix, self_contiguous, self_contiguous, p, dist_matrix);
}
} // namespace op_api