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

// NOTE (rms_norm design v1.1): this shell calls the CANN official aclnnRmsNorm
// interface DIRECTLY (no reuse of the custom op op_api::npu_rms_norm) to keep
// the aten open-source path independent from the custom-op implementations, and
// aligns capability/edge behavior with _fused_rms_norm_cuda
// (aten/src/ATen/native/cuda/layer_norm_kernel.cu:1996-2046).

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <limits>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> _fused_rms_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    c10::optional<double> eps) {
  // normalized_shape validation: mirrors CUDA _check_layer_norm_inputs
  // (avoids a negative-axis UB when building stat_shape below)
  TORCH_CHECK(!normalized_shape.empty() && normalized_shape.size() <= (size_t)input.dim(),
              "rms_norm(): normalized_shape must have 1 ~ input.dim() elements, but got ",
              normalized_shape.size());
  // fp64: aclnnRmsNorm does not support DOUBLE (cann-ops doc:
  // "?????DOUBLE") and the core reference (rms_norm_composite) is not
  // exported cross-DSO from libtorch_cpu — fp64 is not supported by the fused
  // path on NPU. Fail here (before backward) so training never reaches an
  // unusable backward with an opaque aclnn error.
  TORCH_CHECK(input.scalar_type() != at::kDouble,
              "_fused_rms_norm: aclnnRmsNorm does not support DOUBLE input, "
              "fp64 is not supported by the fused kernel on NPU");
  // eps default: identical to _fused_rms_norm_cuda (layer_norm_kernel.cu:2012-2017);
  // fp64 rejected above, so the remaining opmath dtype is float
  double eps_val = eps.value_or(std::numeric_limits<float>::epsilon());
  // weight: aclnnRmsNorm requires gamma to be passed; an undefined weight
  // behaves as None (CUDA borrow semantics) and is materialized as ones
  // (mathematically equivalent to no scaling).
  at::Tensor weight = (weight_opt.has_value() && weight_opt->defined())
      ? *weight_opt : at::ones(normalized_shape, input.options());
  // fold: the aclnnRmsNorm contract is 1-D gamma + >=2-D x. The master
  // cann-ops doc shows multi-dim gamma support (gamma_shape = x_shape[n:]),
  // so the fold may be dropped once verified on the target CANN build; CUDA
  // handles this inside the kernel (_check_layer_norm_inputs computes M/N).
  bool need_fold = (weight.dim() > 1) || (input.dim() == 1);
  at::Tensor x = need_fold ? input.reshape({-1, weight.numel()}) : input;
  at::Tensor gamma = weight.reshape({-1});
  const int64_t M = x.size(0), N = x.size(-1);
  // contiguity: aclnnRmsNorm rejects non-contiguous input ("????????");
  // same as CUDA expect_contiguous (zero-copy when already contiguous).
  x = *x.expect_contiguous();
  // rstd shape aligned with CUDA rstd.view(stat_shape) (kernel.cu:2035-2043):
  // stat_shape = input dims with the trailing normalized dims set to 1, for
  // ALL cases (no fold-specific shape). The aclnn contract is the same
  // (rstdOut covers the x dims minus the gamma dims).
  const int64_t axis = input.dim() - (int64_t)normalized_shape.size();
  std::vector<int64_t> stat_shape(input.sizes().begin(), input.sizes().begin() + axis);
  stat_shape.insert(stat_shape.end(), normalized_shape.size(), 1);
  std::vector<int64_t> y_sizes = need_fold ? std::vector<int64_t>{M, N} : input.sizes().vec();
  at::Tensor y = npu_preparation::apply_tensor_with_format(y_sizes, input.options(), ACL_FORMAT_ND);
  at::Tensor rstd = npu_preparation::apply_tensor_with_format(
      need_fold ? std::vector<int64_t>{M, 1} : stat_shape,
      input.options().dtype(at::kFloat), ACL_FORMAT_ND);
  // empty-input guard: same as CUDA `if (M > 0)` (kernel.cu:2028); the aclnn
  // doc states empty tensors are not supported ("???????").
  if (M > 0 && N > 0) {
    // official signature (cann-ops master):
    // aclnnRmsNormGetWorkspaceSize(x, gamma, epsilon, yOut, rstdOut, ...)
    EXEC_NPU_CMD(aclnnRmsNorm, x, gamma, eps_val, y, rstd);
  }
  if (need_fold) {
    y = y.reshape(input.sizes());
    rstd = rstd.view(stat_shape);
  }
  return std::make_tuple(y, rstd);
}

} // namespace op_api
