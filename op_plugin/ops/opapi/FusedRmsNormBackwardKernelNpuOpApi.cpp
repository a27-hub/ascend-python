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

// NOTE (rms_norm design v1.1): this shell calls the CANN official
// aclnnRmsNormGrad interface DIRECTLY (no reuse of the custom op
// op_api::npu_rms_norm_backward) for aten-path independence, and aligns
// capability/edge behavior with _fused_rms_norm_backward_cuda
// (aten/src/ATen/native/cuda/layer_norm_kernel.cu:2049-2100).

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor> _fused_rms_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    std::array<bool, 2> output_mask) {
  // fp64: DOUBLE is not in the aclnnRmsNormGrad dtype matrix (CANN docs) —
  // the same capability boundary as the LayerNorm/GroupNorm backward shells.
  // fp64 is rejected at the forward shell (see _fused_rms_norm) so training
  // never reaches an unusable backward; direct primitive calls get a readable
  // error here as well.
  TORCH_CHECK(input.scalar_type() != at::kDouble,
              "_fused_rms_norm_backward: aclnnRmsNormGrad does not support DOUBLE input, "
              "fp64 is not supported by the fused backward kernel on NPU");
  // weight: dgamma shape/dtype follow gamma (CUDA: empty_like(gamma), and
  // zeros_like(gamma) when M == 0 — kernel.cu:2079-2092).
  at::Tensor weight = (weight_opt.has_value() && weight_opt->defined())
      ? *weight_opt : at::ones(normalized_shape, input.options());
  // fold consistent with the forward; the aclnnRmsNormGrad contract is 1-D
  // gamma + >=2-D dy/x (same as the forward).
  bool need_fold = (weight.dim() > 1) || (input.dim() == 1);
  at::Tensor dy = need_fold ? grad_out.reshape({-1, weight.numel()}) : grad_out;
  at::Tensor x = need_fold ? input.reshape({-1, weight.numel()}) : input;
  at::Tensor gamma = weight.reshape({-1});
  const int64_t M = x.size(0), N = x.size(-1);
  // contiguity: same as the forward (aclnn rejects non-contiguous input).
  dy = *dy.expect_contiguous();
  x = *x.expect_contiguous();
  // rstd restored to the kernel-expected shape: the forward returns stat_shape
  // (aligned with _fused_rms_norm_cuda rstd.view(stat_shape)); the kernel wants
  // (M, 1) for folded inputs — reshape is zero-copy (contiguous, same numel).
  at::Tensor rstd_kernel = need_fold ? rstd.reshape({M, 1}) : rstd;
  // output buffers: aclnnRmsNormGrad computes both outputs (interface is
  // fixed), so both buffers are allocated; masked-out outputs are returned as
  // undefined afterwards (CUDA semantics: masked outputs stay undefined and
  // the kernel skips them — same observable contract, interface-limited extra
  // compute on NPU).
  at::Tensor dx = at::empty_like(x);
  // aclnnRmsNormGrad accumulates dgamma in fp32 (official dtype matrix:
  // dgamma is always FLOAT32).
  at::Tensor dgamma = at::empty({gamma.numel()}, x.options().dtype(at::kFloat));
  // empty-input guard: same as CUDA `if (M > 0 && N > 0)` (kernel.cu:2095).
  // M == 0 edge: dgamma must be zeros (CUDA zeros_like) to avoid polluting
  // gradients; dx stays the (empty) buffer.
  if (M > 0 && N > 0) {
    // official signature (cann-ops master):
    // aclnnRmsNormGradGetWorkspaceSize(dy, x, rstd, gamma, dx, dgamma, ...)
    // NOTE: rstd comes BEFORE gamma (the legacy custom-op shell passed
    // (dy, x, gamma, rstd), which is the same-source parameter-order bug).
    EXEC_NPU_CMD(aclnnRmsNormGrad, dy, x, rstd_kernel, gamma, dx, dgamma);
  }
  // masked-out dx stays undefined (CUDA contract, LayerNorm/GroupNorm backward
  // shell precedent) — keeps the "Meta = real execution" invariant with the
  // registered backward Meta (which returns None when output_mask[0] is false)
  at::Tensor dx_out;
  if (output_mask[0]) {
    dx_out = need_fold ? dx.reshape(input.sizes()) : dx;
  }
  at::Tensor dgamma_out = at::Tensor();
  if (output_mask[1]) {
    at::Tensor dg = (M > 0) ? dgamma.to(weight.scalar_type())
                            : at::zeros_like(weight);
    if (need_fold) {
      dg = dg.reshape(weight.sizes());
    }
    dgamma_out = dg;
  }
  return std::tuple<at::Tensor, at::Tensor>(dx_out, dgamma_out);
}

} // namespace op_api
