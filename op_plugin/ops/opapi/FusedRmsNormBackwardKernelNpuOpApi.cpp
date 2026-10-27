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
// capability/edge behavior with _fused_rms_norm_backward_cuda.

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
  // fp64: DOUBLE is not in the aclnnRmsNormGrad dtype matrix (CANN docs)
  TORCH_CHECK(input.scalar_type() != at::kDouble,
              "_fused_rms_norm_backward: aclnnRmsNormGrad does not support DOUBLE input, "
              "fp64 is not supported by the fused backward kernel on NPU");
  at::Tensor weight = (weight_opt.has_value() && weight_opt->defined())
      ? *weight_opt : at::ones(normalized_shape, input.options());
  bool need_fold = (weight.dim() > 1) || (input.dim() == 1);
  at::Tensor dy = need_fold ? grad_out.reshape({-1, weight.numel()}) : grad_out;
  at::Tensor x = need_fold ? input.reshape({-1, weight.numel()}) : input;
  at::Tensor gamma = weight.reshape({-1});
  const int64_t M = x.size(0), N = x.size(-1);
  dy = *dy.expect_contiguous();
  x = *x.expect_contiguous();
  at::Tensor rstd_kernel = need_fold ? rstd.reshape({M, 1}) : rstd;
  at::Tensor dx = at::empty_like(x);
  // aclnnRmsNormGrad accumulates dgamma in fp32 (official dtype matrix:
  // dgamma is always FLOAT32).
  at::Tensor dgamma = at::empty({gamma.numel()}, x.options().dtype(at::kFloat));

  if (x.numel() > 0) {
    EXEC_NPU_CMD(aclnnRmsNormGrad, dy, x, rstd_kernel, gamma, dx, dgamma);
  }
  at::Tensor dx_out;
  if (output_mask[0]) {
    dx_out = need_fold ? dx.reshape(input.sizes()) : dx;
  }
  at::Tensor dgamma_out = at::Tensor();
  if (output_mask[1]) {
    at::Tensor dg = (x.numel() > 0) ? dgamma.to(weight.scalar_type())
                                    : at::zeros_like(weight);
    if (need_fold) {
      dg = dg.reshape(weight.sizes());
    }
    dgamma_out = dg;
  }
  return std::tuple<at::Tensor, at::Tensor>(dx_out, dgamma_out);
}

} // namespace op_api
