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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch/csrc/autograd/custom_function.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// Composite fallback for Ascend910B and unsupported cases (matches aten composite).
std::tuple<at::Tensor, at::Tensor> thnn_fused_gru_cell_composite(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias) {
  at::Tensor igates_b = (input_bias.has_value() && input_bias.value().defined())
      ? input_gates + input_bias.value()
      : input_gates;
  at::Tensor hgates_b =
      (hidden_bias.has_value() && hidden_bias.value().defined())
      ? hidden_gates + hidden_bias.value()
      : hidden_gates;

  auto chunked_igates = igates_b.unsafe_chunk(3, 1);
  auto chunked_hgates = hgates_b.unsafe_chunk(3, 1);

  auto reset_gate = (chunked_igates[0] + chunked_hgates[0]).sigmoid_();
  auto input_gate = (chunked_igates[1] + chunked_hgates[1]).sigmoid_();
  auto new_gate = (chunked_igates[2] + reset_gate * chunked_hgates[2]).tanh_();

  auto hy = (hx - new_gate).mul_(input_gate).add_(new_gate);

  return std::make_tuple(hy, at::Tensor());
}

bool thnn_fused_gru_cell_aclnn_supported(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx) {
  static const bool is_support_soc = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950;
  static const bool is_aclnn_available = check_aclnn_kernel_available("aclnnThnnFusedGruCell");
  bool type_support = (input_gates.scalar_type() == at::kFloat) ||
      (input_gates.scalar_type() == at::kHalf) || (input_gates.scalar_type() == at::kBFloat16);
  bool rank_support = (input_gates.dim() == 2) && (hidden_gates.dim() == 2) && (hx.dim() == 2);
  return is_support_soc && is_aclnn_available && type_support && rank_support;
}

bool thnn_fused_gru_cell_need_graph(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias) {
  return at::GradMode::is_enabled() && (input_gates.requires_grad() || hidden_gates.requires_grad() ||
      hx.requires_grad() || (input_bias.has_value() && input_bias->requires_grad()) ||
      (hidden_bias.has_value() && hidden_bias->requires_grad()));
}

// Fused forward (no-graph path: inference / no_grad calls).
std::tuple<at::Tensor, at::Tensor> thnn_fused_gru_cell_aclnn(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias) {
  const int64_t batch_size = hx.size(0);
  const int64_t hidden_size = hx.size(1);
  at::Tensor hy = npu_preparation::apply_tensor_without_format(hx.sizes(), hx.options());
  at::Tensor storage =
      npu_preparation::apply_tensor_without_format({batch_size, 5 * hidden_size}, hx.options());
  EXEC_NPU_CMD(
      aclnnThnnFusedGruCell, input_gates, hidden_gates, hx, input_bias, hidden_bias, hy, storage);
  return std::make_tuple(std::move(hy), std::move(storage));
}
} // namespace

// Training path on Ascend950: hand-written autograd Function.
// Forward runs the fused kernel and saves gates + workspace.
// Backward uses the fused backward op (first-order) or the
// differentiable decomposed formula (double-backward).
class ThnnFusedGruCellAutogradFunction : public torch::autograd::Function<ThnnFusedGruCellAutogradFunction> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor input_gates,
      at::Tensor hidden_gates,
      at::Tensor hx,
      c10::optional<at::Tensor> input_bias,
      c10::optional<at::Tensor> hidden_bias) {
    at::AutoDispatchBelowADInplaceOrView guard;
    bool has_bias = input_bias.has_value() && input_bias->defined();
    auto result = thnn_fused_gru_cell_aclnn(input_gates, hidden_gates, hx, input_bias, hidden_bias);
    ctx->saved_data["has_bias"] = has_bias;
    ctx->save_for_backward({input_gates, hidden_gates, hx, std::get<1>(result),
                            input_bias.value_or(at::Tensor()), hidden_bias.value_or(at::Tensor())});
    return {std::get<0>(result), std::get<1>(result)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
    const auto& grad_hy = grad_outputs[0];
    auto saved = ctx->get_saved_variables();
    auto input_gates = saved[0];
    auto hidden_gates = saved[1];
    auto hx = saved[2];
    auto workspace = saved[3];
    auto input_bias = saved[4];
    auto hidden_bias = saved[5];
    bool has_bias = ctx->saved_data["has_bias"].toBool();

    at::Tensor grad_input_gates;
    at::Tensor grad_hidden_gates;
    at::Tensor grad_hx;
    at::Tensor grad_input_bias;
    at::Tensor grad_hidden_bias;
    if (at::GradMode::is_enabled()) {
      // double-backward: differentiable decomposed formula
      auto grads = at::_thnn_differentiable_gru_cell_backward(
          grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
      grad_input_gates = std::get<0>(grads);
      grad_hidden_gates = std::get<1>(grads);
      grad_hx = std::get<2>(grads);
      grad_input_bias = std::get<3>(grads);
      grad_hidden_bias = std::get<4>(grads);
    } else {
      // first-order: fused backward op
      auto grads = at::_thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias);
      grad_input_gates = std::get<0>(grads);
      grad_hidden_gates = std::get<1>(grads);
      grad_hx = std::get<2>(grads);
      grad_input_bias = std::get<3>(grads);
      grad_hidden_bias = std::get<4>(grads);
    }
    return {grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias, grad_hidden_bias};
  }
};

std::tuple<at::Tensor, at::Tensor> _thnn_fused_gru_cell(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias) {
  if (thnn_fused_gru_cell_aclnn_supported(input_gates, hidden_gates, hx)) {
    if (thnn_fused_gru_cell_need_graph(input_gates, hidden_gates, hx, input_bias, hidden_bias)) {
      auto out = ThnnFusedGruCellAutogradFunction::apply(
          input_gates, hidden_gates, hx, input_bias, hidden_bias);
      return std::make_tuple(out[0], out[1]);
    }
    return thnn_fused_gru_cell_aclnn(input_gates, hidden_gates, hx, input_bias, hidden_bias);
  }
  return thnn_fused_gru_cell_composite(input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

// gru_cell entry (called by torch.nn.GRUCell via _VF):
// matmul produces gates, then routes through _thnn_fused_gru_cell
// (fused on Ascend950, composite on Ascend910B).
at::Tensor gru_cell(const at::Tensor& input, const at::Tensor& hx, const at::Tensor& w_ih,
    const at::Tensor& w_hh, const c10::optional<at::Tensor>& b_ih,
    const c10::optional<at::Tensor>& b_hh) {
  auto igates = at::matmul(input, w_ih.t());
  auto hgates = at::matmul(hx, w_hh.t());
  auto result = at::_thnn_fused_gru_cell(igates, hgates, hx, b_ih, b_hh);
  return std::get<0>(result);
}
} // namespace op_api
