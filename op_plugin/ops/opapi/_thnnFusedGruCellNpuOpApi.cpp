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

#include <c10/core/GradMode.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// Workspace layout shared by the aclnn fused kernel and the composite fallback:
// 2D (B, 5H) planes [r | z | n | hx | gh_n], where r/z are the post-sigmoid gates,
// n is the post-tanh new gate and gh_n is the (biased) hidden new-gate input.
// It is consumed by _thnn_fused_gru_cell_backward only.
constexpr int64_t GRU_CELL_WORKSPACE_PLANES = 5;

// Composite fallback, identical to the aten reference (aten/src/ATen/native/RNN.cpp,
// GRUCell). The (B, 5H) workspace is built only when the autograd formula will
// record a node for this call, see thnn_fused_gru_cell_need_workspace.
std::tuple<at::Tensor, at::Tensor> thnn_fused_gru_cell_composite(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias,
    bool need_workspace)
{
    at::Tensor igates_b = (input_bias.has_value() && input_bias->defined())
        ? input_gates + input_bias.value()
        : input_gates;
    at::Tensor hgates_b = (hidden_bias.has_value() && hidden_bias->defined())
        ? hidden_gates + hidden_bias.value()
        : hidden_gates;

    auto chunked_igates = igates_b.unsafe_chunk(3, 1); // [reset, input, new]
    auto chunked_hgates = hgates_b.unsafe_chunk(3, 1);

    auto reset_gate = (chunked_igates[0] + chunked_hgates[0]).sigmoid_();
    auto input_gate = (chunked_igates[1] + chunked_hgates[1]).sigmoid_();
    auto new_gate = (chunked_igates[2] + reset_gate * chunked_hgates[2]).tanh_();

    auto hy = (hx - new_gate).mul_(input_gate).add_(new_gate);

    if (!need_workspace) {
        return std::make_tuple(hy, at::Tensor());
    }
    at::Tensor workspace = at::cat({reset_gate, input_gate, new_gate, hx, chunked_hgates[2]}, 1);
    return std::make_tuple(hy, workspace);
}

// The fused aclnn path is usable when the vendor kernel is deployed and the
// inputs match its supported dtype/rank. Kernel availability is detected by
// symbol lookup, which also covers SoCs where the fused kernel is absent.
bool thnn_fused_gru_cell_aclnn_supported(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx)
{
    static const bool is_aclnn_available = check_aclnn_kernel_available("aclnnThnnFusedGruCell");
    bool type_support = (input_gates.scalar_type() == at::kFloat) ||
        (input_gates.scalar_type() == at::kHalf) || (input_gates.scalar_type() == at::kBFloat16);
    bool rank_support = (input_gates.dim() == 2) && (hidden_gates.dim() == 2) && (hx.dim() == 2);
    return is_aclnn_available && type_support && rank_support;
}

// The workspace is consumed only by the first-order backward of the aten
// derivative formula, which records a node iff grad mode is enabled and some
// input requires grad (compute_requires_grad). The VariableType redispatch
// guard excludes the autograd dispatch keys but keeps the grad mode TLS
// untouched, so both signals are still observable here.
bool thnn_fused_gru_cell_need_workspace(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias)
{
    return at::GradMode::is_enabled() &&
        (input_gates.requires_grad() || hidden_gates.requires_grad() || hx.requires_grad() ||
        (input_bias.has_value() && input_bias->defined() && input_bias->requires_grad()) ||
        (hidden_bias.has_value() && hidden_bias->defined() && hidden_bias->requires_grad()));
}
} // namespace

// Autograd is fully owned by the aten derivative formula (derivatives.yaml):
// first-order backward dispatches to _thnn_fused_gru_cell_backward and
// double-backward to _thnn_differentiable_gru_cell_backward. Both paths of this
// kernel honor the op contract (a real workspace whenever a node is recorded),
// so no plugin-side autograd registration is needed.
//
// gru_cell is intentionally NOT registered: aten's composite implementation
// (aten/src/ATen/native/RNN.cpp GRUCell) already routes NPU tensors through
// this op via its is_privateuseone() branch, and keeps the input shape
// validation of check_rnn_cell_forward_input/hidden.
std::tuple<at::Tensor, at::Tensor> _thnn_fused_gru_cell(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias)
{
    if (thnn_fused_gru_cell_aclnn_supported(input_gates, hidden_gates, hx)) {
        // Fused path: hy plus the real (B, 5H) workspace required by the aten
        // derivative formula (first-order backward).
        const int64_t batch_size = hx.size(0);
        const int64_t hidden_size = hx.size(1);
        at::Tensor hy = npu_preparation::apply_tensor_without_format(hx.sizes(), hx.options());
        at::Tensor workspace = npu_preparation::apply_tensor_without_format(
            {batch_size, GRU_CELL_WORKSPACE_PLANES * hidden_size}, hx.options());
        EXEC_NPU_CMD(
            aclnnThnnFusedGruCell, input_gates, hidden_gates, hx, input_bias, hidden_bias, hy, workspace);
        return std::make_tuple(std::move(hy), std::move(workspace));
    }
    bool need_workspace =
        thnn_fused_gru_cell_need_workspace(input_gates, hidden_gates, hx, input_bias, hidden_bias);
    return thnn_fused_gru_cell_composite(
        input_gates, hidden_gates, hx, input_bias, hidden_bias, need_workspace);
}
} // namespace op_api
