// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
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

  auto chunked_igates = igates_b.unsafe_chunk(3, 1); // [reset, input, new]
  auto chunked_hgates = hgates_b.unsafe_chunk(3, 1);

  auto reset_gate = (chunked_igates[0] + chunked_hgates[0]).sigmoid_();
  auto input_gate = (chunked_igates[1] + chunked_hgates[1]).sigmoid_();
  auto new_gate = (chunked_igates[2] + reset_gate * chunked_hgates[2]).tanh_();

  auto hy = (hx - new_gate).mul_(input_gate).add_(new_gate);

  return std::make_tuple(hy, at::Tensor());
}
} // namespace

std::tuple<at::Tensor, at::Tensor> _thnn_fused_gru_cell(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias) {
  static bool isAscend950 = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950;
  static const bool is_aclnn_available = check_aclnn_kernel_available("aclnnThnnFusedGruCell");
  bool typeSupport = (input_gates.scalar_type() == at::kFloat) ||
                     (input_gates.scalar_type() == at::kHalf) ||
                     (input_gates.scalar_type() == at::kBFloat16);
  bool rankSupport =
      (input_gates.dim() == 2) && (hidden_gates.dim() == 2) && (hx.dim() == 2);
  if (isAscend950 && is_aclnn_available && typeSupport && rankSupport) {
    const int64_t batch_size = hx.size(0);
    const int64_t hidden_size = hx.size(1);
    at::Tensor hy = npu_preparation::apply_tensor_without_format(hx.sizes(), hx.options());
    at::Tensor storage =
        npu_preparation::apply_tensor_without_format({batch_size, 5 * hidden_size}, hx.options());
    EXEC_NPU_CMD(
        aclnnThnnFusedGruCell, input_gates, hidden_gates, hx, input_bias, hidden_bias, hy, storage);
    return std::make_tuple(hy, storage);
  } else {
    return thnn_fused_gru_cell_composite(input_gates, hidden_gates, hx, input_bias, hidden_bias);
  }
}
} // namespace op_api
