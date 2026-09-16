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
using npu_utils = at_npu::native::NpuUtils;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> _thnn_fused_gru_cell_backward(
    const at::Tensor& grad_hy, const at::Tensor& workspace, bool has_bias)
{
    TORCH_CHECK(c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
        "_thnn_fused_gru_cell_backward is not supported on this SoC, it is supported only on the "
        "Ascend950 platform and after", OPS_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(workspace.defined(),
        "workspace is undefined: fused GRU backward requires the fused forward with a real workspace",
        OPS_ERROR(ErrCode::PARAM));
    static const bool fused_available = check_aclnn_kernel_available("aclnnThnnFusedGruCellBackward");
    TORCH_CHECK(fused_available,
        "aclnnThnnFusedGruCellBackward is not available: check the ThnnFusedGruCellGrad vendor package "
        "installation", OPS_ERROR(ErrCode::NOT_SUPPORT));

    // Matches aten checkGRUBackwardSizes (RNN.cu:628): workspace must be 2D with size (B, 5H)
    TORCH_CHECK(grad_hy.dim() == 2 && workspace.dim() == 2 && workspace.size(0) == grad_hy.size(0) &&
        workspace.size(1) == grad_hy.size(1) * 5,
        "expected workspace to be 2D with size (B, 5H) = (", grad_hy.size(0), ", ", grad_hy.size(1) * 5,
        ") to match grad_hy, but got ", workspace.sizes(), OPS_ERROR(ErrCode::PARAM));

    auto grad_hy_cont = npu_utils::format_contiguous(grad_hy);
    auto storage = npu_utils::format_contiguous(workspace);
    const auto batch_size = storage.size(0);
    const auto hidden_size = storage.size(1) / 5;

    at::Tensor grad_input_gates = at::empty({batch_size, 3 * hidden_size}, storage.options());
    at::Tensor grad_hidden_gates = at::empty({batch_size, 3 * hidden_size}, storage.options());
    at::Tensor grad_hx = at::empty({batch_size, hidden_size}, storage.options());
    // has_bias=false: bias outputs are undefined (matches CUDA RNN.cu:661-663).
    // Undefined tensors are converted to nullptr by the EXEC_NPU_CMD layer
    // (op_api_common.cpp ConvertType), and the aclnn L2 skips bias slot
    // checks entirely when hasBias=false.
    at::Tensor grad_input_bias = has_bias ? at::empty({3 * hidden_size}, storage.options()) : at::Tensor();
    at::Tensor grad_hidden_bias = has_bias ? at::empty({3 * hidden_size}, storage.options()) : at::Tensor();

    EXEC_NPU_CMD(aclnnThnnFusedGruCellBackward, grad_hy_cont, storage, has_bias, grad_input_gates, grad_hidden_gates, grad_hx,
        grad_input_bias, grad_hidden_bias);

    return std::make_tuple(std::move(grad_input_gates), std::move(grad_hidden_gates), std::move(grad_hx),
        std::move(grad_input_bias), std::move(grad_hidden_bias));
}
} // namespace op_api
