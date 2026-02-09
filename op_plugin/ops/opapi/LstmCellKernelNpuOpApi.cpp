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

std::tuple<at::Tensor, at::Tensor, at::Tensor> _thnn_fused_lstm_cell_backward_impl(
    const std::optional<at::Tensor>& grad_hy_opt, const std::optional<at::Tensor>& grad_cy_opt,
    const at::Tensor& cx, const at::Tensor& cy,
    const at::Tensor& workspace, bool has_bias) 
{
    std::cout << "use aclnn ThnnFusedLstmCellBackward" << std::endl;
    //grad_gates的数据类型和input保持一致，维度是batch, 4*hidden_size
    //grad_cx的数据类型和input保持一致，维度是batch, hidden_size
    //grad_bias的数据类型和input保持一致，维度是4*hidden_size

    //开始构造grad_gates, grad_cx, grad_bias
    int64_t hidden_size = cy.size(1);
    at::SmallVector<int64_t, op_infer::SIZE> grad_bias_size = {hidden_size};
    at::Tensor grad_gates = npu_preparation::apply_tensor(workspace);
    at::Tensor grad_cx = npu_preparation::apply_tensor(cx);
    at::Tensor grad_bias = npu_preparation::apply_tensor_without_format(grad_bias_size, c10::dtype(cx.scalar_type()));
    std::cout << "begin aclnnThnnFusedLstmCellBackward" << std::endl;
    EXEC_NPU_CMD(aclnnThnnFusedLstmCellBackward, grad_hy_opt, grad_cy_opt, cx, cy, workspace,
        has_bias, grad_gates, grad_cx, grad_bias);
    std::cout << "end aclnnThnnFusedLstmCellBackward" << std::endl;
    return std::tie(grad_gates, grad_cx, grad_bias);
}

} // namespace op_api

