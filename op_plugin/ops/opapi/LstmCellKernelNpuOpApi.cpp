// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

// LSTMCell operator parameter validation function
// Refer to validation logic in aclops/LstmCellKernelNpu.cpp
static void lstm_cell_param_check(const at::Tensor &input,
                                   const at::Tensor &w_ih,
                                   const at::Tensor &w_hh,
                                   const at::Tensor &h,
                                   const at::Tensor &c,
                                   const c10::optional<at::Tensor> &b_ih_opt,
                                   const c10::optional<at::Tensor> &b_hh_opt)
{
    // Check input dimension - must be 1D(batch,) or 2D(batch, input_size)
    TORCH_CHECK(input.dim() == 2 || input.dim() == 1,
        "lstm_cell: input must be 1D or 2D tensor, but got ", input.dim(), "D tensor",
        OPS_ERROR(ErrCode::PARAM));

    // Check w_ih dimension - [4*hidden_size, input_size + hidden_size]
    TORCH_CHECK(w_ih.dim() == 2,
        "lstm_cell: w_ih must be 2D tensor, but got ", w_ih.dim(), "D tensor",
        OPS_ERROR(ErrCode::PARAM));

    // Check w_hh dimension - [4*hidden_size, hidden_size]
    TORCH_CHECK(w_hh.dim() == 2,
        "lstm_cell: w_hh must be 2D tensor, but got ", w_hh.dim(), "D tensor",
        OPS_ERROR(ErrCode::PARAM));

    // Check h dimension - [batch, hidden_size]
    TORCH_CHECK(h.dim() == 2,
        "lstm_cell: h must be 2D tensor, but got ", h.dim(), "D tensor",
        OPS_ERROR(ErrCode::PARAM));
    
    // Check c dimension - [batch, hidden_size]
    TORCH_CHECK(c.dim() == 2,
        "lstm_cell: c must be 2D tensor, but got ", c.dim(), "D tensor",
        OPS_ERROR(ErrCode::PARAM));

    // Check w_ih shape: w_ih.size(0) == input_size + hidden_size
    int64_t input_size = input.dim() == 1 ? input.size(0) : input.size(1);
    int64_t hidden_size = h.size(1);
    TORCH_CHECK(w_ih.size(0) == input_size + hidden_size,
        "lstm_cell: w_ih should have ", input_size + hidden_size, " rows, but got ", w_ih.size(0),
        OPS_ERROR(ErrCode::PARAM));

    // Check w_hh shape: w_hh.size(0) == 4*hidden_size
    TORCH_CHECK(w_hh.size(0) == 4 * hidden_size,
        "lstm_cell: w_hh should have ", 4 * hidden_size, " rows, but got ", w_hh.size(0),
        OPS_ERROR(ErrCode::PARAM));

    // Check h and c shape consistency
    TORCH_CHECK(h.sizes().equals(c.sizes()),
        "lstm_cell: h and c must have the same shape, but got h:", h.sizes(), " c:", c.sizes(),
        OPS_ERROR(ErrCode::PARAM));

    // Check w_ih and w_hh column count consistency
    TORCH_CHECK(w_ih.size(1) == w_hh.size(1),
        "lstm_cell: w_ih and w_hh must have the same number of columns, but got w_ih:", w_ih.size(1),
        " w_hh:", w_hh.size(1),
        OPS_ERROR(ErrCode::PARAM));

    // Check batch size consistency
    int64_t batch_size = input.dim() == 1 ? 1 : input.size(0);
    TORCH_CHECK(h.size(0) == batch_size,
        "lstm_cell: h and input must have the same batch size, but got h:", h.size(0),
        " input:", batch_size,
        OPS_ERROR(ErrCode::PARAM));

    // Check b_ih shape
    const at::Tensor &b_ih = c10::value_or_else(b_ih_opt, [] { return at::Tensor(); });
    if (b_ih.defined()) {
        TORCH_CHECK(b_ih.dim() == 1,
            "lstm_cell: b_ih must be 1D tensor if defined, but got ", b_ih.dim(), "D tensor",
            OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(b_ih.size(0) == w_ih.size(1),
            "lstm_cell: b_ih should have ", w_ih.size(1), " elements, but got ", b_ih.size(0),
            OPS_ERROR(ErrCode::PARAM));
    }

    // Check b_hh shape
    const at::Tensor &b_hh = c10::value_or_else(b_hh_opt, [] { return at::Tensor(); });
    if (b_hh.defined()) {
        TORCH_CHECK(b_hh.dim() == 1,
            "lstm_cell: b_hh must be 1D tensor if defined, but got ", b_hh.dim(), "D tensor",
            OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(b_hh.size(0) == w_hh.size(1),
            "lstm_cell: b_hh should have ", w_hh.size(1), " elements, but got ", b_hh.size(0),
            OPS_ERROR(ErrCode::PARAM));
    }

    // Check data type consistency
    TORCH_CHECK(input.scalar_type() == w_ih.scalar_type(),
        "lstm_cell: input and w_ih must have the same dtype, but got input:", input.scalar_type(),
        " w_ih:", w_ih.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(input.scalar_type() == w_hh.scalar_type(),
        "lstm_cell: input and w_hh must have the same dtype, but got input:", input.scalar_type(),
        " w_hh:", w_hh.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(input.scalar_type() == h.scalar_type(),
        "lstm_cell: input and h must have the same dtype, but got input:", input.scalar_type(),
        " h:", h.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(input.scalar_type() == c.scalar_type(),
        "lstm_cell: input and c must have the same dtype, but got input:", input.scalar_type(),
        " c:", c.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
}

std::tuple<at::Tensor, at::Tensor> lstm_cell(const at::Tensor &input,
                                                  const at::Tensor &w_ih,
                                                  const at::Tensor &w_hh,
                                                  const at::Tensor &h,
                                                  const at::Tensor &c,
                                                  const c10::optional<at::Tensor> &b_ih_opt,
                                                  const c10::optional<at::Tensor> &b_hh_opt)
{
    // Execute parameter validation
    lstm_cell_param_check(input, w_ih, w_hh, h, c, b_ih_opt, b_hh_opt);

    DO_COMPATIBILITY(aclnnLSTMCell, acl_op::lstm_cell(input, w_ih, w_hh, h, c, b_ih_opt, b_hh_opt));

    // Get actual bias tensors
    const at::Tensor &b_ih = c10::value_or_else(b_ih_opt, [] { return at::Tensor(); });
    const at::Tensor &b_hh = c10::value_or_else(b_hh_opt, [] { return at::Tensor(); });

    // Calculate hidden size
    auto hidden_size = w_hh.size(1) / 4;
    auto output_size = std::vector<int64_t>{input.size(0), hidden_size};

    // Create output tensors
    at::Tensor h_out = npu_preparation::apply_tensor_without_format(output_size, h.options());
    at::Tensor c_out = npu_preparation::apply_tensor_without_format(output_size, c.options());

    // Call aclnnLSTMCell operator
    EXEC_NPU_CMD(aclnnLSTMCell, input, w_ih, w_hh, h, c, b_ih, b_hh, h_out, c_out);

    return std::make_tuple(h_out, c_out);
}

} // namespace op_api
