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
std::tuple<at::Tensor, at::Tensor, at::Tensor> _thnn_fused_lstm_cell(const Tensor& input_gates,
                                                                     const Tensor& hidden_gates,
                                                                     const Tensor& cx,
                                                                     const c10::optional<at::Tensor> &input_bias_opt,
                                                                     const c10::optional<at::Tensor> &hidden_bias_opt)
{
    c10::MaybeOwned<Tensor> input_bias_maybe_owned = at::borrow_from_optional_tensor(input_bias_opt);
    const Tensor& input_bias = *input_bias_maybe_owned;
    const Tensor& hidden_bias = c10::value_or_else(hidden_bias_opt, [] { return at::Tensor(); });
    at::Tensor storage = at::empty_like(input_gates, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    at::Tensor hout = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    at::Tensor cout = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    // Call aclnnLSTMCell operator
    EXEC_NPU_CMD(aclnnThnnFusedLstmCell, input_gates, hidden_gates, cx, input_bias, hidden_bias, hout, cout, storage);
    return std::make_tuple(std::move(hout), std::move(cout), std::move(storage));
}
}  // namespace op_api