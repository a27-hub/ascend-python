// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>> _lstm_backward_npu(
    const at::Tensor &grad_y,
    const at::Tensor &grad_hy,
    const at::Tensor &grad_cy,
    const at::Tensor &input,
    const at::TensorList hx,
    const at::TensorList params,
    const at::Tensor &i,
    const at::Tensor &j,
    const at::Tensor &f,
    const at::Tensor &o,
    const at::Tensor &h,
    const at::Tensor &c,
    const at::Tensor &tanhc,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    // std::array<bool, 3> output_mask,
    c10::optional<bool> batch_first,
    const c10::optional<at::Tensor> &batch_sizes)
{
    const bool batch_first_1 = batch_first.value_or(false);
    auto out0_shape = op_infer::lstm_backward_npu_output_size(input, batch_first_1);
    auto out1_shape = op_infer::lstm_backward_npu_hc_prev_output_size(input, params, num_layers, bidirectional, batch_first_1, batch_sizes);

    // auto out2_shape = op_infer::lstm_npu_params_output_size(params, has_biases, bidirectional);
    int64_t per_layer_params = has_biases ? 4 : 2;
    int64_t D = bidirectional ? 2 : 1;
    int64_t output_format = ACL_FORMAT_ND;
    // at::Tensor out0 = npu_preparation::apply_tensor_with_format(input, out0_shape, output_format);
    // at::Tensor out_h_prev = npu_preparation::apply_tensor_with_format(input, out1_shape, output_format);
    // at::Tensor out_c_prev = npu_preparation::apply_tensor_with_format(input, out1_shape, output_format);
    at::Tensor out0 = npu_preparation::apply_tensor(input, out0_shape);
    at::Tensor out_h_prev = npu_preparation::apply_tensor(input, out1_shape);
    at::Tensor out_c_prev = npu_preparation::apply_tensor(input, out1_shape);
    // int64_t list_length = per_layer_params * num_layers * D;
    std::vector<at::Tensor> param_list;
    // param_list.reserve(list_length);

    for (int64_t idx = 0; idx < params.size(); ++idx) {
        auto i_tensor = npu_preparation::apply_tensor_with_format(input, params[idx].sizes(), output_format);
        param_list.emplace_back(std::move(i_tensor));
    }
    
    // 使用chunk操作将tensor转换为tensorlist
    int64_t list_length = D * num_layers;
    at::TensorList i_list = at::TensorList(at::chunk(i, list_length, 0));
    at::TensorList j_list = at::TensorList(at::chunk(j, list_length, 0));
    at::TensorList f_list = at::TensorList(at::chunk(f, list_length, 0));
    at::TensorList o_list = at::TensorList(at::chunk(o, list_length, 0));
    at::TensorList h_list = at::TensorList(at::chunk(h, list_length, 0));
    at::TensorList c_list = at::TensorList(at::chunk(c, list_length, 0));
    at::TensorList tanhc_list = at::TensorList(at::chunk(tanhc, list_length, 0));
    
    at::TensorList param_list_ = at::TensorList(param_list);
    std::nullptr_t temp_nullptr = nullptr;
    EXEC_NPU_CMD(
        aclnnLstmBackward,
        input,
        hx,
        params,
        grad_y,
        grad_hy,
        grad_cy,
        i_list,
        j_list,
        f_list,
        o_list,
        h_list,
        c_list,
        tanhc_list,
        batch_sizes,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first_1,
        temp_nullptr,
        out0,
        out_h_prev,
        out_c_prev,
        param_list_);

    return std::make_tuple(
        out0,
        out_h_prev,
        out_c_prev,
        param_list_.vec());
}

}
