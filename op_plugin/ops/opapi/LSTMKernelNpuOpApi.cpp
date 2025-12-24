// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> _lstm_npu(
    const at::Tensor &input,
    const at::TensorList hx,
    const at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    c10::optional<bool> batch_first,
    const c10::optional<at::Tensor> &batch_sizes)
{
    const bool batch_first_1 = batch_first.value_or(false);
    auto out0_shape = op_infer::lstm_npu_output_size(input, params, bidirectional, batch_first_1);
    auto ijfo_tanhc_shapes = op_infer::lstm_npu_ijfo_tanhc_output_size(input, params, num_layers, train, bidirectional, batch_first_1);
    auto hc_shapes = op_infer::lstm_npu_hc_output_size(input, params, num_layers, train, bidirectional, batch_first_1);

    int64_t D = bidirectional ? 2 : 1;
    const at::Tensor &batch_sizes_1 = batch_sizes.value_or(at::Tensor());
    int64_t output_format = ACL_FORMAT_ND;

    at::Tensor out0 = npu_preparation::apply_tensor_with_format(input, out0_shape, output_format);
    int64_t list_length = D * num_layers;
    std::vector<at::Tensor> i_list;
    std::vector<at::Tensor> j_list;
    std::vector<at::Tensor> f_list, 
    std::vector<at::Tensor> o_list;
    std::vector<at::Tensor> tanh_list;
    std::vector<at::Tensor> h_list;
    std::vector<at::Tensor> c_list;
    h_list.reserve(list_length);
    c_list.reserve(list_length);

    at::TensorList i_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList j_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList f_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList o_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList tanh_list_ = c10::ArrayRef<at::Tensor>();
    if (train) {
        i_list.reserve(list_length);
        j_list.reserve(list_length);
        f_list.reserve(list_length);
        o_list.reserve(list_length);
        tanh_list.reserve(list_length);

        for (int64_t idx = 0; idx < list_length; ++idx) {
            auto i_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_tanhc_shapes, output_format);
            i_list.emplace_back(std::move(i_tensor));

            auto j_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_tanhc_shapes, output_format);
            j_list.emplace_back(std::move(j_tensor));

            auto f_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_tanhc_shapes, output_format);
            f_list.emplace_back(std::move(f_tensor));

            auto o_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_tanhc_shapes, output_format);
            o_list.emplace_back(std::move(o_tensor));

            auto tanh_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_tanhc_shapes, output_format);
            tanh_list.emplace_back(std::move(tanh_tensor));

            auto h_tensor = npu_preparation::apply_tensor_with_format(input, hc_shapes, output_format);
            h_list.emplace_back(std::move(h_tensor));

            auto c_tensor = npu_preparation::apply_tensor_with_format(input, hc_shapes, output_format);
            c_list.emplace_back(std::move(c_tensor));
        }
        i_list_ = at::TensorList(i_list);
        j_list_ = at::TensorList(j_list);
        f_list_ = at::TensorList(f_list);
        o_list_ = at::TensorList(o_list);
        tanh_list_ = at::TensorList(tanh_list);
    } else {
        for (int64_t idx = 0; idx < list_length; ++idx) {
            auto h_tensor = npu_preparation::apply_tensor_with_format(input, hc_shapes, output_format);
            h_list.emplace_back(std::move(h_tensor));
            auto c_tensor = npu_preparation::apply_tensor_with_format(input, hc_shapes, output_format);
            c_list.emplace_back(std::move(c_tensor));
        }
    }

    at::TensorList h_list_ = at::TensorList(h_list);
    at::TensorList c_list_ = at::TensorList(c_list);

    EXEC_NPU_CMD(
        aclnnLSTM,
        input,
        params,
        hx,
        batch_sizes,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first_1,
        out0,
        i_list_,
        j_list_,
        f_list_,
        o_list_,
        h_list_,
        c_list_,
        tanh_list_);

    return std::make_tuple(
        out0,
        i_list_.vec(),
        j_list_.vec(),
        f_list_.vec(),
        o_list_.vec(),
        h_list_.vec(),
        c_list_.vec(),
        tanh_list_.vec());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(
    const at::Tensor& input, const at::TensorList hx,
    const at::TensorList params, bool has_biases,
    int64_t num_layers, double dropout,
    bool train, bool bidirectional, bool batch_first)
{
    auto output = op_api::_lstm_npu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    std::vector<at::Tensor> output_h_list = std::get<5>(output); // 5 for output
    std::vector<at::Tensor> output_c_list = std::get<6>(output); // 6 for output
    std::vector<at::Tensor> output_h_list_res;
    std::vector<at::Tensor> output_c_list_res;
    at::Tensor final_output_h;
    at::Tensor final_output_c;
    if (train) {
        int64_t dscale = bidirectional ? 2 : 1;
        for (int64_t i = 0; i < num_layers; ++i) {
            int64_t layer_idx = dscale * i;
            at::Tensor h_layer = output_h_list[layer_idx];
            at::Tensor c_layer = output_c_list[layer_idx];
            at::Tensor h_last_step = h_layer.index({-1, at::indexing::Ellipsis});
            at::Tensor c_last_step = c_layer.index({-1, at::indexing::Ellipsis});
            output_h_list_res.emplace_back(h_last_step);
            output_c_list_res.emplace_back(c_last_step);
            if (bidirectional){
                int64_t layer_idx_1 = dscale * i + 1;
                at::Tensor h_layer = output_h_list[layer_idx_1];
                at::Tensor c_layer = output_c_list[layer_idx_1];
                at::Tensor h_first_step = h_layer.index({0, at::indexing::Ellipsis});
                at::Tensor c_first_step = c_layer.index({0, at::indexing::Ellipsis});
                output_h_list_res.emplace_back(h_first_step);
                output_c_list_res.emplace_back(c_first_step);
            }
        }
        final_output_h = output_h_list_res.empty() ? at::Tensor() : at::stack(output_h_list_res, 0);
        final_output_c = output_c_list_res.empty() ? at::Tensor() : at::stack(output_c_list_res, 0);
    } else {
        final_output_h = at::stack(output_h_list, 0);
        final_output_c = at::stack(output_c_list, 0);
    }
    return std::make_tuple(std::get<0>(output), final_output_h, final_output_c);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(
    const at::Tensor& data, const at::Tensor& batch_sizes, const at::TensorList hx,
    const at::TensorList params, bool has_biases,
    int64_t num_layers, double dropout,
    bool train, bool bidirectional)
{
    auto output = op_api::_lstm_npu(data, hx, params, has_biases, num_layers, dropout, train, bidirectional, false, batch_sizes);
    std::vector<at::Tensor> output_h_list = std::get<5>(output); // 5 for output
    std::vector<at::Tensor> output_c_list = std::get<6>(output); // 6 for output
    std::vector<at::Tensor> output_h_list_res;
    std::vector<at::Tensor> output_c_list_res;
    at::Tensor final_output_h;
    at::Tensor final_output_c;
    if (train) {
        int64_t dscale = bidirectional ? 2 : 1;
        for (int64_t i = 0; i < num_layers; ++i) {
            int64_t layer_idx = dscale * i;
            at::Tensor h_layer = output_h_list[layer_idx];
            at::Tensor c_layer = output_c_list[layer_idx];
            at::Tensor h_last_step = h_layer.index({-1, at::indexing::Ellipsis});
            at::Tensor c_last_step = c_layer.index({-1, at::indexing::Ellipsis});
            output_h_list_res.emplace_back(h_last_step);
            output_c_list_res.emplace_back(c_last_step);
            if (bidirectional){
                int64_t layer_idx_1 = dscale * i + 1;
                at::Tensor h_layer = output_h_list[layer_idx_1];
                at::Tensor c_layer = output_c_list[layer_idx_1];
                at::Tensor h_first_step = h_layer.index({0, at::indexing::Ellipsis});
                at::Tensor c_first_step = c_layer.index({0, at::indexing::Ellipsis});
                output_h_list_res.emplace_back(h_first_step);
                output_c_list_res.emplace_back(c_first_step);
            }
        }
        final_output_h = output_h_list_res.empty() ? at::Tensor() : at::stack(output_h_list_res, 0);
        final_output_c = output_c_list_res.empty() ? at::Tensor() : at::stack(output_c_list_res, 0);
    } else {
        final_output_h = at::stack(output_h_list, 0);
        final_output_c = at::stack(output_c_list, 0);
    }
    return std::make_tuple(std::get<0>(output), final_output_h, final_output_c);
}
}
