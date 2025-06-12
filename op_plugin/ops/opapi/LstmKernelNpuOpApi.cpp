// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "torch_npu/csrc/aten/CustomFunctions.h"

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;


std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(const at::Tensor &input, at::TensorList hx, at::TensorList params,
                                                    bool has_biases, int64_t num_layers, double dropout, bool train,
                                                    bool bidirectional, bool batch_first)
{
    auto input_trans = batch_first ? input.transpose(0, 1) : input;
    int64_t input_size = params[0].size(1);
    int64_t hidden_size = params[1].size(1);
    at::Tensor y = npu_preparation::apply_tensor(input, {input_trans.size(0), input_trans.size(1), (bidirectional ? 2 : 1) * hidden_size});
    at::Tensor h = npu_preparation::apply_tensor(input, {(bidirectional ? 2 : 1) * num_layers, input_trans.size(1), hidden_size});
    at::Tensor c = npu_preparation::apply_tensor(input, {(bidirectional ? 2 : 1) * num_layers, input_trans.size(1), hidden_size});

    at::Tensor no_tensor = at::Tensor();
    bool packed = false;
    c10::optional<at::IntArrayRef> no_array;
    if (!bidirectional) {
        if (!has_biases) {
            EXEC_NPU_CMD(aclnnBidirectionLSTMV2, input_trans, hx[0], hx[1], params[0], params[1], no_tensor, no_tensor,no_tensor,no_tensor,no_tensor,no_tensor,no_tensor,
            num_layers, has_biases, batch_first, bidirectional, packed, y, h, c);
        }
        else {
            EXEC_NPU_CMD(aclnnBidirectionLSTMV2, input_trans, hx[0], hx[1], params[0], params[1], params[2], params[3],no_tensor,no_tensor,no_tensor,no_tensor,no_tensor,
            num_layers, has_biases, batch_first, bidirectional, packed, y, h, c);
        }
    } else {
        if (!has_biases) {
            EXEC_NPU_CMD(aclnnBidirectionLSTMV2, input_trans, hx[0], hx[1], params[0], params[1], no_tensor, no_tensor,params[2], params[3],no_tensor,no_tensor,no_tensor,
            num_layers, has_biases, batch_first, bidirectional, packed, y, h, c);
        }
        else {
            EXEC_NPU_CMD(aclnnBidirectionLSTMV2, input_trans, hx[0], hx[1], params[0], params[1], params[2], params[3],params[4], params[5],params[6], params[7],no_tensor,
            num_layers, has_biases, batch_first, bidirectional, packed, y, h, c);
        }
    }

    auto output = batch_first ? y.transpose(0, 1) : y;
    return std::tie(output, h, c);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(const at::Tensor &data, const at::Tensor &batch_sizes,
                                                    at::TensorList hx, at::TensorList params, bool has_biases,
                                                    int64_t num_layers, double dropout_p, bool train,
                                                    bool bidirectional)
{
    int64_t hidden_size = hx[0].size(2);
    at::Tensor y = npu_preparation::apply_tensor(data, {data.size(0), (bidirectional ? 2 : 1) * hidden_size});
    at::Tensor h = npu_preparation::apply_tensor(hx[0]);
    at::Tensor c = npu_preparation::apply_tensor(hx[1]);

    bool batch_first = false;
    bool packed = true;
    at::Tensor no_tensor = at::Tensor();
    at::Tensor batch_sizes_npu = npu_preparation::copy_tensor_host_to_device(batch_sizes.to(at::kInt));
    if (!bidirectional) {
        if (!has_biases) {
            EXEC_NPU_CMD(aclnnBidirectionLSTMV2, data, hx[0], hx[1], params[0], params[1], no_tensor, no_tensor,no_tensor,no_tensor,no_tensor,no_tensor,batch_sizes_npu,
            num_layers, has_biases, batch_first, bidirectional, packed, y, h, c);
        }
        else {
            EXEC_NPU_CMD(aclnnBidirectionLSTMV2, data, hx[0], hx[1], params[0], params[1], params[2], params[3],no_tensor,no_tensor,no_tensor,no_tensor,batch_sizes_npu,
            num_layers, has_biases, batch_first, bidirectional, packed, y, h, c);
        }
    } else {
        if (!has_biases) {
            EXEC_NPU_CMD(aclnnBidirectionLSTMV2, data, hx[0], hx[1], params[0], params[1], no_tensor, no_tensor,params[2], params[3],no_tensor,no_tensor,batch_sizes_npu,
            num_layers, has_biases, batch_first, bidirectional, packed, y, h, c);
        }
        else {
            EXEC_NPU_CMD(aclnnBidirectionLSTMV2, data, hx[0], hx[1], params[0], params[1], params[2], params[3],params[4], params[5],params[6], params[7],batch_sizes_npu,
            num_layers, has_biases, batch_first, bidirectional, packed, y, h, c);
        }
    }

    return std::tie(y, h, c);
}
} // namespace op_api
