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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// GRU 主输出 output：(seq_len, batch, D*hidden_size)，按 batch_first 决定前两维顺序。
// 对照 lstm_npu_output_size，唯一差别：GRU 3 个门(r/z/n) → /3，LSTM 4 个门 → /4。
c10::SmallVector<int64_t, SIZE> gru_output_size(const at::Tensor &input, at::TensorList params,
                                                bool bidirectional, bool batch_first)
{
    const at::Tensor &w_ih = params[0];
    int64_t D = bidirectional ? 2 : 1;
    int64_t hidden_size = w_ih.size(0) / 3;
    int64_t batch_size = batch_first ? input.size(0) : input.size(1);
    int64_t time_step = batch_first ? input.size(1) : input.size(0);
    if (batch_first) {
        return {batch_size, time_step, D * hidden_size};
    }
    return {time_step, batch_size, D * hidden_size};
}

// GRU hy(h_n)：(num_layers*D, batch, hidden_size)，对照 lstm_npu_output1_2_size。
c10::SmallVector<int64_t, SIZE> gru_hy_size(const at::Tensor &input, at::TensorList params,
                                            int64_t num_layers, bool bidirectional, bool batch_first)
{
    const at::Tensor &w_ih = params[0];
    int64_t D = bidirectional ? 2 : 1;
    int64_t hidden_size = w_ih.size(0) / 3;
    int64_t batch_size = batch_first ? input.size(0) : input.size(1);
    return {D * num_layers, batch_size, hidden_size};
}

// 中间态 r/z/n/h 单元素 shape：(seq_len, batch, hidden_size)，对照
// lstm_npu_ijfo_hc_tanhc_output_size。
c10::SmallVector<int64_t, SIZE> gru_gate_single_size(const at::Tensor &input, at::TensorList params, bool batch_first)
{
    const at::Tensor &w_ih = params[0];
    int64_t hidden_size = w_ih.size(0) / 3;
    int64_t batch_size = batch_first ? input.size(0) : input.size(1);
    int64_t time_step = batch_first ? input.size(1) : input.size(0);
    return {time_step, batch_size, hidden_size};
}
} // namespace

std::tuple<at::Tensor, at::Tensor> gru(const at::Tensor &input, const at::Tensor &hx, at::TensorList params,
                                       bool has_biases, int64_t num_layers, double dropout, bool train,
                                       bool bidirectional, bool batch_first)
{
    // 当前 CANN 找不到 aclnnGRU 符号则降级到老的 DynamicGRUV2 实现。
    DO_COMPATIBILITY(aclnnGRU,
        acl_op::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first));

    int64_t D = bidirectional ? 2 : 1;
    int64_t list_length = D * num_layers;
    int64_t output_format = ACL_FORMAT_ND;

    at::Tensor out_y =
        npu_preparation::apply_tensor(input, gru_output_size(input, params, bidirectional, batch_first));
    at::Tensor hy =
        npu_preparation::apply_tensor(input, gru_hy_size(input, params, num_layers, bidirectional, batch_first));

    // batchSizes：gru.input schema 没有（属 gru.data packed 模式），传空 tensor。
    at::Tensor batch_sizes = at::Tensor();

    // rOut/zOut/nOut/hOut：aclnnGRU 为反向准备的中间态，schema 只返回 (output, h_n)，用完即弃。
    std::vector<at::Tensor> r_vec, z_vec, n_vec, h_vec;
    if (train) {
        auto single_shape = gru_gate_single_size(input, params, batch_first);
        r_vec.reserve(list_length);
        z_vec.reserve(list_length);
        n_vec.reserve(list_length);
        h_vec.reserve(list_length);
        for (int64_t idx = 0; idx < list_length; ++idx) {
            r_vec.emplace_back(npu_preparation::apply_tensor_with_format(input, single_shape, output_format));
            z_vec.emplace_back(npu_preparation::apply_tensor_with_format(input, single_shape, output_format));
            n_vec.emplace_back(npu_preparation::apply_tensor_with_format(input, single_shape, output_format));
            h_vec.emplace_back(npu_preparation::apply_tensor_with_format(input, single_shape, output_format));
        }
    }
    at::TensorList r_list(r_vec);
    at::TensorList z_list(z_vec);
    at::TensorList n_list(n_vec);
    at::TensorList h_list(h_vec);

    // 参数顺序严格对齐 aclnnGRUGetWorkspaceSize：
    // input, hx, params, batchSizes, hasBias, numlayers, dropout, train, bidirectional, batchFirst,
    // output, hy, rOut, zOut, nOut, hOut。
    EXEC_NPU_CMD(aclnnGRU,
        input, hx, params, batch_sizes,
        has_biases, num_layers, dropout, train, bidirectional, batch_first,
        out_y, hy,
        r_list, z_list, n_list, h_list);

    return std::make_tuple(out_y, hy);
}
} // namespace op_api
