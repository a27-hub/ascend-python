// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor construct_sparse_flash_attention_output_tensor(
    const at::Tensor& query, const at::Tensor& value, std::string layout)
{
    at::SmallVector<int64_t, SIZE> output_size;
    if (layout == "TND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    } else {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3)};
    }
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, query.options().dtype(query.dtype()));

    return output;
}

at::Tensor npu_selected_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &selected_indices, double scale_value, int64_t selected_block_size,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    // construct the output tensor
    at::Tensor selected_flash_attention_output = op_api::construct_selected_flash_attention_output_tensor(
            query, value, layout_query_str);
    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnSelectedFlashAttention, query,
        key, value, selected_indices, block_table, actual_seq_lengths_query,
        actual_seq_lengths_kv, query_rope, key_rope, scale_value, selected_block_size, 
        layout_query_ptr, layout_kv_ptr, sparse_mode,
        selected_flash_attention_output);

    return selected_flash_attention_output;
}

} // namespace op_api