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

#include <variant>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

// hifloat8 dtype value: 290 (ACL_HIFLOAT8 + 256)
constexpr int64_t HIFLOAT8_DTYPE_VALUE = 290;

// Check if dtype is hifloat8
bool IsHifloat8Dtype(c10::optional<int64_t> dtype)
{
    return dtype.has_value() && dtype.value() == HIFLOAT8_DTYPE_VALUE;
}

// 工具函数，推导输出shape
at::Tensor construct_quant_lightning_indexer_output_tensor(const at::Tensor& query, const at::Tensor& key,
                                                           int64_t sparse_count, std::string query_layout_str,
                                                           std::string key_layout_str)
{
    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);
    int64_t keyHeadNum = (key_layout_str == "TND")? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparse_count};
    } else {
        output_size = {query.size(DIM_0), keyHeadNum, sparse_count};
    }
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, query.options().dtype(at::kInt));

    return output;
}

at::Tensor npu_quant_lightning_indexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    int64_t query_quant_mode, int64_t key_quant_mode,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    c10::optional<int64_t> query_dtype,
    c10::optional<int64_t> key_dtype)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    // 根据设计文档要求：
    // - 如果是 hifloat8 类型，使用 TensorWrapper 将 dtype 显式写入 tensor
    // - 如果不是 hifloat8 类型，保持原本逻辑（直接使用原始 tensor）
    std::variant<at::Tensor, TensorWrapper> query_processed;
    std::variant<at::Tensor, TensorWrapper> key_processed;

    if (IsHifloat8Dtype(query_dtype)) {
        query_processed = make_wrapper(query, query_dtype);
    } else {
        query_processed = query;
    }

    if (IsHifloat8Dtype(key_dtype)) {
        key_processed = make_wrapper(key, key_dtype);
    } else {
        key_processed = key;
    }

    // construct the output tensor
    at::Tensor quant_lightning_indexer_output = construct_quant_lightning_indexer_output_tensor(
        query, key, sparse_count, query_layout_str, key_layout_str);
    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    // 使用 std::visit 处理 variant 类型，根据实际类型调用 aclnn 接口
    std::visit([&](const auto& query_arg, const auto& key_arg) {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnQuantLightningIndexer, query_arg, key_arg,
            weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query, actual_seq_lengths_key,
            block_table, query_quant_mode, key_quant_mode, query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode,
            pre_tokens, next_tokens, quant_lightning_indexer_output);
    }, query_processed, key_processed);

    return quant_lightning_indexer_output;
}

} // namespace op_api