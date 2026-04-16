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

#include <torch/extension.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
using namespace at_npu::native;
constexpr int64_t OUTPUT_SIZE = 1024;

const c10::optional<at::Tensor> qli_get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device) {
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_quant_lightning_indexer_metadata(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, int64_t query_quant_mode, int64_t key_quant_mode, 
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_key, 
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k, const c10::string_view layout_query, 
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, 
    int64_t cmp_ratio)
{
    const c10::string_view device = "npu:0";
    at::Device output_device = at::Device(std::string(device));
    if (actual_seq_lengths_query.has_value()) {
        output_device = actual_seq_lengths_query.value().device();
    } else if (actual_seq_lengths_key.has_value()) {
        output_device = actual_seq_lengths_key.value().device();
    }

    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto actual_seq_lengths_query_val = qli_get_valid_tensor(actual_seq_lengths_query, output_device);
    auto actual_seq_lengths_key_val = qli_get_valid_tensor(actual_seq_lengths_key, output_device);

    // convert str
    std::string layout_query_str = std::string(layout_query);
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    std::string layout_key_str = std::string(layout_key);
    char *layout_key_ptr = const_cast<char *>(layout_key_str.c_str());

    EXEC_NPU_CMD_V1(aclnnQuantLightningIndexerMetadata, actual_seq_lengths_query_val, actual_seq_lengths_key_val,
                    num_heads_q, num_heads_k, head_dim, query_quant_mode, key_quant_mode, batch_size, 
                    max_seqlen_q, max_seqlen_k, layout_query_ptr, layout_key_ptr, sparse_count, 
                    sparse_mode, pre_tokens, next_tokens, cmp_ratio, output);
    return output;
}
} // namespace op_api

namespace npu {
at::Tensor npu_quant_lightning_indexer_metadata_fallback_None(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, int64_t query_quant_mode, int64_t key_quant_mode, 
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_key, 
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k, const c10::string_view layout_query, 
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, 
    int64_t cmp_ratio)
{
    at::Tensor output = op_api::npu_quant_lightning_indexer_metadata(
        num_heads_q, num_heads_k, head_dim, query_quant_mode, key_quant_mode, actual_seq_lengths_query, 
        actual_seq_lengths_key, batch_size, max_seqlen_q, max_seqlen_k, layout_query, layout_key, sparse_count, 
        sparse_mode, pre_tokens, next_tokens, cmp_ratio);
    return output;
}
}

TORCH_LIBRARY_IMPL(npu, BackendSelect, m) {
    m.impl("npu_quant_lightning_indexer_metadata", &npu::npu_quant_lightning_indexer_metadata_fallback_None);
}