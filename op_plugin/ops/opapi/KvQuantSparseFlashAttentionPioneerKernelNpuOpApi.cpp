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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor _npu_kv_quant_sparse_flash_attention_pioneer(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &sparse_indices,
    double scale_value,
    int64_t key_quant_mode,
    int64_t value_quant_mode,
    const c10::optional<at::Tensor> &key_dequant_scale,
    const c10::optional<at::Tensor> &value_dequant_scale,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &key_sink,
    const c10::optional<at::Tensor> &value_sink,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    int64_t quant_scale_repo_mode,
    int64_t tile_size,
    int64_t rope_head_dim,
    c10::optional<int64_t> key_dtype,
    c10::optional<int64_t> value_dtype)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.", OPS_ERROR(ErrCode::PARAM));

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    // Construct output shape
    at::SmallVector<int64_t, 5> output_size;
    if (layout_query_str == "TND") {
        TORCH_CHECK(query.dim() == 3,
            "When layout_query is TND, query must be 3D, but got ",
            query.dim(), OPS_ERROR(ErrCode::PARAM));
        int64_t out_d = query.size(2);
        if (attention_mode == 2) {
            out_d = query.size(2) - rope_head_dim;
        }
        output_size = {query.size(0), query.size(1), out_d};
    } else {
        TORCH_CHECK(query.dim() == 4,
            "When layout_query is BSND, query must be 4D, but got ",
            query.dim(), OPS_ERROR(ErrCode::PARAM));
        int64_t out_d = query.size(3);
        if (attention_mode == 2) {
            out_d = query.size(3) - rope_head_dim;
        }
        output_size = {query.size(0), query.size(1), query.size(2), out_d};
    }

    at::Tensor output = npu_preparation::apply_tensor_without_format(
        output_size, query.options().dtype(query.dtype()));

    // Convert string parameters
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    // Handle key_dtype and value_dtype for hifloat8 support
    std::variant<at::Tensor, TensorWrapper> key_processed;
    std::variant<at::Tensor, TensorWrapper> value_processed;

    if (key_dtype.has_value() &&
        c10_npu::GetAclDataType(key_dtype.value()) == ACL_HIFLOAT8) {
        key_processed = make_wrapper(key, key_dtype);
    } else {
        key_processed = key;
    }

    if (value_dtype.has_value() &&
        c10_npu::GetAclDataType(value_dtype.value()) == ACL_HIFLOAT8) {
        value_processed = make_wrapper(value, value_dtype);
    } else {
        value_processed = value;
    }

    std::visit([&](const auto &key_arg, const auto &value_arg) {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvQuantSparseFlashAttentionPioneer,
            query, key_arg, value_arg, sparse_indices, scale_value,
            key_quant_mode, value_quant_mode,
            key_dequant_scale, value_dequant_scale,
            block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
            key_sink, value_sink,
            sparse_block_size, layout_query_ptr, layout_kv_ptr,
            sparse_mode, pre_tokens, next_tokens,
            attention_mode, quant_scale_repo_mode, tile_size, rope_head_dim,
            output);
    }, key_processed, value_processed);

    return output;
}

} // namespace op_api