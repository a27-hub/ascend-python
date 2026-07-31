// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include <cstring>

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const int LAYOUT_MAX_LENGTH = 20;

namespace {

// Convert optional tensor to IntArrayRef for seq-length parameters.
// Returns empty IntArrayRef when tensor is not defined.
at::IntArrayRef OptionalTensorToIntArrayRef(
    const c10::optional<at::Tensor>& opt_tensor,
    std::vector<int64_t>& buffer)
{
    if (!opt_tensor.has_value() || !opt_tensor.value().defined()) {
        return at::IntArrayRef{};
    }
    const at::Tensor& t = opt_tensor.value();
    TORCH_CHECK(t.is_cpu(), "Sequence length tensor must be on CPU", OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(t.dim() == 1, "Sequence length tensor must be 1-D", OPS_ERROR(ErrCode::PARAM));
    buffer.clear();
    if (t.scalar_type() == at::kLong) {
        auto* data = t.data_ptr<int64_t>();
        buffer.assign(data, data + t.numel());
    } else {
        auto t_contig = t.contiguous().to(at::kLong);
        auto* data = t_contig.data_ptr<int64_t>();
        buffer.assign(data, data + t_contig.numel());
    }
    return at::IntArrayRef(buffer.data(), buffer.size());
}

// Determine output layout from input_layout string.
// Returns the output component of compound layouts like "BSH_BNSD" -> "BNSD".
std::string GetOutputLayout(c10::string_view input_layout)
{
    std::string layout_str(input_layout);
    auto underscore_pos = layout_str.find('_');
    if (underscore_pos != std::string::npos) {
        return layout_str.substr(underscore_pos + 1);
    }
    return layout_str;
}

// Construct output tensors for FusedInferAttentionScore.
// attention_out: D dim = value's D, other dims = query's shape
// softmax_lse: shape depends on layout and softmax_lse_flag
std::tuple<at::Tensor, at::Tensor> construct_fia_output_tensor(
    const at::Tensor& query,
    const at::Tensor& value,
    c10::string_view input_layout,
    bool softmax_lse_flag,
    int64_t out_dtype)
{
    // attention_out shape: follow query's shape, but last dim (D) from value
    auto query_sizes = query.sizes();
    auto value_sizes = value.sizes();
    c10::SmallVector<int64_t, 8> attn_out_shape(query_sizes.begin(), query_sizes.end());
    // Use value's last dim for D
    attn_out_shape.back() = value_sizes.back();

    // Determine dtype for attention_out
    at::ScalarType attn_dtype = query.scalar_type();
    if (out_dtype != 0) {
        // out_dtype uses ge::DataType convention
        auto acl_dtype = static_cast<aclDataType>(out_dtype);
        attn_dtype = npu_preparation::convert_to_scalar_type(acl_dtype);
        if (attn_dtype == at::ScalarType::Undefined) {
            // For INT4/HIFLOAT8 or other types not in the direct mapping,
            // use the value's dtype or query dtype as fallback
            attn_dtype = query.scalar_type();
        }
    }

    at::Tensor attention_out = npu_preparation::apply_tensor_without_format(
        attn_out_shape, query.options().dtype(attn_dtype));

    // softmax_lse shape
    at::Tensor softmax_lse;
    if (softmax_lse_flag) {
        // Per .md: output layout is TND/NTD → [T, N, 1]; otherwise → [B, N, Q_S, 1]
        std::string out_layout = GetOutputLayout(input_layout);
        c10::SmallVector<int64_t, 4> lse_shape;
        if (out_layout == "TND" || out_layout == "NTD") {
            lse_shape = {query_sizes[0], query_sizes[1], 1};
        } else {
            lse_shape = {query_sizes[0], query_sizes[1], query_sizes[2], 1};
        }
        softmax_lse = npu_preparation::apply_tensor_without_format(
            lse_shape, query.options().dtype(at::kFloat));
    } else {
        // When softmax_lse_flag is false, allocate a {1} zero tensor as per .md
        softmax_lse = npu_preparation::apply_tensor_without_format(
            {1}, query.options().dtype(at::kFloat));
    }

    return std::make_tuple(attention_out, softmax_lse);
}

}  // namespace

// ============================================================================
// .default entry point (_symint variant)
// ============================================================================
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_symint(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    int64_t num_heads,
    c10::string_view input_layout,
    const c10::optional<at::Tensor>& pse_shift,
    const c10::optional<at::Tensor>& atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& dequant_scale1,
    const c10::optional<at::Tensor>& quant_scale1,
    const c10::optional<at::Tensor>& dequant_scale2,
    const c10::optional<at::Tensor>& quant_scale2,
    const c10::optional<at::Tensor>& quant_offset2,
    const c10::optional<at::Tensor>& antiquant_scale,
    const c10::optional<at::Tensor>& antiquant_offset,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& query_padding_size,
    const c10::optional<at::Tensor>& kv_padding_size,
    const c10::optional<at::Tensor>& key_antiquant_scale,
    const c10::optional<at::Tensor>& key_antiquant_offset,
    const c10::optional<at::Tensor>& value_antiquant_scale,
    const c10::optional<at::Tensor>& value_antiquant_offset,
    const c10::optional<at::Tensor>& key_shared_prefix,
    const c10::optional<at::Tensor>& value_shared_prefix,
    c10::OptionalArrayRef<int64_t> actual_shared_prefix_len,
    const c10::optional<at::Tensor>& query_rope,
    const c10::optional<at::Tensor>& key_rope,
    const c10::optional<at::Tensor>& key_rope_antiquant_scale,
    const c10::optional<at::Tensor>& dequant_scale_query,
    const c10::optional<at::Tensor>& learnable_sink,
    c10::OptionalArrayRef<int64_t> q_start_idx,
    c10::OptionalArrayRef<int64_t> kv_start_idx,
    double scale,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t num_key_value_heads,
    int64_t sparse_mode,
    int64_t inner_precise,
    int64_t block_size,
    int64_t antiquant_mode,
    bool softmax_lse_flag,
    int64_t key_antiquant_mode,
    int64_t value_antiquant_mode,
    int64_t query_quant_mode,
    int64_t pse_type,
    c10::optional<int64_t> out_dtype)
{
    // Construct output tensors
    auto [attention_out, softmax_lse] = construct_fia_output_tensor(
        query, value, input_layout, softmax_lse_flag, out_dtype.value_or(0));

    // Convert key/value single tensors to TensorList (aclTensorList expected by aclnn)
    at::TensorList key_tensors = key;
    at::TensorList value_tensors = value;

    // Convert input_layout to uppercase char*
    std::string input_layout_str(input_layout);
    for (auto& c : input_layout_str) { c = toupper(c); }
    char layout_char[LAYOUT_MAX_LENGTH];
    strncpy(layout_char, input_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
    layout_char[LAYOUT_MAX_LENGTH - 1] = '\0';

    // Convert SymInt[]? to int[] for aclnn
    std::vector<int64_t> actual_seq_len_buffer;
    std::vector<int64_t> actual_seq_len_kv_buffer;
    at::IntArrayRef actual_seq_len_ref;
    at::IntArrayRef actual_seq_len_kv_ref;
    if (actual_seq_lengths.has_value()) {
        for (const auto& si : actual_seq_lengths.value()) {
            actual_seq_len_buffer.push_back(si.expect_int());
        }
        actual_seq_len_ref = at::IntArrayRef(actual_seq_len_buffer.data(), actual_seq_len_buffer.size());
    }
    if (actual_seq_lengths_kv.has_value()) {
        for (const auto& si : actual_seq_lengths_kv.value()) {
            actual_seq_len_kv_buffer.push_back(si.expect_int());
        }
        actual_seq_len_kv_ref = at::IntArrayRef(actual_seq_len_kv_buffer.data(), actual_seq_len_kv_buffer.size());
    }

    // Get workspace size and executor
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    uint64_t* workspace_size_ptr = &workspace_size;
    aclOpExecutor** executor_ptr = &executor;
    EXEC_NPU_CMD(aclnnFusedInferAttentionScoreV5GetWorkspaceSize,
        query, key_tensors, value_tensors,
        pse_shift, atten_mask, actual_seq_len_ref, actual_seq_len_kv_ref,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
        antiquant_scale, antiquant_offset, block_table,
        query_padding_size, kv_padding_size,
        key_antiquant_scale, key_antiquant_offset,
        value_antiquant_scale, value_antiquant_offset,
        key_shared_prefix, value_shared_prefix,
        actual_shared_prefix_len,
        query_rope, key_rope, key_rope_antiquant_scale,
        dequant_scale_query, learnable_sink,
        q_start_idx, kv_start_idx,
        num_heads, scale, pre_tokens, next_tokens,
        layout_char, num_key_value_heads, sparse_mode, inner_precise,
        block_size, antiquant_mode, softmax_lse_flag,
        key_antiquant_mode, value_antiquant_mode, query_quant_mode, pse_type,
        attention_out, softmax_lse, workspace_size_ptr, executor_ptr);

    // Allocate workspace
    at::Tensor workspace;
    if (workspace_size > 0) {
        workspace = npu_preparation::apply_tensor_without_format(
            {static_cast<int64_t>(workspace_size)}, query.options().dtype(at::kByte));
    }
    void* workspace_addr = workspace_size > 0 ?
        const_cast<void*>(workspace.storage().data()) : nullptr;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV5,
        workspace_addr, workspace_size, executor, stream);

    // Cleanup executor
    NPU_CHECK_ERROR(at_npu::native::AclDestroyAclOpExecutor(executor));

    return std::make_tuple(attention_out, softmax_lse);
}

// ============================================================================
// .out variant (ACLGraph update)
// ============================================================================
std::tuple<at::Tensor&, at::Tensor&> npu_fused_infer_attention_score_out(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    int64_t num_heads,
    c10::string_view input_layout,
    const c10::optional<at::Tensor>& pse_shift,
    const c10::optional<at::Tensor>& atten_mask,
    c10::OptionalArrayRef<int64_t> actual_seq_lengths,
    c10::OptionalArrayRef<int64_t> actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& dequant_scale1,
    const c10::optional<at::Tensor>& quant_scale1,
    const c10::optional<at::Tensor>& dequant_scale2,
    const c10::optional<at::Tensor>& quant_scale2,
    const c10::optional<at::Tensor>& quant_offset2,
    const c10::optional<at::Tensor>& antiquant_scale,
    const c10::optional<at::Tensor>& antiquant_offset,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& query_padding_size,
    const c10::optional<at::Tensor>& kv_padding_size,
    const c10::optional<at::Tensor>& key_antiquant_scale,
    const c10::optional<at::Tensor>& key_antiquant_offset,
    const c10::optional<at::Tensor>& value_antiquant_scale,
    const c10::optional<at::Tensor>& value_antiquant_offset,
    const c10::optional<at::Tensor>& key_shared_prefix,
    const c10::optional<at::Tensor>& value_shared_prefix,
    c10::OptionalArrayRef<int64_t> actual_shared_prefix_len,
    const c10::optional<at::Tensor>& query_rope,
    const c10::optional<at::Tensor>& key_rope,
    const c10::optional<at::Tensor>& key_rope_antiquant_scale,
    const c10::optional<at::Tensor>& dequant_scale_query,
    const c10::optional<at::Tensor>& learnable_sink,
    c10::OptionalArrayRef<int64_t> q_start_idx,
    c10::OptionalArrayRef<int64_t> kv_start_idx,
    double scale,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t num_key_value_heads,
    int64_t sparse_mode,
    int64_t inner_precise,
    int64_t block_size,
    int64_t antiquant_mode,
    bool softmax_lse_flag,
    int64_t key_antiquant_mode,
    int64_t value_antiquant_mode,
    int64_t query_quant_mode,
    int64_t pse_type,
    c10::optional<int64_t> out_dtype,
    const c10::optional<at::Tensor>& workspace,
    at::Tensor& attention_out,
    at::Tensor& softmax_lse)
{
    // Convert key/value to TensorList
    at::TensorList key_tensors = key;
    at::TensorList value_tensors = value;

    // Convert input_layout to uppercase char*
    std::string input_layout_str(input_layout);
    for (auto& c : input_layout_str) { c = toupper(c); }
    char layout_char[LAYOUT_MAX_LENGTH];
    strncpy(layout_char, input_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
    layout_char[LAYOUT_MAX_LENGTH - 1] = '\0';

    if (workspace.has_value() && workspace.value().defined()) {
        void* workspace_addr = const_cast<void*>(workspace.value().storage().data());
        uint64_t workspace_size = static_cast<uint64_t>(
            workspace.value().numel() * workspace.value().element_size());
        EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(
            aclnnFusedInferAttentionScoreV5, workspace_addr, workspace_size,
            query, key_tensors, value_tensors,
            pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
            dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
            antiquant_scale, antiquant_offset, block_table,
            query_padding_size, kv_padding_size,
            key_antiquant_scale, key_antiquant_offset,
            value_antiquant_scale, value_antiquant_offset,
            key_shared_prefix, value_shared_prefix,
            actual_shared_prefix_len,
            query_rope, key_rope, key_rope_antiquant_scale,
            dequant_scale_query, learnable_sink,
            q_start_idx, kv_start_idx,
            num_heads, scale, pre_tokens, next_tokens,
            layout_char, num_key_value_heads, sparse_mode, inner_precise,
            block_size, antiquant_mode, softmax_lse_flag,
            key_antiquant_mode, value_antiquant_mode, query_quant_mode, pse_type,
            attention_out, softmax_lse);
    } else {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(
            aclnnFusedInferAttentionScoreV5GetWorkspaceSize,
            query, key_tensors, value_tensors,
            pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
            dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
            antiquant_scale, antiquant_offset, block_table,
            query_padding_size, kv_padding_size,
            key_antiquant_scale, key_antiquant_offset,
            value_antiquant_scale, value_antiquant_offset,
            key_shared_prefix, value_shared_prefix,
            actual_shared_prefix_len,
            query_rope, key_rope, key_rope_antiquant_scale,
            dequant_scale_query, learnable_sink,
            q_start_idx, kv_start_idx,
            num_heads, scale, pre_tokens, next_tokens,
            layout_char, num_key_value_heads, sparse_mode, inner_precise,
            block_size, antiquant_mode, softmax_lse_flag,
            key_antiquant_mode, value_antiquant_mode, query_quant_mode, pse_type,
            attention_out, softmax_lse);
    }
    return std::tie(attention_out, softmax_lse);
}

// ============================================================================
// _get_max_workspace variant (ACLGraph update)
// ============================================================================
at::Tensor _npu_fused_infer_attention_score_get_max_workspace(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    int64_t num_heads,
    c10::string_view input_layout,
    const c10::optional<at::Tensor>& pse_shift,
    const c10::optional<at::Tensor>& atten_mask,
    c10::OptionalArrayRef<int64_t> actual_seq_lengths,
    c10::OptionalArrayRef<int64_t> actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& dequant_scale1,
    const c10::optional<at::Tensor>& quant_scale1,
    const c10::optional<at::Tensor>& dequant_scale2,
    const c10::optional<at::Tensor>& quant_scale2,
    const c10::optional<at::Tensor>& quant_offset2,
    const c10::optional<at::Tensor>& antiquant_scale,
    const c10::optional<at::Tensor>& antiquant_offset,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& query_padding_size,
    const c10::optional<at::Tensor>& kv_padding_size,
    const c10::optional<at::Tensor>& key_antiquant_scale,
    const c10::optional<at::Tensor>& key_antiquant_offset,
    const c10::optional<at::Tensor>& value_antiquant_scale,
    const c10::optional<at::Tensor>& value_antiquant_offset,
    const c10::optional<at::Tensor>& key_shared_prefix,
    const c10::optional<at::Tensor>& value_shared_prefix,
    c10::OptionalArrayRef<int64_t> actual_shared_prefix_len,
    const c10::optional<at::Tensor>& query_rope,
    const c10::optional<at::Tensor>& key_rope,
    const c10::optional<at::Tensor>& key_rope_antiquant_scale,
    const c10::optional<at::Tensor>& dequant_scale_query,
    const c10::optional<at::Tensor>& learnable_sink,
    c10::OptionalArrayRef<int64_t> q_start_idx,
    c10::OptionalArrayRef<int64_t> kv_start_idx,
    double scale,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t num_key_value_heads,
    int64_t sparse_mode,
    int64_t inner_precise,
    int64_t block_size,
    int64_t antiquant_mode,
    bool softmax_lse_flag,
    int64_t key_antiquant_mode,
    int64_t value_antiquant_mode,
    int64_t query_quant_mode,
    int64_t pse_type,
    c10::optional<int64_t> out_dtype)
{
    // Construct temporary output tensors for workspace size calculation
    auto [attention_out, softmax_lse] = construct_fia_output_tensor(
        query, value, input_layout, softmax_lse_flag, out_dtype.value_or(0));

    // Convert key/value to TensorList
    at::TensorList key_tensors = key;
    at::TensorList value_tensors = value;

    // Convert input_layout to uppercase char*
    std::string input_layout_str(input_layout);
    for (auto& c : input_layout_str) { c = toupper(c); }
    char layout_char[LAYOUT_MAX_LENGTH];
    strncpy(layout_char, input_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
    layout_char[LAYOUT_MAX_LENGTH - 1] = '\0';

    // Call GetWorkspaceSize to compute the required workspace size
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    uint64_t* ws_size_ptr = &workspace_size;
    aclOpExecutor** exec_ptr = &executor;
    EXEC_NPU_CMD(aclnnFusedInferAttentionScoreV5GetWorkspaceSize,
        query, key_tensors, value_tensors,
        pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
        antiquant_scale, antiquant_offset, block_table,
        query_padding_size, kv_padding_size,
        key_antiquant_scale, key_antiquant_offset,
        value_antiquant_scale, value_antiquant_offset,
        key_shared_prefix, value_shared_prefix,
        actual_shared_prefix_len,
        query_rope, key_rope, key_rope_antiquant_scale,
        dequant_scale_query, learnable_sink,
        q_start_idx, kv_start_idx,
        num_heads, scale, pre_tokens, next_tokens,
        layout_char, num_key_value_heads, sparse_mode, inner_precise,
        block_size, antiquant_mode, softmax_lse_flag,
        key_antiquant_mode, value_antiquant_mode, query_quant_mode, pse_type,
        attention_out, softmax_lse, ws_size_ptr, exec_ptr);

    // Cleanup executor
    NPU_CHECK_ERROR(at_npu::native::AclDestroyAclOpExecutor(executor));

    // Return workspace size as a scalar Byte tensor
    return npu_preparation::apply_tensor_without_format(
        {static_cast<int64_t>(workspace_size)}, query.options().dtype(at::kByte));
}

}  // namespace op_api
