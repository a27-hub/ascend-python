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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;

namespace {
// Shape推导：输出shape跟随query，dtype固定bfloat16
// BSND: query=[B,S1,Q_N,D] → out=[B,S1,Q_N,D]
// TND:  query=[Q_T,Q_N,D]  → out=[Q_T,Q_N,D]
at::Tensor construct_output_tensor(const at::Tensor& query, std::string layout)
{
    TORCH_CHECK(layout == "BSND" || layout == "TND",
        "The layout of query only support BSND and TND, but got ",
        layout, OPS_ERROR(ErrCode::PARAM));
    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
            "All values within query's shape should be greater than 0, but shape[",
            i, "] is ", query.size(i), OPS_ERROR(ErrCode::PARAM));
    }
    if (layout == "TND") {
        TORCH_CHECK(query.dim() == DIM_3,
            "When the layout of query is TND, the query dimension must be 3, but got ",
            query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    } else {
        TORCH_CHECK(query.dim() == DIM_4,
            "When the layout of query is BSND, the query dimension must be 4, but got ",
            query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3)};
    }
    at::Tensor output = npu_preparation::apply_tensor_without_format(
        output_size, query.options().dtype(at::kBFloat16));
    return output;
}
} // namespace

at::Tensor _npu_kv_quant_sparse_flash_attention_pioneer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value,
    int64_t key_quant_mode, int64_t value_quant_mode,
    const c10::optional<at::Tensor> &key_dequant_scale,
    const c10::optional<at::Tensor> &value_dequant_scale,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &key_sink,
    const c10::optional<at::Tensor> &value_sink,
    int64_t sparse_block_size, c10::string_view layout_query,
    c10::string_view layout_kv, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens,
    int64_t attention_mode, int64_t quant_scale_repo_mode,
    int64_t tile_size, int64_t rope_head_dim,
    c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.", OPS_ERROR(ErrCode::PARAM));

    bool check_kdtype = key_dtype.has_value() &&
        c10_npu::GetAclDataType(key_dtype.value()) != aclDataType::ACL_HIFLOAT8;
    TORCH_CHECK(!check_kdtype,
        "The key_dtype is only used to support hifloat8, and should be default "
        "when the dtype of key tensor is not hifloat8.", OPS_ERROR(ErrCode::PARAM));
    bool check_vdtype = value_dtype.has_value() &&
        c10_npu::GetAclDataType(value_dtype.value()) != aclDataType::ACL_HIFLOAT8;
    TORCH_CHECK(!check_vdtype,
        "The value_dtype is only used to support hifloat8, and should be default "
        "when the dtype of value tensor is not hifloat8.", OPS_ERROR(ErrCode::PARAM));

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    // construct the output tensor
    at::Tensor output = construct_output_tensor(query, layout_query_str);

    // convert str to char*
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    // handle hifloat8 dtype for key/value via TensorWrapper
    bool is_hifloat8_kv = key_dtype.has_value() &&
        c10_npu::GetAclDataType(key_dtype.value()) == aclDataType::ACL_HIFLOAT8;
    if (is_hifloat8_kv) {
        TensorWrapper key_wrapper = make_wrapper(key, key_dtype);
        TensorWrapper value_wrapper = make_wrapper(value, value_dtype);
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvQuantSparseFlashAttentionPioneer,
            query, key_wrapper, value_wrapper, sparse_indices, scale_value,
            key_quant_mode, value_quant_mode,
            key_dequant_scale, value_dequant_scale, block_table,
            actual_seq_lengths_query, actual_seq_lengths_kv,
            key_sink, value_sink,
            sparse_block_size, layout_query_ptr, layout_kv_ptr,
            sparse_mode, pre_tokens, next_tokens,
            attention_mode, quant_scale_repo_mode, tile_size, rope_head_dim,
            output);
    } else {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvQuantSparseFlashAttentionPioneer,
            query, key, value, sparse_indices, scale_value,
            key_quant_mode, value_quant_mode,
            key_dequant_scale, value_dequant_scale, block_table,
            actual_seq_lengths_query, actual_seq_lengths_kv,
            key_sink, value_sink,
            sparse_block_size, layout_query_ptr, layout_kv_ptr,
            sparse_mode, pre_tokens, next_tokens,
            attention_mode, quant_scale_repo_mode, tile_size, rope_head_dim,
            output);
    }
    return output;
}

} // namespace op_api