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
#include "op_plugin/utils/custom_functions/tensor_wrapper.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t DIM_4 = 4;
const static int64_t ACL_HIFLOAT8 = 34;  // hifloat8 ACL enum value

at::Tensor construct_kv_quant_sparse_flash_attention_pioneer_output(
    const at::Tensor& query, std::string layout_query, int64_t rope_head_dim)
{
    TORCH_CHECK(layout_query == "BSND" || layout_query == "TND",
        "The layout of query only support BSND and TND, but got ",
        layout_query, OPS_ERROR(ErrCode::PARAM));

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i), OPS_ERROR(ErrCode::PARAM));
    }

    // Output D dimension only contains nope part, excluding rope part
    int64_t output_d = query.size(DIM_3) - rope_head_dim;
    TORCH_CHECK(output_d > 0, "Output D dimension (query.D - rope_head_dim) must be greater than 0, "
        "but got query.D=", query.size(DIM_3), ", rope_head_dim=", rope_head_dim,
        OPS_ERROR(ErrCode::PARAM));

    if (layout_query == "TND") {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ",
                    query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), output_d};
    } else {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ",
                    query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), output_d};
    }

    // Output dtype is bfloat16 according to design spec
    at::Tensor output = npu_preparation::apply_tensor_without_format(
        output_size, query.options().dtype(at::kBFloat16));

    return output;
}

bool is_hifloat8_dtype(const c10::optional<int64_t>& dtype_opt)
{
    if (!dtype_opt.has_value()) {
        return false;
    }
    aclDataType acl_type = c10_npu::GetAclDataType(dtype_opt.value());
    return acl_type == ACL_HIFLOAT8;
}
}

at::Tensor _npu_kv_quant_sparse_flash_attention_pioneer(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    double scale_value,
    int64_t key_quant_mode,
    int64_t value_quant_mode,
    const c10::optional<at::Tensor>& key_dequant_scale,
    const c10::optional<at::Tensor>& value_dequant_scale,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& key_sink,
    const c10::optional<at::Tensor>& value_sink,
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
    const c10::optional<int64_t>& key_dtype,
    const c10::optional<int64_t>& value_dtype)
{
    // Input validation
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(value.numel() > 0, "Tensor value is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.", OPS_ERROR(ErrCode::PARAM));

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    // Construct output tensor (D dimension excludes rope part)
    at::Tensor output = construct_kv_quant_sparse_flash_attention_pioneer_output(query, layout_query_str, rope_head_dim);

    // Convert string params to char*
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    // Process key tensor - wrap with TensorWrapper if hifloat8
    std::variant<at::Tensor, TensorWrapper> key_processed;
    if (is_hifloat8_dtype(key_dtype)) {
        key_processed = make_wrapper(key, key_dtype.value());
    } else {
        key_processed = key;
    }

    // Process value tensor - wrap with TensorWrapper if hifloat8
    std::variant<at::Tensor, TensorWrapper> value_processed;
    if (is_hifloat8_dtype(value_dtype)) {
        value_processed = make_wrapper(value, value_dtype.value());
    } else {
        value_processed = value;
    }

    // Use std::visit to handle all combinations of key/value types (2x2=4 cases)
    std::visit([&](const auto& key_arg, const auto& value_arg) {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(
            aclnnKvQuantSparseFlashAttentionPioneer,
            query,
            key_arg,
            value_arg,
            sparse_indices,
            scale_value,
            key_quant_mode,
            value_quant_mode,
            key_dequant_scale,
            value_dequant_scale,
            block_table,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
            key_sink,
            value_sink,
            sparse_block_size,
            layout_query_ptr,
            layout_kv_ptr,
            sparse_mode,
            pre_tokens,
            next_tokens,
            attention_mode,
            quant_scale_repo_mode,
            tile_size,
            rope_head_dim,
            key_dtype.value_or(0),
            value_dtype.value_or(0),
            output);
    }, key_processed, value_processed);

    return output;
}

} // namespace op_api
