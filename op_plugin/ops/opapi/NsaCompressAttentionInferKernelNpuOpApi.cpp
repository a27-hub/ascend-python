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
// limitations under the License. #include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

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
at::Tensor construct_nsa_compress_infer_output_tensor(const at::Tensor& query,
    const at::Tensor& value, int64_t key_value_head_num, std::string layout)
{
    at::SmallVector<int64_t, SIZE> output_size;
    if (layout == "TND") {
        auto key_head_dim = value.size(DIM_2) / key_value_head_num;
        output_size = {query.size(DIM_0), query.size(DIM_1), key_head_dim};
    } else {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), value.size(DIM_3)};
    }
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, query.options().dtype(query.dtype()));
    return output;
}

at::Tensor construct_nsa_compress_infer_topk_output_tensor(const at::Tensor& query,
    int64_t key_value_head_num, int64_t select_block_count, std::string layout)
{
    at::SmallVector<int64_t, SIZE> output_size;
    if (layout == "TND") {
        output_size = {query.size(DIM_0), key_value_head_num, select_block_count};
    } else {
        output_size = {query.size(DIM_0), query.size(DIM_1), key_value_head_num, select_block_count};
    }
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, at::kInt);
    return output;
}

std::tuple<at::Tensor, at::Tensor> npu_nsa_compress_attention_infer_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size,
    int64_t select_block_count, int64_t page_block_size, int64_t compress_block_size,
    int64_t compress_stride, const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, c10::string_view layout,
    int64_t sparse_mode, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &topk_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_cmp_seq_kvlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen )
{
    std::string layout_str = std::string(layout);
    // construct the output tensor
    at::Tensor nsa_compress_infer_output = op_api::construct_nsa_compress_infer_output_tensor(
        query, value, key_value_head_num, layout_str);
    at::Tensor nsa_compress_infer_topk_output = op_api::construct_nsa_compress_infer_topk_output_tensor(
        query, key_value_head_num, select_block_count, layout_str);
    // convert str
    char *layout_ptr = const_cast<char *>(layout_str.c_str());
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaCompressAttentionInferV2, query, key, value,
        atten_mask, block_table, actual_seq_qlen, actual_cmp_seq_kvlen, actual_seq_kvlen,
        topk_mask, query_rope, key_rope, head_num, key_value_head_num, select_block_size,
        select_block_count, compress_block_size, compress_stride, scale_value, layout_ptr,
        page_block_size, sparse_mode, nsa_compress_infer_output, nsa_compress_infer_topk_output);
    return std::tuple<at::Tensor, at::Tensor>(nsa_compress_infer_output, nsa_compress_infer_topk_output);
}

at::Tensor _npu_nsa_compress_attention_infer_get_max_workspace_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double scale_value, int64_t head_num, int64_t key_value_head_num,
    int64_t select_block_size, int64_t select_block_count, int64_t page_block_size,
    int64_t compress_block_size, int64_t compress_stride,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    c10::string_view layout, int64_t sparse_mode,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &topk_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_cmp_seq_kvlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen)
{
    std::string layout_str = std::string(layout);
    // construct the output tensor
    at::Tensor nsa_compress_infer_output = op_api::construct_nsa_compress_infer_output_tensor(
        query, value, key_value_head_num, layout_str);

    at::Tensor nsa_compress_infer_topk_output = op_api::construct_nsa_compress_infer_topk_output_tensor(
        query, key_value_head_num, select_block_count, layout_str);

    // convert str
    char *layout_ptr = const_cast<char *>(layout_str.c_str());
    uint64_t workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnNsaCompressAttentionInferV2, query,
        key, value, atten_mask, block_table, actual_seq_qlen, actual_cmp_seq_kvlen, actual_seq_kvlen,
        topk_mask, query_rope, key_rope, head_num, key_value_head_num, select_block_size,
        select_block_count, compress_block_size, compress_stride, scale_value, layout_ptr,
        page_block_size, sparse_mode, nsa_compress_infer_output, nsa_compress_infer_topk_output);

    at::Tensor workspace_tensor = npu_preparation::apply_tensor_without_format({workspace_size},
        query.options().dtype(query.dtype()));
    return workspace_tensor;
}

} // namespace op_api