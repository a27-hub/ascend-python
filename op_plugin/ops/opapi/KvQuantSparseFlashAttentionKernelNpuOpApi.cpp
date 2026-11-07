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
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int DIM_4 = 4;

at::Tensor construct_quant_sparse_infer_output_tensor(
    const at::Tensor& query, std::string layout_query_str,
    std::string layout_kv_str, const uint64_t &rope_head_dim)
{
    TORCH_CHECK(layout_query_str == "BSND" || layout_query_str == "TND",
                "The layout of query only support BSND and TND, but got ", layout_query_str,
                OPS_ERROR(ErrCode::PARAM));
    for (auto i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    at::SmallVector<int64_t, SIZE> output_size;
    if (layout_query_str == "BSND") {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ",
                    query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3) - rope_head_dim};
    } else {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ",
                    query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2) - rope_head_dim};
    }
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, query.options().dtype(query.dtype()));

    return output;
}

// LSE(softmax_max / softmax_sum) 的输出张量构造。
// 形状与 CANN 内置算子 infershape 完全一致：
//   TND  -> [N2, T, G] ; BSND -> [B, N2, S1, G] ; G = N1 / N2
// MLA(attention_mode==2) 下 N2 恒为 1、G 即 N1，可直接由 query 的第 1/2 维得到。
at::Tensor construct_lse_output_tensor(
    const at::Tensor& query, const std::string& layout_query_str,
    const uint64_t& rope_head_dim, const int64_t attention_mode)
{
    (void)rope_head_dim;
    TORCH_CHECK(attention_mode == 2,
                "return_softmax_lse is only supported when attention_mode is 2 (MLA), but got ",
                attention_mode, OPS_ERROR(ErrCode::PARAM));
    constexpr int64_t kMlaKvHeadNum = 1;
    at::SmallVector<int64_t, SIZE> lse_size;
    if (layout_query_str == "BSND") {
        // query [B, S1, N1, D] -> [B, N2, S1, G]
        lse_size = {query.size(DIM_0), kMlaKvHeadNum, query.size(DIM_1),
                    query.size(DIM_2) / kMlaKvHeadNum};
    } else {
        // query [T, N1, D] -> [N2, T, G]
        lse_size = {kMlaKvHeadNum, query.size(DIM_0),
                    query.size(DIM_1) / kMlaKvHeadNum};
    }
    return npu_preparation::apply_tensor_without_format(
        lse_size, query.options().dtype(at::kFloat));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_kv_quant_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value,
    int64_t key_quant_mode, int64_t value_quant_mode,
    const c10::optional<at::Tensor> &key_dequant_scale,
    const c10::optional<at::Tensor> &value_dequant_scale,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t sparse_block_size, c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t attention_mode, int64_t quant_scale_repo_mode,
    int64_t tile_size, int64_t rope_head_dim, bool return_softmax_lse,
    c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")
    TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.")

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    bool check_kdtype = key_dtype.has_value() && c10_npu::GetAclDataType(key_dtype.value()) != aclDataType::ACL_HIFLOAT8;
    TORCH_CHECK(!check_kdtype, "The key_dtype is only used to support hifloat8, and should be default when the dtype of key tensor is not hifloat8.")
    bool check_vdtype = value_dtype.has_value() && c10_npu::GetAclDataType(value_dtype.value()) != aclDataType::ACL_HIFLOAT8;
    TORCH_CHECK(!check_vdtype, "The value_dtype is only used to support hifloat8, and should be default when the dtype of value tensor is not hifloat8.")

    // construct the output tensor
    at::Tensor output = op_api::construct_quant_sparse_infer_output_tensor(
        query, layout_query_str, layout_kv_str, rope_head_dim);
    // LSE (softmax_max / softmax_sum) 的形状与 CANN 内置算子一致：
    //   TND  -> [N2, T, G]    BSND -> [B, N2, S1, G]，G = N1 / N2
    // return_softmax_lse=false 时传空 optional，aclnn 侧推导为 [0] 且不写入。
    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    // 具名 optional：EXEC_NPU 系列宏按 non-const 引用转发参数，
    // 三目表达式会产生 rvalue，无法绑定，必须先落到左值。
    c10::optional<at::Tensor> softmax_max_opt;
    c10::optional<at::Tensor> softmax_sum_opt;
    if (return_softmax_lse) {
        softmax_max = op_api::construct_lse_output_tensor(
            query, layout_query_str, rope_head_dim, attention_mode);
        softmax_sum = op_api::construct_lse_output_tensor(
            query, layout_query_str, rope_head_dim, attention_mode);
        softmax_max_opt = softmax_max;
        softmax_sum_opt = softmax_sum;
    }
    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    bool is_hifloat8_kv = key_dtype.has_value() && c10_npu::GetAclDataType(key_dtype.value()) == aclDataType::ACL_HIFLOAT8;
    if (is_hifloat8_kv) {
        TensorWrapper key_wrapper = make_wrapper(key, key_dtype);
        TensorWrapper value_wrapper = make_wrapper(value, value_dtype);
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvQuantSparseFlashAttention, query, key_wrapper,
            value_wrapper, sparse_indices, key_dequant_scale, value_dequant_scale, block_table, actual_seq_lengths_query,
            actual_seq_lengths_kv, scale_value, key_quant_mode, value_quant_mode, sparse_block_size, layout_query_ptr,
            layout_kv_ptr, sparse_mode, pre_tokens, next_tokens, attention_mode, quant_scale_repo_mode, tile_size,
            rope_head_dim, return_softmax_lse, output, softmax_max_opt, softmax_sum_opt);
    } else {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvQuantSparseFlashAttention, query,
            key, value, sparse_indices, key_dequant_scale, value_dequant_scale, block_table, actual_seq_lengths_query,
            actual_seq_lengths_kv, scale_value, key_quant_mode, value_quant_mode, sparse_block_size, layout_query_ptr,
            layout_kv_ptr, sparse_mode, pre_tokens, next_tokens, attention_mode, quant_scale_repo_mode, tile_size,
            rope_head_dim, return_softmax_lse, output, softmax_max_opt, softmax_sum_opt);
    }
    if (!return_softmax_lse) {
        // 未请求 LSE 时返回空张量，保持与旧调用方的返回形态兼容
        return std::make_tuple(output, at::Tensor(), at::Tensor());
    }
    return std::make_tuple(output, softmax_max, softmax_sum);
}

} // namespace op_api
