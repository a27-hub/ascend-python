/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <tuple>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace {
constexpr int64_t QUANT_MODE_NONE = -1;
constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2;
constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3;
constexpr int64_t QUANT_MODE_MXFP4_E2M1 = 9;
constexpr int64_t MX_QUANT_BLOCK_SIZE = 32;
constexpr int64_t MXFP4_SCALE_BLOCK_SIZE = 64;
constexpr int64_t PAD_TO_EVEN_FACTOR = 2;

bool IsMxQuantMode(int64_t quant_mode)
{
    return quant_mode == QUANT_MODE_MXFP8_E5M2 || quant_mode == QUANT_MODE_MXFP8_E4M3FN ||
           quant_mode == QUANT_MODE_MXFP4_E2M1;
}
}  // namespace

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_moe_token_permute(
    const at::Tensor &tokens, const at::Tensor &indices, c10::optional<int64_t> num_out_tokens, bool padded_mode,
    int64_t quant_mode)
{
    TORCH_CHECK(tokens.dim() == 2,
        "tokens must be a 2D tensor, but got ",
        tokens.dim(),
        " dimensions.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(indices.dim() == 1 || indices.dim() == 2,
        "indices must be a 1D or 2D tensor, but got ",
        indices.dim(),
        " dimensions.",
        OPS_ERROR(ErrCode::PARAM));

    int64_t flatten_size = indices.numel();
    int64_t num_out_tokens_value = num_out_tokens.value_or(0);
    int64_t actual_num_out_tokens = num_out_tokens_value > 0
        ? std::min(num_out_tokens_value, flatten_size)
        : std::max<int64_t>(num_out_tokens_value + flatten_size, 0);
    bool is_ascend950 = c10_npu::GetSocVersion() == c10_npu::SocVersion::Ascend950;
    int64_t effective_quant_mode = is_ascend950 ? quant_mode : QUANT_MODE_NONE;

    TORCH_CHECK(effective_quant_mode == QUANT_MODE_NONE || IsMxQuantMode(effective_quant_mode),
        "quant_mode only supports -1, 2, 3 or 9 on Ascend 950.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!tokens.requires_grad() || effective_quant_mode == QUANT_MODE_NONE,
        "npu_moe_token_permute does not support autograd when quant_mode is 2, 3 or 9.",
        OPS_ERROR(ErrCode::NOT_SUPPORT));

    auto token_options = tokens.options();
    int64_t hidden_size = tokens.size(1);
    at::Tensor permuted_tokens;
    at::Tensor expanded_scale;

#if VERSION_BETWEEN(V2R7, VERSION_NEWEST)
    if (effective_quant_mode == QUANT_MODE_MXFP8_E5M2) {
        permuted_tokens = npu_preparation::apply_tensor_without_format(
            {actual_num_out_tokens, hidden_size}, token_options.dtype(at::kFloat8_e5m2));
        int64_t scale_cols = (hidden_size + MX_QUANT_BLOCK_SIZE - 1) / MX_QUANT_BLOCK_SIZE;
        scale_cols = (scale_cols + PAD_TO_EVEN_FACTOR - 1) / PAD_TO_EVEN_FACTOR * PAD_TO_EVEN_FACTOR;
        expanded_scale = npu_preparation::apply_tensor_without_format(
            {actual_num_out_tokens, scale_cols}, token_options.dtype(at::kFloat8_e8m0fnu));
    } else if (effective_quant_mode == QUANT_MODE_MXFP8_E4M3FN) {
        permuted_tokens = npu_preparation::apply_tensor_without_format(
            {actual_num_out_tokens, hidden_size}, token_options.dtype(at::kFloat8_e4m3fn));
        int64_t scale_cols = (hidden_size + MX_QUANT_BLOCK_SIZE - 1) / MX_QUANT_BLOCK_SIZE;
        scale_cols = (scale_cols + PAD_TO_EVEN_FACTOR - 1) / PAD_TO_EVEN_FACTOR * PAD_TO_EVEN_FACTOR;
        expanded_scale = npu_preparation::apply_tensor_without_format(
            {actual_num_out_tokens, scale_cols}, token_options.dtype(at::kFloat8_e8m0fnu));
    } else if (effective_quant_mode == QUANT_MODE_MXFP4_E2M1) {
        TORCH_CHECK(hidden_size % 2 == 0,
            "The hidden size must be even when quant_mode is 9.",
            OPS_ERROR(ErrCode::PARAM));
        permuted_tokens = npu_preparation::apply_tensor_without_format(
            {actual_num_out_tokens, hidden_size / 2}, token_options.dtype(at::kByte));
        expanded_scale = npu_preparation::apply_tensor_without_format(
            {actual_num_out_tokens, (hidden_size + MXFP4_SCALE_BLOCK_SIZE - 1) / MXFP4_SCALE_BLOCK_SIZE, 2},
            token_options.dtype(at::kByte));
    } else {
        permuted_tokens = npu_preparation::apply_tensor_without_format(
            {actual_num_out_tokens, hidden_size}, token_options);
        expanded_scale = npu_preparation::apply_tensor_without_format({0}, token_options.dtype(at::kFloat));
    }
#else
    TORCH_CHECK(!IsMxQuantMode(effective_quant_mode),
        "quant_mode 2, 3 and 9 require PyTorch and torch_npu v2.7 or later.",
        OPS_ERROR(ErrCode::NOT_SUPPORT));
    permuted_tokens = npu_preparation::apply_tensor_without_format(
        {actual_num_out_tokens, hidden_size}, token_options);
    expanded_scale = npu_preparation::apply_tensor_without_format({0}, token_options.dtype(at::kFloat));
#endif

    at::Tensor sorted_indices = npu_preparation::apply_tensor_without_format(
        {flatten_size}, indices.options().dtype(at::kInt));
    TensorWrapper permuted_tokens_wrapper = {
        permuted_tokens, npu_preparation::convert_to_acl_data_type(permuted_tokens.scalar_type())};
    TensorWrapper expanded_scale_wrapper = {
        expanded_scale, npu_preparation::convert_to_acl_data_type(expanded_scale.scalar_type())};
    if (effective_quant_mode == QUANT_MODE_MXFP4_E2M1) {
        permuted_tokens_wrapper.dtype = aclDataType::ACL_FLOAT4_E2M1;
        expanded_scale_wrapper.dtype = aclDataType::ACL_FLOAT8_E8M0;
    }

    EXEC_NPU_CMD(aclnnMoeTokenPermute,
        tokens,
        indices,
        actual_num_out_tokens,
        padded_mode,
        quant_mode,
        permuted_tokens_wrapper,
        sorted_indices,
        expanded_scale_wrapper);
    return std::make_tuple(std::move(permuted_tokens), std::move(sorted_indices), std::move(expanded_scale));
}
}  // namespace op_api
