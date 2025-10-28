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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &sparse_indices,
    const at::Tensor &d_out,
    const at::Tensor &out,
    const at::Tensor &softmax_max,
    const at::Tensor &softmax_sum,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    double scale_value,
    int64_t sparse_block_size,
    c10::optional<c10::string_view> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<bool> deterministic)
{
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
    const at::Tensor &actual_seq_lengths_query_const = actual_seq_lengths_query.value_or(at::Tensor());
    const at::Tensor &actual_seq_lengths_key_const = actual_seq_lengths_key.value_or(at::Tensor());
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (query_rope_const.defined()) {
        TORCH_CHECK(query_rope_const.dim() == DIMENSION_3D || query_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input query_rope should be 3 or 4 dimensional, but got ",
            query_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (key_rope_const.defined()) {
        TORCH_CHECK(key_rope_const.dim() == DIMENSION_3D || key_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input key_rope should be 3 or 4 dimensional, but got ",
            key_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "The shapes of the input value should be 3 or 4 dimensional, but got ",
        value.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    at::Tensor format_query = format_trans(query);
    at::Tensor format_key = format_trans(key);
    at::Tensor format_value = format_trans(value);
    at::Tensor format_sparse_indices = format_trans(sparse_indices);
    at::Tensor format_d_out = format_trans(d_out);
    at::Tensor format_out = format_trans(out);
    at::Tensor format_softmax_max = format_trans(softmax_max);
    at::Tensor format_softmax_sum = format_trans(softmax_sum);
    at::Tensor format_query_rope = format_trans(query_rope_const);
    at::Tensor format_key_rope = format_trans(key_rope_const);
    at::Tensor format_actual_seq_lengths_query = format_trans(actual_seq_lengths_query_const);
    at::Tensor format_actual_seq_lengths_key = format_trans(actual_seq_lengths_key_const);
    at::Tensor d_query = OpPreparation::apply_tensor_without_format(format_query);
    at::Tensor d_key = OpPreparation::apply_tensor_without_format(format_key);
    at::Tensor d_value = OpPreparation::apply_tensor_without_format(format_value);
    at::Tensor d_query_rope;
    at::Tensor d_key_rope;
    if (format_query_rope.defined()) {
        d_query_rope = OpPreparation::apply_tensor_without_format(format_query_rope);
    } else {
        d_query_rope = at::empty({0}, query.options());
    }
    if (format_key_rope.defined()) {
        d_key_rope = OpPreparation::apply_tensor_without_format(format_key_rope);
    } else {
        d_key_rope = at::empty({0}, key.options());
    }

    EXEC_NPU_CMD(
        aclnnSparseFlashAttentionGrad, format_query, format_key, format_value, format_sparse_indices, format_d_out
        format_out, format_softmax_max, format_softmax_sum, d_query, d_key, d_value, d_query_rope, d_key_rope,
        format_query_rope, format_key_rope, format_actual_seq_lengths_query, format_actual_seq_lengths_key, scale_value,
        sparse_block_size, layout, sparse_mode, deterministic);

    return std::make_tuple(d_query, d_key, d_value, d_query_rope, d_key_rope);
}
}
