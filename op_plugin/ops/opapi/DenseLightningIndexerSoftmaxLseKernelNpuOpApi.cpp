// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
const int64_t DIMENSION_3D = 3;
const int64_t DIMENSION_4D = 4;
const int64_t SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3;
const int64_t MAX_INT64 = 9223372036854775807;

std::tuple<at::Tensor, at::Tensor> npu_dense_lightning_indexer_softmax_lse_symint(
    const at::Tensor &query_index,
    const at::Tensor &key_index,
    const at::Tensor &weights,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_klen,
    c10::optional<c10::string_view> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens)
{
    c10::string_view layout_str = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str.data());
    int64_t sparse_mode_const = sparse_mode.value_or(SPARSE_MODE_RIGHT_DOWN_CAUSAL);
    int64_t pre_tokens_const = pre_tokens.value_or(MAX_INT64);
    int64_t next_tokens_const = next_tokens.value_or(MAX_INT64);
    std::string layout_str_std = std::string(layout_str);

    TORCH_CHECK(layout_str_std == "BSND" || layout_str_std == "TND",
        "layout only supports BSND or TND, but got ", layout_str_std, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(query_index.dim() == DIMENSION_3D || query_index.dim() == DIMENSION_4D,
        "The shapes of the input query_index should be 3 or 4 dimensional, but got ",
        query_index.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key_index.dim() == DIMENSION_3D || key_index.dim() == DIMENSION_4D,
        "The shapes of the input key_index should be 3 or 4 dimensional, but got ",
        key_index.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(query_index.dtype() == key_index.dtype(),
        "query_index and key_index should have the same dtype, but got query_index ",
        query_index.dtype(), " and key_index ", key_index.dtype(), OPS_ERROR(ErrCode::PARAM));
    if (weights.dtype() != at::kFloat) {
        TORCH_CHECK(query_index.dtype() == weights.dtype(),
            "When weights is not float32, query_index, key_index and weights should have the same dtype, "
            "but got query_index ", query_index.dtype(), " and weights ", weights.dtype(),
            OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(sparse_mode_const == SPARSE_MODE_RIGHT_DOWN_CAUSAL,
        "sparse_mode only supports 3 (rightDownCausal), but got ", sparse_mode_const,
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(pre_tokens_const == MAX_INT64,
        "pre_tokens only supports 2^63-1, but got ", pre_tokens_const, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(next_tokens_const == MAX_INT64,
        "next_tokens only supports 2^63-1, but got ", next_tokens_const, OPS_ERROR(ErrCode::PARAM));

    // softmax_max / softmax_sum share the same shape:
    //   BSND: query_index (B,S1,Nidx1,D), key_index (B,S2,Nidx2,D) -> (B, Nidx2, S1)
    //   TND:  query_index (T1,Nidx1,D),   key_index (T2,Nidx2,D)   -> (Nidx2, T1)
    at::SmallVector<int64_t, SIZE> softmax_shape;
    if (layout_str_std == "BSND") {
        softmax_shape = {query_index.size(0), key_index.size(2), query_index.size(1)};
    } else {
        softmax_shape = {key_index.size(1), query_index.size(0)};
    }
    at::Tensor softmax_max = OpPreparation::apply_tensor_without_format(
        softmax_shape, query_index.options().dtype(at::kFloat));
    at::Tensor softmax_sum = OpPreparation::apply_tensor_without_format(
        softmax_shape, query_index.options().dtype(at::kFloat));

    // Empty query_index: return directly without launching the kernel.
    if (query_index.numel() == 0) {
        return std::make_tuple(softmax_max, softmax_sum);
    }

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnDenseLightningIndexerSoftmaxLse, query_index, key_index, weights,
        actual_seq_qlen, actual_seq_klen, layout_ptr, sparse_mode_const,
        pre_tokens_const, next_tokens_const, softmax_max, softmax_sum);

    return std::make_tuple(softmax_max, softmax_sum);
}
}
