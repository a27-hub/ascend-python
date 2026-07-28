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

#include <cstring>
#include <string>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

// construct softmax_max / softmax_sum output shape:
//   BSND: query_index=(B,S1,Nidx1,D), key_index=(B,S2,Nidx2,D) -> (B, Nidx2, S1)
//   TND:  query_index=(T1,Nidx1,D),   key_index=(T2,Nidx2,D)   -> (Nidx2, T1)
// NOTE: sync with meta (graph mode shape derivation)
static c10::SmallVector<int64_t, SIZE> dense_lightning_indexer_softmax_lse_out_size(
    const at::Tensor &query_index, const at::Tensor &key_index, const std::string &layout_str)
{
    TORCH_CHECK(layout_str == "BSND" || layout_str == "TND",
        "layout only supports BSND or TND, but got ", layout_str, OPS_ERROR(ErrCode::VALUE));
    c10::SmallVector<int64_t, SIZE> output_size;
    if (layout_str == "BSND") {
        output_size = {query_index.size(DIM_0), key_index.size(DIM_2), query_index.size(DIM_1)};
    } else {  // TND
        output_size = {key_index.size(DIM_1), query_index.size(DIM_0)};
    }
    return output_size;
}

std::tuple<at::Tensor, at::Tensor> npu_dense_lightning_indexer_softmax_lse_symint(
    const at::Tensor &query_index, const at::Tensor &key_index, const at::Tensor &weight,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_klen,
    c10::string_view layout, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens)
{
    // dtype constraints (from .md 约束说明)
    TORCH_CHECK(query_index.scalar_type() == key_index.scalar_type(),
        "query_index and key_index must have the same dtype, got query_index=", query_index.scalar_type(),
        ", key_index=", key_index.scalar_type(), OPS_ERROR(ErrCode::PARAM));
    const bool weight_is_fp32 = weight.scalar_type() == at::kFloat;
    TORCH_CHECK(weight_is_fp32 || (weight.scalar_type() == query_index.scalar_type()
        && weight.scalar_type() == key_index.scalar_type()),
        "when weight is not float32, query_index/key_index/weight must share the same dtype, got query_index=",
        query_index.scalar_type(), ", key_index=", key_index.scalar_type(),
        ", weight=", weight.scalar_type(), OPS_ERROR(ErrCode::PARAM));
    // value constraints
    TORCH_CHECK(sparse_mode == 3,
        "sparse_mode only supports 3 (rightDownCausal), but got ", sparse_mode, OPS_ERROR(ErrCode::PARAM));

    std::string layout_str = std::string(layout);

    // construct the output tensors (dtype always FP32)
    auto output_size = dense_lightning_indexer_softmax_lse_out_size(query_index, key_index, layout_str);
    at::Tensor softmax_max_out = npu_preparation::apply_tensor_without_format(output_size, at::kFloat);
    at::Tensor softmax_sum_out = npu_preparation::apply_tensor_without_format(output_size, at::kFloat);

    // layout string -> char* (aclnn reads the C string)
    char *layout_ptr = const_cast<char *>(layout_str.c_str());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnDenseLightningIndexerSoftmaxLse, query_index,
        key_index, weight, actual_seq_qlen, actual_seq_klen, layout_ptr,
        sparse_mode, pre_tokens, next_tokens, softmax_max_out, softmax_sum_out);

    return std::make_tuple(softmax_max_out, softmax_sum_out);
}
} // namespace op_api
