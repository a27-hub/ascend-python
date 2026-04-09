// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)

const static int64_t METADATA_SIZE = 1024;

using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor _npu_attention_pioneer_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim_qk,
    int64_t head_dim_v,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t batch_size,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    c10::string_view input_layout,
    int64_t sink_number,
    int64_t rope_head_dim,
    int64_t block_size)
{
    // parameter validation
    TORCH_CHECK(num_heads_q > 0,
        "num_heads_q should be greater than 0, but got ", num_heads_q,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(num_heads_kv > 0,
        "num_heads_kv should be greater than 0, but got ", num_heads_kv,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(num_heads_q % num_heads_kv == 0,
        "num_heads_q should be divisible by num_heads_kv, but got num_heads_q=",
        num_heads_q, ", num_heads_kv=", num_heads_kv,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(head_dim_qk > 0,
        "head_dim_qk should be greater than 0, but got ", head_dim_qk,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(head_dim_v > 0,
        "head_dim_v should be greater than 0, but got ", head_dim_v,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(sparse_mode >= 0 && sparse_mode <= 4,
        "sparse_mode should be in [0, 4], but got ", sparse_mode,
        OPS_ERROR(ErrCode::VALUE));
    if (block_size > 0) {
        TORCH_CHECK(block_size % 16 == 0,
            "block_size should be 16-aligned, but got ", block_size,
            OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(block_size <= 1024,
            "block_size should not exceed 1024, but got ", block_size,
            OPS_ERROR(ErrCode::VALUE));
    }

    // convert string
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    // create output tensor: [METADATA_SIZE] int32
    auto options = at::TensorOptions().dtype(at::ScalarType::Int);
    if (actual_seq_lengths.has_value() && actual_seq_lengths.value().defined()) {
        options = options.device(actual_seq_lengths.value().device());
    } else if (actual_seq_lengths_kv.has_value() && actual_seq_lengths_kv.value().defined()) {
        options = options.device(actual_seq_lengths_kv.value().device());
    }
    at::Tensor meta_data = npu_preparation::apply_tensor_without_format({METADATA_SIZE}, options);

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAttentionPioneerMetadata,
        actual_seq_lengths, actual_seq_lengths_kv,
        num_heads_q, num_heads_kv, head_dim_qk, head_dim_v,
        batch_size, sparse_mode, pre_tokens, next_tokens,
        input_layout_ptr, sink_number, rope_head_dim, block_size,
        meta_data);

    return meta_data;
}

#endif
} // namespace op_api
