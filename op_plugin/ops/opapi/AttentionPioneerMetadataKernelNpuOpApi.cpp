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

using npu_preparation = at_npu::native::OpPreparation;

const static int64_t METADATA_SIZE = 1024;

at::Tensor _npu_attention_pioneer_metadata(
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t num_heads_q, int64_t num_heads_kv,
    int64_t head_dim_qk, int64_t head_dim_v,
    int64_t batch_size, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, c10::string_view input_layout_kv,
    int64_t k_sink_num, int64_t rope_head_dim,
    int64_t block_size, int64_t aic_core_num,
    int64_t aiv_core_num)
{
    // allocate output tensor: 1024 x int32
    at::Tensor meta_data = npu_preparation::apply_tensor_without_format(
        {METADATA_SIZE}, c10::dtype(c10::ScalarType::Int));

    std::string input_layout_str = std::string(input_layout);
    std::string input_layout_kv_str = std::string(input_layout_kv);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    char *input_layout_kv_ptr = const_cast<char *>(input_layout_kv_str.c_str());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAttentionPioneerMetadata,
        actual_seq_lengths, actual_seq_lengths_kv,
        num_heads_q, num_heads_kv, head_dim_qk, head_dim_v,
        batch_size, sparse_mode, pre_tokens, next_tokens,
        input_layout_ptr, input_layout_kv_ptr,
        k_sink_num, rope_head_dim, block_size,
        aic_core_num, aiv_core_num,
        meta_data);

    return meta_data;
}

#endif
} // namespace op_api
