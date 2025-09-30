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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
void npu_ffn_to_attention(const at::Tensor &x, const at::Tensor &session_ids,
                          const at::Tensor &micro_batch_ids, const at::Tensor &token_ids, const at::Tensor &expert_offsets,
                          const at::Tensor &actual_token_num, const at::Tensor &attn_rank_table, c10::string_view group, int64_t world_size,
                          c10::IntArrayRef token_info_table_shape, c10::IntArrayRef token_data_shape)
{
    char *group_ptr = const_cast<char *>(group.data());
    EXEC_NPU_CMD(aclnnFFNToAttention, x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table,
                 group_ptr, world_size, token_info_table_shape, token_data_shape);
    return;
}
}