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
void npu_attention_to_ffn(const at::Tensor &x, const at::Tensor &session_id,
                          const at::Tensor &micro_batch_id, const at::Tensor &layer_id,
                          const at::Tensor &expert_ids, const at::Tensor &expert_rank_table,
                          c10::string_view group, int64_t world_size,
                          c10::IntArrayRef ffn_token_info_table_shape,
                          c10::IntArrayRef ffn_token_data_shape,
                          c10::IntArrayRef attn_token_info_table_shape,
                          int64_t moe_expert_num,
                          const c10::optional<at::Tensor> &scales,
                          const c10::optional<at::Tensor> &active_mask,
                          int64_t quant_mode, int64_t sync_flag,
                          int64_t ffn_start_rank_id)
{
    char *group_ptr = const_cast<char *>(group.data());
    EXEC_NPU_CMD(aclnnAttentionToFFN, x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, scales, 
        active_mask, group_ptr, world_size, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape,
        moe_expert_num, quant_mode, sync_flag, ffn_start_rank_id);
    return;
}
}