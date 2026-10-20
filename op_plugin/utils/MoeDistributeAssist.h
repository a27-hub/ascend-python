// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#ifndef OP_PLUGIN_UTILS_MOE_DISTRIBUTE_ASSIST_H
#define OP_PLUGIN_UTILS_MOE_DISTRIBUTE_ASSIST_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include "op_plugin/utils/op_api_common.h"

namespace op_plugin {
namespace utils {

inline bool IsMoeRoceHierarchy(c10::string_view comm_alg)
{
    return comm_alg == "hierarchy" && c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950;
}

inline int64_t GetMoeRoceAssistSize(int64_t global_bs, int64_t ep_world_size,
                                   int64_t moe_expert_num, int64_t k)
{
    TORCH_CHECK(ep_world_size > 0 && global_bs > 0 && global_bs % ep_world_size == 0,
                "RoCE requires positive ep_world_size and global_bs divisible by ep_world_size.",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(moe_expert_num > 0 && moe_expert_num % ep_world_size == 0 && k > 0,
                "RoCE requires ordinary experts evenly sharded across EP and positive topk.",
                OPS_ERROR(ErrCode::PARAM));
    const int64_t limit = std::numeric_limits<int64_t>::max();
    const int64_t local_experts = moe_expert_num / ep_world_size;
    const int64_t routes_per_token = std::min(local_experts, k);
    const int64_t tokens_per_rank = global_bs / ep_world_size;
    TORCH_CHECK(global_bs <= limit / routes_per_token / 3 && tokens_per_rank <= limit - 15,
                "RoCE assist capacity exceeds int64 range.", OPS_ERROR(ErrCode::PARAM));
    const int64_t triple_elements = global_bs * routes_per_token * 3;
    const int64_t bitmap_elements_per_rank = ((tokens_per_rank + 15) / 16 * 16) / 2;
    TORCH_CHECK(ep_world_size <= (limit - triple_elements) / bitmap_elements_per_rank,
                "RoCE assist capacity exceeds int64 range.", OPS_ERROR(ErrCode::PARAM));
    // Three int32 values per expert row, followed immediately by uint16[P, Align16(B)].
    return triple_elements + ep_world_size * bitmap_elements_per_rank;
}

} // namespace utils
} // namespace op_plugin

#endif
