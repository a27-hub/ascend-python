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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_gather_selection_kv_cache(
    at::Tensor& selection_k_rope,
    at::Tensor& selection_kv_cache,
    at::Tensor& selection_kv_block_table,
    at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size)
{
    TORCH_CHECK((selection_k_rope.dim() == 3), "3D tensor expected for input selection_k_rope" + OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, SIZE> selection_kv_actual_seq_shape = {selection_kv_block_table.size(0)};
    at::Tensor selection_kv_actual_seq = npu_preparation::apply_tensor_without_format(selection_kv_actual_seq_shape,
                                                                                     selection_kv_block_table.options());
    
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnGatherSelectionKvCache, selection_k_rope, selection_kv_cache, selection_kv_block_table,
                 selection_kv_block_status, selection_topk_indices, full_k_rope, full_kv_cache, full_kv_block_table,
                 full_kv_actual_seq, full_q_actual_seq, selection_topk_block_size, selection_kv_actual_seq);                                                                                 

    return selection_kv_actual_seq;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gather_selection_kv_cache_functional(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size)
{
    TORCH_CHECK((selection_k_rope.dim() == 3), "3D tensor expected for input selection_k_rope" + OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, SIZE> selection_kv_actual_seq_shape = {selection_kv_block_table.size(0)};
    at::Tensor selection_kv_actual_seq = npu_preparation::apply_tensor_without_format(selection_kv_actual_seq_shape,
                                                                                     selection_kv_block_table.options());
    // 保证接口对外为functional
    at::Tensor selection_k_rope_inplace = selection_k_rope.clone();
    at::Tensor selection_kv_cache_inplace = selection_kv_cache.clone();
    at::Tensor selection_kv_block_table_inplace = selection_kv_block_table.clone();
    at::Tensor selection_kv_block_status_inplace = selection_kv_block_status.clone();
    
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnGatherSelectionKvCache, selection_k_rope_inplace, selection_kv_cache_inplace, selection_kv_block_table_inplace,
                                 selection_kv_block_status_inplace, selection_topk_indices, full_k_rope, full_kv_cache, full_kv_block_table,
                                 full_kv_actual_seq, full_q_actual_seq, selection_topk_block_size, selection_kv_actual_seq);                                                                                 

    return std::tie(selection_kv_actual_seq, selection_k_rope_inplace, selection_kv_cache_inplace, selection_kv_block_table_inplace, selection_kv_block_status);
}

}