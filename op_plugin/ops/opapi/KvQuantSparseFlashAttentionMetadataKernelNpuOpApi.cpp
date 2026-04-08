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

#include <torch/extension.h>
#include <torch/library.h>
//#include "aclnn_common.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using npu_preparation = at_npu::native::OpPreparation;
using namespace at_npu::native;
constexpr int64_t OUTPUT_SIZE = 1024;

const c10::optional<at::Tensor> get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device) {
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_kv_quant_sparse_flash_attention_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, int64_t kv_quant_mode, 
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &topk_length, c10::optional<int64_t> batch_size, c10::optional<int64_t> max_seqlen_q, c10::optional<int64_t> max_seqlen_kv,
    c10::optional<int64_t> topk, c10::optional<int64_t> tile_size, c10::optional<int64_t> rope_head_dim, c10::optional<int64_t> mask_mode, c10::optional<int64_t> win_left, c10::optional<int64_t> win_right,
    c10::string_view layout_q, c10::string_view layout_kv)
{
    const c10::string_view device = "npu:0";
    at::Device output_device = at::Device(std::string(device));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_kv.has_value()) {
        output_device = cu_seqlens_kv.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_kv.has_value()) {
        output_device = seqused_kv.value().device();
    } else if (topk_length.has_value()) {
        output_device = topk_length.value().device();
    }
    at::Tensor output = at::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device).memory_format(c10::MemoryFormat::Contiguous));

    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_kv_val = get_valid_tensor(cu_seqlens_kv, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_kv_val = get_valid_tensor(seqused_kv, output_device);
    auto topk_length_val = get_valid_tensor(topk_length, output_device);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD_V1(aclnnKvQuantSparseFlashAttentionMetadata, cu_seqlens_q_val, cu_seqlens_kv_val, seqused_q_val, 
              seqused_kv_val, topk_length_val, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, 
              max_seqlen_kv, topk, kv_quant_mode, tile_size, rope_head_dim, mask_mode, win_left, win_right, 
              layout_q_ptr, layout_kv_ptr, output);
    return output;
}
} // namespace op_api