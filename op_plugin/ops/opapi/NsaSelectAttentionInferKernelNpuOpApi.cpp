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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
    const static int64_t DIM_0 = 0;
    const static int64_t DIM_1 = 1;
    const static int64_t DIM_2 = 2;
    const static int64_t DIM_3 = 3;
    using namespace at_npu::native;
    using npu_preparation = at_npu::native::OpPreparation;

    at::Tensor construct_nsa_sai_output_tensor(
        const at::Tensor &query,
        const at::Tensor &value,
        int64_t head_num,
        int64_t key_value_head_num,
        c10::string_view layout
    )
    {
        std::string input_layout = std::string(layout);
        at::Tensor output;
        TORCH_CHECK(input_layout == "BSH" || input_layout == "BSND" || input_layout == "TND", "layout only support BSH or BSND or TND now.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(key_value_head_num >0, "key_value_head_num must be greater than 0." + OPS_ERROR(ErrCode::VALUE));
        auto key_head_dim = value.size(DIM_2) / key_value_head_num;
        if (input_layout == "BSH") {
            output = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), head_num * key_head_dim}, query.options().dtype(query.dtype()));
        } else if (input_layout == "TND") {
            output = npu_preparation::apply_tensor_without_format({query.size(DIM_0), head_num, key_head_dim}, query.options().dtype(query.dtype()));
        } else {
            output = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), value.size(DIM_3)}, query.options().dtype(query.dtype()));
        }
        return output;
    }

    at::Tensor npu_nsa_select_attention_infer_symint(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        const at::Tensor &topk_indices,
        double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size,
        int64_t select_block_count, int64_t page_block_size, c10::string_view layout,
        const c10::optional<at::Tensor> &attentionMaskOptional,
        const c10::optional<at::Tensor> &blockTableOptional,
        c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualKvSeqLenOptional
    )
    {
        std::string input_layout = std::string(layout);
        char *layout_ptr = input_layout.data();

        int64_t sparse_mode = 0;
        at::Tensor output = construct_nsa_sai_output_tensor(query, value, head_num, key_value_head_num, layout);
        // dispatch hostAPI
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaSelectedAttentionInfer, query, key, value, topk_indices, attentionMaskOptional, blockTableOptional,
            actualQSeqLenOptional, actualKvSeqLenOptional, layout_ptr, head_num, key_value_head_num,
            select_block_size, select_block_count, page_block_size, scale_value, sparse_mode, output);
        return output;
    }

    at::Tensor&  npu_nsa_select_attention_infer_out_symint(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        const at::Tensor &topk_indices,
        double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size,
        int64_t select_block_count, int64_t page_block_size, c10::string_view layout,
        const c10::optional<at::Tensor> &attentionMaskOptional,
        const c10::optional<at::Tensor> &blockTableOptional,
        c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualKvSeqLenOptional,
        const c10::optional<at::Tensor> &workspace, at::Tensor &output
    )
    {
        std::string input_layout = std::string(layout);
        char *layout_ptr = input_layout.data();

        int64_t sparse_mode = 0;
        if (workspace.has_value()) {
            void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
            uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
            EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaSelectedAttentionInfer, workspace_addr, workspace_size, query, key, value, topk_indices, attentionMaskOptional, blockTableOptional,
                actualQSeqLenOptional, actualKvSeqLenOptional, layout_ptr, head_num, key_value_head_num,
                select_block_size, select_block_count, page_block_size, scale_value, sparse_mode, output);
        } else {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaSelectedAttentionInfer, query, key, value, topk_indices, attentionMaskOptional, blockTableOptional,
                actualQSeqLenOptional, actualKvSeqLenOptional, layout_ptr, head_num, key_value_head_num,
                select_block_size, select_block_count, page_block_size, scale_value, sparse_mode, output);
        }
        return output;
    }

    at::Tensor _npu_nsa_select_attention_infer_get_max_workspace_symint(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        const at::Tensor &topk_indices,
        double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size,
        int64_t select_block_count, int64_t page_block_size, c10::string_view layout,
        const c10::optional<at::Tensor> &attentionMaskOptional,
        const c10::optional<at::Tensor> &blockTableOptional,
        c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualKvSeqLenOptional
    )
    {
        std::string input_layout = std::string(layout);
        char *layout_ptr = input_layout.data();

        int64_t sparse_mode = 0;
        at::Tensor output = construct_nsa_sai_output_tensor(query, value, head_num, key_value_head_num, layout);
        uint64_t workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnNsaSelectedAttentionInfer, query, key, value, topk_indices, attentionMaskOptional, blockTableOptional,
            actualQSeqLenOptional, actualKvSeqLenOptional, layout_ptr, head_num, key_value_head_num,
            select_block_size, select_block_count, page_block_size, scale_value, sparse_mode, output);
        at::Tensor workspace_tensor = npu_preparation::apply_tensor_without_format({workspace_size}, query.options().dtype(query.dtype()));
        return workspace_tensor;
    }
} // namespace op_api