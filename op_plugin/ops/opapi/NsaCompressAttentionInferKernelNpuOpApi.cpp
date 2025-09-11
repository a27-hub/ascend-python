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

    std::tuple<at::Tensor, at::Tensor> construct_nsa_cai_output_tensor(
        const at::Tensor &query,
        const at::Tensor &value,
        int64_t num_key_value_heads,
        int64_t selectBlockCount,
        c10::string_view layout
    )
    {
        at::Tensor output;
        at::Tensor topKOutput;
        std::string input_layout = std::string(layout);
        TORCH_CHECK(input_layout == "TND" || input_layout == "BSND", "layout only support TND or BSND now.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(num_key_value_heads > 0, "key_value_head_num must be greater than 0." + OPS_ERROR(ErrCode::VALUE));
        int64_t head_dim_vo = value.size(DIM_2) / num_key_value_heads;
        if (input_layout == "TND") {
            output = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), head_dim_vo}, query.options().dtype(query.dtype()));
            topKOutput = npu_preparation::apply_tensor_without_format({query.size(DIM_0), num_key_value_heads, selectBlockCount}, c10::ScalarType::Int);
        } else {
            output = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), head_dim_vo}, query.options().dtype(query.dtype()));
            topKOutput = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), num_key_value_heads, selectBlockCount}, c10::ScalarType::Int);
        }
        return std::tuple<at::Tensor, at::Tensor>(output, topKOutput);
    }

    std::tuple<at::Tensor, at::Tensor> npu_nsa_compress_attention_infer_symint(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        double scaleValue, int64_t numHeads, int64_t numKeyValueHeads,
        int64_t selectBlockSize,   int64_t selectBlockCount, int64_t pageBlockSize,
        int64_t compressBlockSize, int64_t compressBlockStride, c10::string_view layout,
        const c10::optional<at::Tensor> &attentionMaskOptional,
        const c10::optional<at::Tensor> &blockTableOptional,
        const c10::optional<at::Tensor> &topKMaskOptional,
        c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualCmpKvSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualSelKvSeqLenOptional)
    {
        // convert str
        std::string input_layout = std::string(layout);
        char *layOut = input_layout.data();

        int64_t sparse_mode = 0;
        // construct the output tensor
        std::tuple<at::Tensor, at::Tensor> nsa_compress_attninfer_output = construct_nsa_cai_output_tensor(query, value,
            numKeyValueHeads, selectBlockCount, layout);
        at::Tensor output = std::get<0>(nsa_compress_attninfer_output);
        at::Tensor topKOutput = std::get<1>(nsa_compress_attninfer_output);
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaCompressAttentionInfer, query, key, value, attentionMaskOptional, blockTableOptional,
            actualQSeqLenOptional, actualCmpKvSeqLenOptional, actualSelKvSeqLenOptional, topKMaskOptional, numHeads, numKeyValueHeads,
            selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride, scaleValue, layOut, pageBlockSize,
            sparse_mode, output, topKOutput);
        return std::tuple<at::Tensor, at::Tensor>(output, topKOutput);
    }

    std::tuple<at::Tensor &, at::Tensor &> npu_nsa_compress_attention_infer_out_symint(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        double scaleValue, int64_t numHeads, int64_t numKeyValueHeads,
        int64_t selectBlockSize,   int64_t selectBlockCount, int64_t pageBlockSize,
        int64_t compressBlockSize, int64_t compressBlockStride, c10::string_view layout,
        const c10::optional<at::Tensor> &attentionMaskOptional,
        const c10::optional<at::Tensor> &blockTableOptional,
        const c10::optional<at::Tensor> &topKMaskOptional,
        c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualCmpKvSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualSelKvSeqLenOptional,
        const c10::optional<at::Tensor> &workspace,
        at::Tensor &output, at::Tensor &topKOutput)
    {
        std::string input_layout = std::string(layout);
        char *layOut = input_layout.data();
        int64_t sparse_mode = 0;
        if (workspace.has_value()) {
            void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
            uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
            EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaCompressAttentionInfer, workspace_addr, workspace_size, query, key, value, attentionMaskOptional, blockTableOptional,
                                                actualQSeqLenOptional, actualCmpKvSeqLenOptional, actualSelKvSeqLenOptional, topKMaskOptional, numHeads,
                                                numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride,
                                                scaleValue, layOut, pageBlockSize, sparse_mode, output, topKOutput);
        } else {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaCompressAttentionInfer, query, key, value, attentionMaskOptional, blockTableOptional,
                actualQSeqLenOptional, actualCmpKvSeqLenOptional, actualSelKvSeqLenOptional, topKMaskOptional, numHeads,
                numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride,
                scaleValue, layOut, pageBlockSize, sparse_mode, output, topKOutput);
        }
        return std::tuple<at::Tensor&, at::Tensor&>(output, topKOutput);
    }

    at::Tensor _npu_nsa_compress_attention_infer_get_max_workspace_symint(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        double scaleValue, int64_t numHeads, int64_t numKeyValueHeads,
        int64_t selectBlockSize,   int64_t selectBlockCount, int64_t pageBlockSize,
        int64_t compressBlockSize, int64_t compressBlockStride, c10::string_view layout,
        const c10::optional<at::Tensor> &attentionMaskOptional,
        const c10::optional<at::Tensor> &blockTableOptional,
        const c10::optional<at::Tensor> &topKMaskOptional,
        c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualCmpKvSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualSelKvSeqLenOptional)
    {
        // convert str
        std::string input_layout = std::string(layout);
        char *layOut = input_layout.data();
        int64_t sparse_mode = 0;

        // construct the output tensor
        std::tuple<at::Tensor, at::Tensor> nsa_compress_attninfer_output = construct_nsa_cai_output_tensor(query, value,
            numKeyValueHeads, selectBlockCount, layout);
        at::Tensor output = std::get<0>(nsa_compress_attninfer_output);
        at::Tensor topKOutput = std::get<1>(nsa_compress_attninfer_output);

        // dispatch hostAPI
        uint64_t workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnNsaCompressAttentionInfer, query, key, value, attentionMaskOptional, blockTableOptional,
            actualQSeqLenOptional, actualCmpKvSeqLenOptional, actualSelKvSeqLenOptional, topKMaskOptional, numHeads,
            numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride,
            scaleValue, layOut, pageBlockSize, sparse_mode, output, topKOutput);
        at::Tensor workspace_tensor = npu_preparation::apply_tensor_without_format({workspace_size}, query.options().dtype(query.dtype()));
        return workspace_tensor;
    }
} // namespace op_api