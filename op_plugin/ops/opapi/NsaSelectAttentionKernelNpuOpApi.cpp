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

    std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_nsa_sa_output_tensor(
        const at::Tensor &query, const at::Tensor &value
    )
    {
        at::Tensor output;
        at::Tensor softmax_max_out;
        at::Tensor softmax_sum_out;
        
        output = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), value.size(DIM_2)}, query.options().dtype(query.dtype()));
        softmax_max_out = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), 8}, c10::dtype(c10::ScalarType::Float));
        softmax_sum_out = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), 8}, c10::dtype(c10::ScalarType::Float));
        return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, softmax_max_out, softmax_sum_out);
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_nsa_select_attention_symint(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        const at::Tensor &topk_indices,
        double scale_value, int64_t head_num, int64_t select_block_size,
        int64_t select_block_count,
        const c10::optional<at::Tensor> &attentionMaskOptional,
        c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualKvSeqLenOptional
    )
    {
        std::string input_layout = "TND";
        char *layout_ptr = input_layout.data();

        int64_t sparse_mode = 2;
        std::tuple<at::Tensor, at::Tensor, at::Tensor> output = construct_nsa_sa_output_tensor(query, value);
        at::Tensor softmax_max_out = std::get<1>(output);
        at::Tensor softmax_sum_out = std::get<2>(output);
        at::Tensor attention_out = std::get<0>(output);
        // dispatch hostAPI
        
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaSelectedAttention, query, key, value, topk_indices, attentionMaskOptional,
            actualQSeqLenOptional, actualKvSeqLenOptional, scale_value, head_num, layout_ptr, sparse_mode,
            select_block_size, select_block_count, softmax_max_out, softmax_sum_out, attention_out);
        
        return output;
    }

    std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_nsa_select_attention_out_symint(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        const at::Tensor &topk_indices,
        double scale_value, int64_t head_num, int64_t select_block_size,
        int64_t select_block_count,
        const c10::optional<at::Tensor> &attentionMaskOptional,
        c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualKvSeqLenOptional,
        const c10::optional<at::Tensor> &workspace,
        at::Tensor &attention_out, at::Tensor &softmax_max_out, at::Tensor &softmax_sum_out
    )
    {
        std::string input_layout = "TND";
        char *layout_ptr = input_layout.data();
        int64_t sparse_mode = 2;
        
        if (workspace.has_value()) {
            void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
            uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
            EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaSelectedAttention, workspace_addr, workspace_size, query, key, value, topk_indices, attentionMaskOptional,
                actualQSeqLenOptional, actualKvSeqLenOptional, scale_value, head_num, layout_ptr, sparse_mode,
                select_block_size, select_block_count, softmax_max_out, softmax_sum_out, attention_out);
        } else {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaSelectedAttention, query, key, value, topk_indices, attentionMaskOptional,
                actualQSeqLenOptional, actualKvSeqLenOptional, scale_value, head_num, layout_ptr, sparse_mode,
                select_block_size, select_block_count, softmax_max_out, softmax_sum_out, attention_out);
        }
        return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(attention_out, softmax_max_out, softmax_sum_out);
    }

    at::Tensor _npu_nsa_select_attention_get_max_workspace_symint(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        const at::Tensor &topk_indices,
        double scale_value, int64_t head_num, int64_t select_block_size,
        int64_t select_block_count,
        const c10::optional<at::Tensor> &attentionMaskOptional,
        c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
        c10::OptionalArrayRef<c10::SymInt> actualKvSeqLenOptional
    )
    {
        std::string input_layout = "TND";
        char *layout_ptr = input_layout.data();

        int64_t sparse_mode = 2;
        std::tuple<at::Tensor, at::Tensor, at::Tensor> output = construct_nsa_sa_output_tensor(query, value);
        at::Tensor softmax_max_out = std::get<1>(output);
        at::Tensor softmax_sum_out = std::get<2>(output);
        at::Tensor attention_out = std::get<0>(output);
        
        uint64_t workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnNsaSelectedAttention, query, key, value, topk_indices, attentionMaskOptional,
            actualQSeqLenOptional, actualKvSeqLenOptional, scale_value, head_num, layout_ptr, sparse_mode,
            select_block_size, select_block_count, softmax_max_out, softmax_sum_out, attention_out);
        at::Tensor workspace_tensor = npu_preparation::apply_tensor_without_format({workspace_size}, query.options().dtype(query.dtype()));
        return workspace_tensor;
    }
} // namespace op_api