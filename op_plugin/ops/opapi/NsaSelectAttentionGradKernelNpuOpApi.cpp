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

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_nsa_sag_output_tensor(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value)
{
    at::Tensor query_grad;
    at::Tensor key_grad;
    at::Tensor value_grad;

    query_grad = npu_preparation::apply_tensor_without_format(
        {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)}, query.options().dtype(query.dtype()));
    key_grad = npu_preparation::apply_tensor_without_format(
        {key.size(DIM_0), key.size(DIM_1), key.size(DIM_2)}, key.options().dtype(key.dtype()));
    value_grad = npu_preparation::apply_tensor_without_format(
        {value.size(DIM_0), value.size(DIM_1), value.size(DIM_2)}, value.options().dtype(value.dtype()));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(query_grad, key_grad, value_grad);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_nsa_select_attention_grad_symint(const at::Tensor &grad,
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &attention_out,
    const at::Tensor &softmax_max, const at::Tensor &softmax_sum, const at::Tensor &topk_indices, double scale_value,
    int64_t head_num, int64_t select_block_size, int64_t select_block_count,
    const c10::optional<at::Tensor> &attentionMaskOptional, c10::OptionalArrayRef<c10::SymInt> actualQSeqLenOptional,
    c10::OptionalArrayRef<c10::SymInt> actualKvSeqLenOptional)
{
    std::string input_layout = "TND";
    char *layout_ptr = input_layout.data();

    int64_t sparse_mode = 2;
    std::tuple<at::Tensor, at::Tensor, at::Tensor> output = construct_nsa_sag_output_tensor(query, key, value);
    at::Tensor query_grad = std::get<0>(output);
    at::Tensor key_grad = std::get<1>(output);
    at::Tensor value_grad = std::get<2>(output);
    // dispatch hostAPI

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnNsaSelectedAttention,
        query,
        key,
        value,
        attention_out,
        grad,
        softmax_max,
        softmax_sum,
        topk_indices,
        actualQSeqLenOptional,
        actualKvSeqLenOptional,
        attentionMaskOptional,
        scale_value,
        select_block_size,
        select_block_count,
        head_num,
        layout_ptr,
        sparse_mode,
        query_grad,
        key_grad,
        value_grad);

    return output;
}
}  // namespace op_api