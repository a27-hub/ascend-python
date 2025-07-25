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
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
const static int FLASH_THRESHOLD = 512;
const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> construct_fia_output_tensor_v2(
    const at::Tensor &query,
    const at::Tensor &value,
    c10::optional<int64_t> value_dtype,
    std::string input_layout_str,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads,
    int64_t num_key_value_heads,
    bool return_softmax_lse,
    const c10::optional<at::Tensor> &query_rope)
{
    at::Tensor output;
    int64_t batchSize = 1;
    int64_t qsSize = 1;
    TORCH_CHECK(num_query_heads > 0, "num_query_heads should be greater than 0, but the actual value is ",
        num_query_heads, OPS_ERROR(ErrCode::VALUE));
    if (num_key_value_heads == 0) {
        num_key_value_heads = num_query_heads;
    }
    int64_t changeDScale = 1;
    int64_t changeDForInt32 = 8;
    int64_t changeDForFP4 = 2;
    // int4 伪装成 int32
    if (value.scalar_type() == at::kInt) {
        changeDScale = changeDForInt32;
    }
    // fp4 伪装成 uint8
    if (value_dtype.has_value() && (c10_npu::GetAclDataType(value_dtype.value()) == aclDataType::ACL_FLOAT4_E1M2 ||
        c10_npu::GetAclDataType(value_dtype.value()) == aclDataType::ACL_FLOAT4_E2M1)) {
        changeDScale = changeDForFP4;
    }
    at::Tensor tmp_output = npu_preparation::apply_tensor_without_format(query);
    if (input_layout_str == "BNSD_BSND") {
        if (block_table.has_value() && value.dim() == 3) {  // 3 for dim num
            int64_t outputD = value.size(DIM_2) / num_key_value_heads * changeDScale;
            outputD = (outputD == 0 || query.size(DIM_3) == 0) ? query.size(DIM_3) : outputD;
            tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_2), query.size(DIM_1), outputD},
                query.options().dtype(query.dtype()));
        } else {
            int64_t outputD = value.size(DIM_3) * changeDScale;
            outputD = (outputD == 0 || query.size(DIM_3) == 0) ? query.size(DIM_3) : outputD;
            tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_2), query.size(DIM_1), outputD},
                query.options().dtype(query.dtype()));
        }
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BNSD_NBSD") {
        tmp_output = OpPreparation::apply_tensor_without_format(
            {query.size(DIM_1), query.size(DIM_0), query.size(DIM_2), query.size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BSND_NBSD") {
        tmp_output = OpPreparation::apply_tensor_without_format(
            {query.size(DIM_2), query.size(DIM_0), query.size(DIM_1), query.size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSH_NBSD") {
        tmp_output = OpPreparation::apply_tensor_without_format(
            {num_query_heads, query.size(DIM_0), query.size(DIM_1), query.size(DIM_2) / num_query_heads},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "TND_NTD") {
        tmp_output = OpPreparation::apply_tensor_without_format(
            {query.size(DIM_1), query.size(DIM_0), query.size(DIM_2)},
            query.options().dtype(query.dtype()));
    } else if (input_layout_str == "NSD") {
        int64_t outputD = value.size(DIM_2) * changeDScale;
        outputD = (outputD == 0 || query.size(DIM_2) == 0) ? query.size(DIM_2) : outputD;
        tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), outputD},
            query.options().dtype(query.dtype()));
        batchSize = 1;
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSH") {
        int64_t outputH = value.size(DIM_2) / num_key_value_heads * num_query_heads * changeDScale;
        outputH = (outputH == 0 || query.size(DIM_2) == 0) ? query.size(DIM_2) : outputH;
        tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), outputH},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSND") {
        if (!block_table.has_value()) {
            int64_t outputD = value.size(DIM_3) * changeDScale;
            outputD = (outputD == 0 || query.size(DIM_3) == 0) ? query.size(DIM_3) : outputD;
            tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), outputD},
                query.options().dtype(query.dtype()));
        } else {
            int64_t outputD = value.size(DIM_2) / num_key_value_heads * changeDScale;
            outputD = (outputD == 0 || query.size(DIM_3) == 0) ? query.size(DIM_3) : outputD;
            tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), outputD},
                query.options().dtype(query.dtype()));
        }
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BNSD") {
        if (block_table.has_value() && value.dim() == 3) { // 3 for dim num
            int64_t outputD = value.size(DIM_2) / num_key_value_heads * changeDScale;
            outputD = (outputD == 0 || query.size(DIM_3) == 0) ? query.size(DIM_3) : outputD;
            tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), outputD},
                query.options().dtype(query.dtype()));
        } else {
            int64_t outputD = value.size(DIM_3) * changeDScale;
            outputD = (outputD == 0 || query.size(DIM_3) == 0) ? query.size(DIM_3) : outputD;
            tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), outputD},
                query.options().dtype(query.dtype()));
        }
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "TND") {
        if (block_table.has_value() && value.dim() == 4) { // 4 for dim num
            int64_t outputD = value.size(DIM_3) * changeDScale;
            outputD = (outputD == 0 || query.size(DIM_2) == 0) ? query.size(DIM_2) : outputD;
            tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), outputD},
                query.options().dtype(query.dtype()));
        } else if (block_table.has_value() && value.dim() == 3) { // 3 for dim num
            int64_t outputD = value.size(DIM_2) / num_key_value_heads * changeDScale;
            outputD = (outputD == 0 || query.size(DIM_2) == 0) ? query.size(DIM_2) : outputD;
            tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), outputD},
                query.options().dtype(query.dtype()));
        } else {
            int64_t outputD = value.size(DIM_2) * changeDScale;
            outputD = (outputD == 0 || query.size(DIM_2) == 0) ? query.size(DIM_2) : outputD;
            tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), outputD},
                query.options().dtype(query.dtype()));
        }
        qsSize = query.size(DIM_0);
    } else if (input_layout_str == "NTD_TND") {
        tmp_output = OpPreparation::apply_tensor_without_format(
            {query.size(DIM_1), query.size(DIM_0), value.size(DIM_2)},
            query.options().dtype(query.dtype()));
    }
    if (quant_scale_out.has_value()) {
        output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(c10::ScalarType::Char));
    } else if (query.dtype() == at::kChar) {
        if (query_rope.has_value()) {
            const at::Tensor &query_rope_tensor = c10::value_or_else(query_rope, [] { return at::Tensor(); });
            output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(query_rope_tensor.dtype()));
        } else {
            output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(c10::ScalarType::Half));
        }
    } else {
        output = npu_preparation::apply_tensor_without_format(tmp_output);
    }

    at::Tensor softmax_lse;
    if (input_layout_str == "TND") {
        if (block_table.has_value()) { // IFA目前TND只支持PA场景，PFA目前TND只支持非PA场景
            softmax_lse = npu_preparation::apply_tensor_without_format({query.size(DIM_0), num_query_heads, 1},
            c10::dtype(c10::ScalarType::Float));
        } else {
            softmax_lse = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), 1},
            c10::dtype(c10::ScalarType::Float));
        }
    } else if (input_layout_str == "NTD_TND") {
        softmax_lse = npu_preparation::apply_tensor_without_format({query.size(DIM_1), query.size(DIM_0), 1},
            c10::dtype(c10::ScalarType::Float));
    } else {
        softmax_lse = npu_preparation::apply_tensor_without_format({batchSize, num_query_heads, qsSize, 1},
            c10::dtype(c10::ScalarType::Float));
    }

    if (!return_softmax_lse) {
        softmax_lse = npu_preparation::apply_tensor_without_format({1}, c10::dtype(c10::ScalarType::Float));
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_v2_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_v2(query, value, value_dtype, input_layout_str, quant_scale_out, block_table,
                                                                                           num_query_heads, num_key_value_heads, return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::ArrayRef<c10::SymInt> actual_shared_prefix_len;
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    bool use_wrapper = query_dtype.has_value() || key_dtype.has_value() || value_dtype.has_value();
    if (use_wrapper) {
        TensorWrapper query_wrapper = {query, (query_dtype.has_value()) ?
            c10_npu::GetAclDataType(query_dtype.value()) : npu_preparation::convert_to_acl_data_type(query.scalar_type())};
        TensorListWrapper keyTensors_wrapper = {keyTensors, (key_dtype.has_value()) ?
            c10_npu::GetAclDataType(key_dtype.value()) : npu_preparation::convert_to_acl_data_type(key.scalar_type())};
        TensorListWrapper valueTensors_wrapper = {valueTensors, (value_dtype.has_value()) ?
            c10_npu::GetAclDataType(value_dtype.value()) : npu_preparation::convert_to_acl_data_type(value.scalar_type())};
        
        at::Tensor null_tensor;
        TensorWrapper query_rope_wrapper = {query_rope.has_value() ? query_rope.value() : null_tensor,
            query_rope_dtype.has_value() ? c10_npu::GetAclDataType(query_rope_dtype.value()) : npu_preparation::convert_to_acl_data_type(query.scalar_type())};
        TensorWrapper key_rope_wrapper = {key_rope.has_value() ? key_rope.value() : null_tensor,
            key_rope_dtype.has_value() ? c10_npu::GetAclDataType(key_rope_dtype.value()) : npu_preparation::convert_to_acl_data_type(key.scalar_type())};
        TensorWrapper dequant_scale_key_wrapper = {dequant_scale_key.has_value() ? dequant_scale_key.value() : null_tensor,
            dequant_scale_key_dtype.has_value() ? c10_npu::GetAclDataType(dequant_scale_key_dtype.value()) : npu_preparation::convert_to_acl_data_type(key.scalar_type())};
        TensorWrapper dequant_scale_value_wrapper = {dequant_scale_value.has_value() ? dequant_scale_value.value() : null_tensor,
            dequant_scale_value_dtype.has_value() ? c10_npu::GetAclDataType(dequant_scale_value_dtype.value()) : npu_preparation::convert_to_acl_data_type(key.scalar_type())};

        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, query_wrapper, keyTensors_wrapper, valueTensors_wrapper, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key_wrapper, dequant_offset_key, dequant_scale_value_wrapper,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope_wrapper, key_rope_wrapper, dequant_scale_key_rope, dequant_scale_query, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, output, softmax_lse);
    } else {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key, dequant_offset_key, dequant_scale_value,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, dequant_scale_key_rope, dequant_scale_query, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, output, softmax_lse);
    }

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor &, at::Tensor &> npu_fused_infer_attention_v2_out_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype,
    const c10::optional<at::Tensor> &workspace,
    at::Tensor &attention_out,
    at::Tensor &softmax_lse)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::ArrayRef<c10::SymInt> actual_shared_prefix_len;
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    if (workspace.has_value()) {
        void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
        uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
        EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, workspace_addr, workspace_size, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key, dequant_offset_key, dequant_scale_value,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, dequant_scale_key_rope, dequant_scale_query, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, attention_out, softmax_lse);
    } else {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key, dequant_offset_key, dequant_scale_value,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, dequant_scale_key_rope, dequant_scale_query, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, attention_out, softmax_lse);
    }
    return std::tuple<at::Tensor&, at::Tensor&>(attention_out, softmax_lse);
}

at::Tensor _npu_fused_infer_attention_v2_get_max_workspace_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_v2(query, value, value_dtype, input_layout_str, quant_scale_out, block_table,
                                                                                           num_query_heads, num_key_value_heads, return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    // convert str
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::ArrayRef<c10::SymInt> actual_shared_prefix_len;
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    uint64_t workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnFusedInferAttentionScoreV4, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
        quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key, dequant_offset_key, dequant_scale_value,
        dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, dequant_scale_key_rope, dequant_scale_query, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
        num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, output, softmax_lse);
    at::Tensor workspace_tensor = npu_preparation::apply_tensor_without_format({workspace_size}, query.options().dtype(query.dtype()));
    return workspace_tensor;
}

#endif
} // namespace op_api
