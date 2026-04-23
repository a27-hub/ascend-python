// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
const static int64_t DIM_4 = 4;
const static int64_t DIM_NUMS_3 = 3;
const static int64_t DIM_NUMS_4 = 4;
const static int SHAPE_3D = 3;
const static int SHAPE_4D = 4;
const static int SHAPE_5D = 5;

const int THIRD_ELEMENT = 2;
const int FORTH_ELEMENT = 3;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;
const int LAYOUT_MAX_LENGTH = 20;
const double EPSILON = 1e-9;
const int64_t LENGTH_BIAS = 32;
const static int64_t SOFTMAXMAX_LAST_DIMSHAPE = 8;
const static int64_t MAX_SEQUENCE_LENGTH = 1000000;
const static int64_t DEFAULT_PSE_TYPE = 1;
const static int64_t DEFAULT_OUT_DTYPE = 0;
const static auto DEFAULT_START_IDX = at::IntArrayRef{};

using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

enum class SparseMode {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    PREFIX,
    PREFIX_COMPRESS,
    RIGHT_DOWN_CAUSAL_BAND,
    BAND_LEFT_UP_CAUSAL
};

static std::pair<std::string, std::string> get_query_and_attention_out_layout(
    const at::Tensor query,
    std::string input_layout_str)
{
    struct parserLayout {
        std::string qLayout;
        std::string outLayout;
        int32_t qDim;
    };

    const std::map<std::string, parserLayout> LAYOUT_MAP = {
        {"BSH",              {"BSH", "BSH", DIM_NUMS_3}},
        {"BSND",             {"BSND", "BSND", DIM_NUMS_4}},
        {"BNSD",             {"BNSD", "BNSD", DIM_NUMS_4}},
        {"TND",              {"TND", "TND", DIM_NUMS_3}},
        {"NTD",              {"NTD", "NTD", DIM_NUMS_3}},
        {"BNSD_BSND",        {"BNSD", "BSND", DIM_NUMS_4}},
        {"BSH_BNSD",         {"BSH", "BNSD", DIM_NUMS_3}},
        {"BSND_BNSD",        {"BSND", "BNSD", DIM_NUMS_4}},
        {"NTD_TND",          {"NTD", "TND", DIM_NUMS_3}},
        {"BSH_NBSD",         {"BSH", "NBSD", DIM_NUMS_3}},
        {"BSND_NBSD",        {"BSND", "NBSD", DIM_NUMS_4}},
        {"BNSD_NBSD",        {"BNSD", "NBSD", DIM_NUMS_4}},
        {"TND_NTD",          {"TND", "NTD", DIM_NUMS_3}},
        {"NSD",              {"NSD", "NSD", DIM_NUMS_3}}
    };

    std::string query_layout = "BSH";
    std::string attention_out_layout = "BSH";
    int32_t query_dim;
    auto it = LAYOUT_MAP.find(input_layout_str);
    if (it != LAYOUT_MAP.end()) {
        query_layout = it->second.qLayout;
        attention_out_layout = it->second.outLayout;
        query_dim = it->second.qDim;
        
        TORCH_CHECK(query.dim() == query_dim,
            "query's dim should be consistent with that of Layout", OPS_ERROR(ErrCode::VALUE));
    } else {
        TORCH_CHECK(
            false,
            "layout only support BSH, BSND, TND, NTD, BNSD_BSND, BSH_BNSD, BSND_BNSD, NTD_TND, ",
            "BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, but got ",
            query_layout,
            OPS_ERROR(ErrCode::VALUE)
        );
    }
    return {query_layout, attention_out_layout};
}

static std::tuple<int64_t, int64_t, int64_t, int64_t> get_query_b_n_s_d(
    const at::Tensor &query,
    std::string query_layout,
    int64_t num_heads)
{
    int64_t b = 0;
    int64_t n1 = 0;
    int64_t s1 = 0;
    int64_t d1 = 0;
    if (query_layout == "BSH") {
        b = query.size(DIM_0);
        s1 = query.size(DIM_1);
        n1 = num_heads;
        d1 = query.size(DIM_2) / num_heads;
    } else if (query_layout == "BSND") {
        b = query.size(DIM_0);
        s1 = query.size(DIM_1);
        n1 = query.size(DIM_2);
        d1 = query.size(DIM_3);
    } else if (query_layout == "BNSD") {
        b = query.size(DIM_0);
        s1 = query.size(DIM_2);
        n1 = query.size(DIM_1);
        d1 = query.size(DIM_3);
    } else if (query_layout == "NSD") {
        b = 1;
        s1 = query.size(DIM_1);
        n1 = query.size(DIM_0);
        d1 = query.size(DIM_2);
    } else {
        TORCH_CHECK(
            false,
            "It is not supported in get_query_b_n_s_d function, layout ",
            query_layout,
            OPS_ERROR(ErrCode::VALUE)
        );
    }
    return {b, n1, s1, d1};
}

static std::tuple<int64_t, int64_t, int64_t> get_query_t_n_d(
    const at::Tensor &query,
    std::string query_layout)
{
    int64_t t = 0;
    int64_t n1 = 0;
    int64_t d1 = 0;
    if (query_layout == "TND") {
        t = query.size(DIM_0);
        n1 = query.size(DIM_1);
        d1 = query.size(DIM_2);
    } else if (query_layout == "NTD") {
        t = query.size(DIM_1);
        n1 = query.size(DIM_0);
        d1 = query.size(DIM_2);
    } else {
        TORCH_CHECK(
            false,
            "It is not supported in get_query_t_n_d function, layout ",
            query_layout,
            OPS_ERROR(ErrCode::VALUE)
        );
    }
    return {t, n1, d1};
}

static int64_t get_value_d(
    const c10::optional<at::Tensor> &block_table,
    const at::Tensor &query,
    const at::Tensor &value,
    std::string query_layout,
    int64_t kv_num_heads)
{
    int64_t valueD = 0;
    if (block_table.has_value()) { // PA场景
        if (value.dim() == SHAPE_3D) {
            valueD = value.size(DIM_2) / kv_num_heads;
        } else if (value.dim() == SHAPE_4D) {
            valueD = value.size(DIM_3);
        } else if (value.dim() == SHAPE_5D) {
            valueD = value.size(DIM_2) * value.size(DIM_4);
        } else {
            TORCH_CHECK(
                false,
                "when Page Attention enabled, value's dim should be 3/4/5, but got ",
                value.dim(),
                OPS_ERROR(ErrCode::VALUE)
            );
        }
    } else { // 非PA场景
        TORCH_CHECK(
            value.dim() == query.dim(),
            "when Page Attention not enabled, value'dim should equal to query's dim!",
            OPS_ERROR(ErrCode::VALUE)
        );
        if (query_layout == "BSH") {
            valueD = value.size(DIM_2) / kv_num_heads;
        } else if (query_layout == "BSND" || query_layout == "BNSD") {
            valueD = value.size(DIM_3);
        } else if (query_layout == "TND" || query_layout == "NTD" || query_layout == "NSD") {
            valueD = value.size(DIM_2);
        }
    }
    return valueD;
}

static int get_change_d_scale(
    const at::Tensor &value)
{
    const static int changeDScale = 1;
    const static int changeDForInt32 = 8;
    // int4 伪装成 int32
    if (value.scalar_type() == at::kInt) {
        return changeDForInt32;
    }
    return changeDScale;
}

static at::Tensor infer_attention_out_shape(
    std::string attention_out_layout,
    const at::Tensor &query,
    std::string query_layout,
    int64_t num_heads,
    int64_t valueD)
{
    int64_t b = 0;
    int64_t n1 = 0;
    int64_t s1 = 0;
    int64_t d1 = 0;
    int64_t t = 0;
    at::Tensor attention_out = npu_preparation::apply_tensor_without_format(query);
    if (attention_out_layout == "BSH") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outH = num_heads * valueD;
        outH = (outH == 0 || query.size(DIM_2) == 0) ? query.size(DIM_2) : outH;
        attention_out = OpPreparation::apply_tensor_without_format(
            {b, s1, outH},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "BSND") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {b, s1, n1, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "BNSD") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {b, n1, s1, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "NBSD") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {n1, b, s1, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "TND") {
        auto [t, n1, d1] = get_query_t_n_d(query, query_layout);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {t, n1, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "NTD") {
        auto [t, n1, d1] = get_query_t_n_d(query, query_layout);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {n1, t, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "NSD") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {n1, s1, outD},
            query.options().dtype(query.dtype())
        );
    }
    return attention_out;
}

static at::Tensor infer_lse_out_shape(
    std::string input_layout_str,
    const at::Tensor &query,
    std::string query_layout,
    int64_t num_heads)
{
    int64_t b = 0;
    int64_t n1 = 0;
    int64_t s1 = 0;
    int64_t d1 = 0;
    int64_t t = 0;
    at::Tensor lse_out;
    if (input_layout_str == "TND" || input_layout_str == "NTD" ||
        input_layout_str == "TND_NTD" || input_layout_str == "NTD_TND") {
        auto [t, n1, d1] = get_query_t_n_d(query, query_layout);
        lse_out = npu_preparation::apply_tensor_without_format({t, n1, 1}, c10::dtype(c10::ScalarType::Float));
    } else {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        lse_out = npu_preparation::apply_tensor_without_format({b, n1, s1, 1}, c10::dtype(c10::ScalarType::Float));
    }
    return lse_out;
}

std::tuple<at::Tensor, at::Tensor> construct_fia_output_tensor_experimental(
    const at::Tensor &query,
    const at::Tensor &value,
    std::string input_layout_str,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_heads,
    int64_t kv_num_heads,
    bool softmax_lse_flag,
    const c10::optional<at::Tensor> &query_rope)
{
    TORCH_CHECK(
        num_heads > 0,
        "num_heads should be greater than 0, but the actual value is",
        num_heads,
        OPS_ERROR(ErrCode::VALUE)
    );
    kv_num_heads = (kv_num_heads == 0) ? num_heads : kv_num_heads;
    
    // get query_layout, attention_out_layout
    auto [query_layout, attention_out_layout] = get_query_and_attention_out_layout(query, input_layout_str);

    // cal valueD
    int64_t valueD = get_value_d(block_table, query, value, query_layout, kv_num_heads);

    // 计算changeDScale
    int changeDScale = get_change_d_scale(value);
    valueD = valueD * changeDScale;

    // 推导attenout shape
    at::Tensor tmp_output = infer_attention_out_shape(attention_out_layout, query, query_layout, num_heads, valueD);

    at::Tensor output;
    if (query.dtype() == at::kChar) {
        if (query_rope.has_value()) {
            const at::Tensor &query_rope_tensor = c10::value_or_else(query_rope, [] { return at::Tensor(); });
            output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(query_rope_tensor.dtype()));
        } else {
            output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(c10::ScalarType::Half));
        }
    } else {
        output = npu_preparation::apply_tensor_without_format(tmp_output);
    }

    // infer lseout shape
    at::Tensor softmax_lse;
    if (softmax_lse_flag) {
        softmax_lse = infer_lse_out_shape(input_layout_str, query, query_layout, num_heads);
    } else {
        softmax_lse = npu_preparation::apply_tensor_without_format({0}, c10::dtype(c10::ScalarType::Float));
    }

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> _npu_attention_pioneer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &metaData,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &key_sink,
    const c10::optional<at::Tensor> &key_rope_sink,
    const c10::optional<at::Tensor> &value_sink,
    int64_t num_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t block_size,
    bool softmax_lse_flag)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_experimental(query, value, input_layout_str,
                                                                                        block_table, num_heads, num_key_value_heads,
                                                                                        softmax_lse_flag, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    at::Tensor pse_shift;
    at::Tensor deq_scale1;
    at::Tensor quant_scale1;
    at::Tensor deq_scale2;
    at::Tensor quant_scale2;
    at::Tensor quant_offset2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor q_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_antiquant_scale;
    at::Tensor key_antiquant_offset;
    at::Tensor value_antiquant_scale;
    at::Tensor value_antiquant_offset;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    at::SymIntArrayRef default_actual_shared_prefix;
    at::Tensor key_rope_antiquant_scale;
    at::Tensor dequant_scale_query;
    at::Tensor learnablesink;
    at::SymIntArrayRef default_q_start_idx;
    at::SymIntArrayRef default_kv_start_idx;
    int64_t inner_precise = 0;
    int64_t antiquant_mode = 0;
    int64_t key_antiquant_mode = 0;
    int64_t value_antiquant_mode = 0;
    int64_t query_quant_mode = 0;
    int64_t pse_type = 1;

    if (!op_plugin::utils::is_gte_cann_version_810rc1() &&
        (query_rope.has_value() || key_rope.has_value())) {
        TORCH_CHECK(false, "query_rope|key_rope|key_rope_antiquant_scale has value, CANN version should greater than 8.1.RC1!"
            + OPS_ERROR(ErrCode::NOT_SUPPORT));
    }
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAiInfraAttentionPioneer, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
            deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table, q_padding_size, kv_padding_size,
            key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix,
            query_rope, key_rope, key_rope_antiquant_scale, dequant_scale_query, learnablesink, default_q_start_idx, default_kv_start_idx, key_sink, key_rope_sink, value_sink, metaData,
            num_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag, key_antiquant_mode,
            value_antiquant_mode, query_quant_mode, pse_type, output, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor &, at::Tensor &> _npu_attention_pioneer_out(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &metaData,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &key_sink,
    const c10::optional<at::Tensor> &key_rope_sink,
    const c10::optional<at::Tensor> &value_sink,
    int64_t num_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t block_size,
    bool softmax_lse_flag,
    const c10::optional<at::Tensor> &workspace,
    at::Tensor &attention_out,
    at::Tensor &softmax_lse)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    at::Tensor pse_shift;
    at::Tensor deq_scale1;
    at::Tensor quant_scale1;
    at::Tensor deq_scale2;
    at::Tensor quant_scale2;
    at::Tensor quant_offset2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor q_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_antiquant_scale;
    at::Tensor key_antiquant_offset;
    at::Tensor value_antiquant_scale;
    at::Tensor value_antiquant_offset;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    at::SymIntArrayRef default_actual_shared_prefix;
    at::Tensor key_rope_antiquant_scale;
    at::Tensor dequant_scale_query;
    at::Tensor learnablesink;
    at::SymIntArrayRef default_q_start_idx;
    at::SymIntArrayRef default_kv_start_idx;
    int64_t inner_precise = 0;
    int64_t antiquant_mode = 0;
    int64_t key_antiquant_mode = 0;
    int64_t value_antiquant_mode = 0;
    int64_t query_quant_mode = 0;
    int64_t pse_type = 1;

    // dispatch hostAPI
    if (!op_plugin::utils::is_gte_cann_version_810rc1() &&
        (query_rope.has_value() || key_rope.has_value())) {
        TORCH_CHECK(false, "query_rope|key_rope|key_rope_antiquant_scale has value, CANN version should greater than 8.1.RC1!"
            + OPS_ERROR(ErrCode::NOT_SUPPORT));
    }
    if (workspace.has_value()) {
        void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
        uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
        EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnnAiInfraAttentionPioneer, workspace_addr, workspace_size, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
            deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table, q_padding_size, kv_padding_size,
            key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix,
            query_rope, key_rope, key_rope_antiquant_scale, dequant_scale_query, learnablesink, default_q_start_idx, default_kv_start_idx, key_sink, key_rope_sink, value_sink, metaData,
            num_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag, key_antiquant_mode,
            value_antiquant_mode, query_quant_mode, pse_type, attention_out, softmax_lse);
    } else {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAiInfraAttentionPioneer, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
            deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table, q_padding_size, kv_padding_size,
            key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix,
            query_rope, key_rope, key_rope_antiquant_scale, dequant_scale_query, learnablesink, default_q_start_idx, default_kv_start_idx, key_sink, key_rope_sink, value_sink, metaData,
            num_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag, key_antiquant_mode,
            value_antiquant_mode, query_quant_mode, pse_type, attention_out, softmax_lse);
    }
    return std::tuple<at::Tensor&, at::Tensor&>(attention_out, softmax_lse);
}

at::Tensor _npu_attention_pioneer_get_max_workspace(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &metaData,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &key_sink,
    const c10::optional<at::Tensor> &key_rope_sink,
    const c10::optional<at::Tensor> &value_sink,
    int64_t num_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t block_size,
    bool softmax_lse_flag)
{
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_experimental(query, value, input_layout_str,
                                                                                        block_table, num_heads, num_key_value_heads,
                                                                                        softmax_lse_flag, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    // convert str
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    at::Tensor pse_shift;
    at::Tensor deq_scale1;
    at::Tensor quant_scale1;
    at::Tensor deq_scale2;
    at::Tensor quant_scale2;
    at::Tensor quant_offset2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor q_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_antiquant_scale;
    at::Tensor key_antiquant_offset;
    at::Tensor value_antiquant_scale;
    at::Tensor value_antiquant_offset;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    at::SymIntArrayRef default_actual_shared_prefix;
    at::Tensor key_rope_antiquant_scale;
    at::Tensor dequant_scale_query;
    at::Tensor learnablesink;
    at::SymIntArrayRef default_q_start_idx;
    at::SymIntArrayRef default_kv_start_idx;
    int64_t inner_precise = 0;
    int64_t antiquant_mode = 0;
    int64_t key_antiquant_mode = 0;
    int64_t value_antiquant_mode = 0;
    int64_t query_quant_mode = 0;
    int64_t pse_type = 1;

    // dispatch hostAPI
    if (!op_plugin::utils::is_gte_cann_version_810rc1() &&
        (query_rope.has_value() || key_rope.has_value())) {
        TORCH_CHECK(false, "query_rope|key_rope|key_rope_antiquant_scale has value, CANN version should greater than 8.1.RC1!"
            + OPS_ERROR(ErrCode::NOT_SUPPORT));
    }
    uint64_t workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnAiInfraAttentionPioneer, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
            deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table, q_padding_size, kv_padding_size,
            key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix,
            query_rope, key_rope, key_rope_antiquant_scale, dequant_scale_query, learnablesink, default_q_start_idx, default_kv_start_idx, key_sink, key_rope_sink, value_sink, metaData,
            num_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag, key_antiquant_mode,
            value_antiquant_mode, query_quant_mode, pse_type, output, softmax_lse);
    at::Tensor workspace_tensor = npu_preparation::apply_tensor_without_format({workspace_size}, query.options().dtype(query.dtype()));
    return workspace_tensor;
}

std::tuple<at::Tensor, at::Tensor> _npu_attention_pioneer_infer_output(
    const at::Tensor &query,
    const at::Tensor &value,
    c10::string_view input_layout,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_heads,
    int64_t num_key_value_heads,
    bool softmax_lse_flag,
    const c10::optional<at::Tensor> &query_rope)
{
    std::string input_layout_str = std::string(input_layout);
    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_experimental(query, value, input_layout_str,
                                                                                        block_table, num_heads, num_key_value_heads,
                                                                                        softmax_lse_flag, query_rope);
    return fia_output;
}

at::Tensor format_trans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
            "Expected all tensors to be on the same device. "
            "Expected NPU tensor, please check whether the input tensor device is correct.",
            OPS_ERROR(ErrCode::TYPE));
        return custom_ops::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_pioneer_backward(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &key_sink,
    const at::Tensor &value,
    const at::Tensor &value_sink,
    const at::Tensor &dy,
    int64_t head_num,
    const std::string input_layout,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &drop_mask,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_lse,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise,
    int64_t seed,
    int64_t offset,
    c10::OptionalIntArrayRef prefix,
    c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen,
    int64_t sparse_mode,
    c10::string_view softmax_layout,
    const c10::optional<at::Tensor> &sink)
{
    double scale = scale_value;
    int64_t sinkNum = 128;

    const at::Tensor &pse_const = pse.value_or(at::Tensor());
    const at::Tensor &drop_mask_const = drop_mask.value_or(at::Tensor());
    const at::Tensor &padding_mask_const = padding_mask.value_or(at::Tensor());
    const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
    const at::Tensor &softmax_max_const = softmax_max.value_or(at::Tensor());
    const at::Tensor &softmax_sum_const = softmax_sum.value_or(at::Tensor());
    const at::Tensor &softmax_lse_const = softmax_lse.value_or(at::Tensor());
    const at::Tensor &softmax_const = softmax_in.value_or(at::Tensor());
    const at::Tensor &attention_const = attention_in.value_or(at::Tensor());
    const at::Tensor &sink_const = sink.value_or(at::Tensor());
    auto prefixN = prefix.value_or(at::IntArrayRef{});
    auto ac_seq_qlen = actual_seq_qlen.value_or(at::IntArrayRef{});
    auto ac_seq_kvlen = actual_seq_kvlen.value_or(at::IntArrayRef{});

    at::Tensor format_query = format_trans(query);
    at::Tensor format_key = format_trans(key);
    at::Tensor format_key_sink = format_trans(key_sink);
    at::Tensor format_value = format_trans(value);
    at::Tensor format_value_sink = format_trans(value_sink);
    at::Tensor format_dy = format_trans(dy);


    at::Tensor format_pse = format_trans(pse_const);
    at::Tensor format_drop_mask = format_trans(drop_mask_const);
    at::Tensor format_padding_mask = format_trans(padding_mask_const);
    at::Tensor format_atten_mask = format_trans(atten_mask_const);
    at::Tensor format_softmax_max = format_trans(softmax_max_const);
    at::Tensor format_softmax_sum = format_trans(softmax_sum_const);
    at::Tensor format_softmax_lse = format_trans(softmax_lse_const);
    at::Tensor format_softmax = format_trans(softmax_const);
    at::Tensor format_attention = format_trans(attention_const);
    at::Tensor format_sink = format_trans(sink_const);
    at::Tensor dq = OpPreparation::apply_tensor_without_format(format_query);
    at::Tensor dk = OpPreparation::apply_tensor_without_format(format_key);
    at::Tensor dv = OpPreparation::apply_tensor_without_format(format_value);
    at::Tensor dkSink = OpPreparation::apply_tensor_without_format(format_key_sink);
    at::Tensor dvSink = OpPreparation::apply_tensor_without_format(format_value_sink);
    at::Tensor dpse;
    at::Tensor dsink;
    if (format_pse.defined()) {
        dpse = OpPreparation::apply_tensor_without_format(format_pse);
    } else {
        dpse = at::empty({0}, query.options());
    }
    if (format_sink.defined()) {
        dsink = OpPreparation::apply_tensor_without_format(format_sink);
    } else {
        dsink = at::empty({0}, key.options().dtype(at::kFloat));
    }

    char input_layout_char[LAYOUT_MAX_LENGTH];
    strncpy(input_layout_char, input_layout.c_str(), LAYOUT_MAX_LENGTH - 1);
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        //todo:失败的校验
    } else { // Ascend950
        c10::optional<at::Tensor> empty_optional_tensor;
        at::Tensor empty_out_tensor;
        char softmax_layout_char[LAYOUT_MAX_LENGTH];
        softmax_layout_char[LAYOUT_MAX_LENGTH - 1] = '\0';
        EXEC_NPU_CMD(
            aclnnAttentionPioneerBackwardV4, format_query, format_key, format_key_sink, format_value, format_value_sink, format_dy,
            format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
            format_softmax_sum, format_softmax_lse, format_softmax, format_attention, empty_optional_tensor, empty_optional_tensor, empty_optional_tensor,
            empty_optional_tensor, empty_optional_tensor, empty_optional_tensor, empty_optional_tensor, empty_optional_tensor,
            prefixN, ac_seq_qlen, ac_seq_kvlen,
            DEFAULT_START_IDX, DEFAULT_START_IDX, scale_value, keep_prob, pre_tockens, next_tockens, head_num,
            input_layout_char, softmax_layout_char, inner_precise, sparse_mode, DEFAULT_PSE_TYPE, seed, offset, DEFAULT_OUT_DTYPE,
            dq, dk, dkSink, dv, dvSink, empty_out_tensor, empty_out_tensor, dpse, empty_out_tensor);
    }
    // FLOP_COUNT(FlopCounter::flash_attention_backward_flop, query, key, value,
    //     dy, head_num, input_layout, actual_seq_qlen, actual_seq_kvlen);

    return std::make_tuple(dq, dk, dv, dkSink, dvSink);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> _npu_attention_pioneer_backward(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &key_sink,
    const at::Tensor &value,
    const at::Tensor &value_sink,
    const at::Tensor &dy,
    int64_t head_num,
    c10::string_view input_layout,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_lse,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    int64_t pre_tockens,
    int64_t next_tockens,
    const c10::optional<at::Tensor> &actual_seq_qlen,
    const c10::optional<at::Tensor> &actual_seq_kvlen,
    int64_t sparse_mode)
{
    at::Tensor actual_seq_qlen_cpu;
    at::Tensor actual_seq_kvlen_cpu;
    c10::OptionalIntArrayRef actual_seq_qlen_list = c10::nullopt;
    if (actual_seq_qlen.has_value() && actual_seq_qlen->defined()) {
        // std::cout << "dtype: " << actual_seq_qlen->dtype() << " device: " << actual_seq_qlen->device() << std::endl;
        actual_seq_qlen_cpu = actual_seq_qlen->to(at::kCPU).to(at::kLong).contiguous();
        // std::cout << "values: " << actual_seq_qlen_cpu << std::endl;
        actual_seq_qlen_list = c10::IntArrayRef(
            actual_seq_qlen_cpu.data_ptr<int64_t>(),
            actual_seq_qlen_cpu.numel()
        );
    }
    c10::OptionalIntArrayRef actual_seq_kvlen_list = c10::nullopt;
    if (actual_seq_kvlen.has_value() && actual_seq_kvlen->defined()) {
        actual_seq_kvlen_cpu = actual_seq_kvlen->to(at::kCPU).to(at::kLong).contiguous();
        actual_seq_kvlen_list = c10::IntArrayRef(
            actual_seq_kvlen_cpu.data_ptr<int64_t>(),
            actual_seq_kvlen_cpu.numel()
        );
    }

    c10::optional<at::Tensor> pse = c10::nullopt;
    c10::optional<at::Tensor> padding_mask = c10::nullopt;
    c10::optional<at::Tensor> softmax_max = c10::nullopt;
    c10::optional<at::Tensor> softmax_sum = c10::nullopt;
    c10::optional<at::Tensor> softmax_in = c10::nullopt;
    double keep_prob = 1.0;
    int64_t inner_precise = 0;
    int64_t seed = 0;
    int64_t offset = 0;
    int64_t numels = 0;
    c10::OptionalIntArrayRef prefix = c10::nullopt;
    c10::string_view softmax_layout = "";
    const c10::optional<at::Tensor> sink = c10::nullopt;

    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ",
        key.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key_sink.dim() == DIMENSION_3D || key_sink.dim() == DIMENSION_4D,
        "The shapes of the input key_sink should be 3 or 4 dimensional, but got ",
        key_sink.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "The shapes of the input value should be 3 or 4 dimensional, but got ",
        value.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(value_sink.dim() == DIMENSION_3D || value_sink.dim() == DIMENSION_4D,
        "The shapes of the input value_sink should be 3 or 4 dimensional, but got ",
        value_sink.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dy.dim() == DIMENSION_3D || dy.dim() == DIMENSION_4D,
        "The shapes of the input dy should be 3 or 4 dimensional, but got ", dy.dim(), "-dimensional",
        OPS_ERROR(ErrCode::PARAM));
    std::string input_layout_str = std::string(input_layout);
    // todo:
    if (input_layout_str == "TND") {
        TORCH_CHECK((sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode < static_cast<int64_t>(SparseMode::PREFIX)) ||
                    (sparse_mode > static_cast<int64_t>(SparseMode::PREFIX) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)),
                    "The sparse_mode value must be in range of [0,5) or (5,8], but got ",
                    sparse_mode, OPS_ERROR(ErrCode::PARAM));
    }
    for (auto &c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" || input_layout_str == "BNSD" ||
        input_layout_str == "BSND" || input_layout_str == "TND",
        "The input_layout should be BSH/SBH/BNSD/BSND/TND(case-insensitive), but got ",
        input_layout, OPS_ERROR(ErrCode::PARAM));

    at::Tensor drop_mask;
    auto result = npu_fusion_attention_pioneer_backward(query,
        key, key_sink, value, value_sink, dy, head_num, input_layout_str, pse, drop_mask, padding_mask, atten_mask,
        softmax_max, softmax_sum, softmax_lse, softmax_in, attention_in, scale_value, keep_prob, pre_tockens,
        next_tockens, inner_precise, seed, offset, prefix, actual_seq_qlen_list, actual_seq_kvlen_list, sparse_mode,
        softmax_layout, sink);

    c10_npu::NPUEvent npu_event;
    npu_event.record(c10_npu::getCurrentNPUStream());
    npu_event.block(c10_npu::getCurrentSecondaryStream());

    return result;
}

#endif
} // namespace op_api
