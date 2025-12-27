// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <cstring>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int THIRD_ELEMENT = 2;
const int FORTH_ELEMENT = 3;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;
const int LAYOUT_MAX_LENGTH = 20;
const double EPSILON = 1e-9;
const int64_t LENGTH_BIAS = 32;
const static int FLASH_THRESHOLD = 512;
const static int64_t SOFTMAXMAX_LAST_DIMSHAPE = 8;
const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;
const static int64_t MAX_SEQUENCE_LENGTH = 1000000;
const static int64_t DEFAULT_PSE_TYPE = 1;
const static int64_t DEFAULT_OUT_DTYPE = 0;
const static auto DEFAULT_START_IDX = at::IntArrayRef{};
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

enum class DropOutStatus {
    DROPOUT_NORMAL = 0,
    DROPOUT_NONE,
    DROPOUT_ALL
};

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

#if VERSION_BETWEEN(V2R6, VERSION_NEWEST)
at::Tensor dropout_gen_mask(const at::Tensor &query, const at::Tensor &key, double keep_prob, int64_t head_num,
    std::string input_layout, bool gen_mask_parallel, bool sync, at::Tensor* seed, at::Tensor* offset, SymInt& numels)
{
    at::Tensor drop_mask;
    if (input_layout == "BSH") {
        numels = query.size(0) * head_num * query.size(1) * key.size(1); // [B,N,S,S]
    } else if (input_layout == "SBH") {
        numels = query.size(1) * head_num * query.size(0) * key.size(0); // [B,N,S,S]
    } else if (input_layout == "BNSD") {
        numels = query.size(0) * query.size(1) * query.size(THIRD_ELEMENT) * key.size(THIRD_ELEMENT); // [B,N,S,S]
    } else if (input_layout == "BSND") {
        numels = query.size(0) * query.size(THIRD_ELEMENT) * query.size(1) * key.size(1); // [B,N,S,S]
    }
    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    length += LENGTH_BIAS;
    auto is_capture = c10_npu::currentStreamCaptureStatusMayInitCtx();
    if (is_capture == c10_npu::CaptureStatus::None) {
        if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
            const auto gen = at_npu::detail::getDefaultNPUGenerator();
            auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
            seed = static_cast<int64_t>(pair.first);
            offset = static_cast<int64_t>(pair.second);
            if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
                drop_mask = dropout_gen_mask_dispatch(query, keep_prob, seed, offset, numels, gen_mask_parallel, sync);
            }
        } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
            drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    } else {

    }

    }

    return drop_mask;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_flash_attention(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, c10::string_view input_layout,
    const c10::optional<at::Tensor> &pse, const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise,
    c10::OptionalArrayRef<c10::SymInt> prefix, c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync,
    c10::string_view softmax_layout, const c10::optional<at::Tensor> &sink)
{
    const at::Tensor &pse_const = pse.value_or(at::Tensor());
    const at::Tensor &padding_mask_const = padding_mask.value_or(at::Tensor());
    const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
    const at::Tensor &sink_const = sink.value_or(at::Tensor());
    auto prefixN = prefix.value_or(at::IntArrayRef{});
    auto ac_seq_qlen = actual_seq_qlen.value_or(at:SymIntArrayRef{});
    auto ac_seq_kvlen = actual_seq_kvlen.value_or(at::SymIntArrayRef{});

        TORCH_CHECK(head_num > 0, "head_num must > 0, but got ", head_num, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "The shapes of the input value should be 3 or 4 dimensional, but got ", value.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(keep_prob > 0 && keep_prob <= 1,
        "The keep_prob value must be in range of (0, 1], but got ", keep_prob, OPS_ERROR(ErrCode::PARAM));
    std::string input_layout_str = std::string(input_layout);
    std::string softmax_layout_str = std::string(softmax_layout);

    TORCH_CHECK(
        (softmax_layout_str == "TND" || softmax_layout_str == ""),
        "only supported softmax_layout=TND",
        OPS_ERROR(ErrCode::PARAM)
    );
    TORCH_CHECK(
        !(softmax_layout_str == "TND" && input_layout_str != "TND"),
        "softmax_layout=TND only supported when input_layout_str=TND",
        OPS_ERROR(ErrCode::PARAM)
    );

    if (input_layout_str == "TND") {
        TORCH_CHECK((sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode < static_cast<int64_t>(SparseMode::PREFIX)) ||
                    (sparse_mode > static_cast<int64_t>(SparseMode::PREFIX) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)),
                    "The sparse_mode value must be in range of [0,5) or (5,8], but got ",
                    sparse_mode, OPS_ERROR(ErrCode::PARAM));

        TORCH_CHECK(ac_seq_qlen.size() != 0 && ac_seq_kvlen.size() != 0 && ac_seq_qlen.size() == ac_seq_kvlen.size(),
                    "the size of actual_seq_qlen and actual_seq_kvlen must be the same and cannot be empty." +
                    OPS_ERROR(ErrCode::PARAM));
    } else {
        TORCH_CHECK(sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::PREFIX_COMPRESS),
                    "The sparse_mode value must be in range of [0,6], but got ",
                    sparse_mode, OPS_ERROR(ErrCode::PARAM));
    }
    for (auto &c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" ||
        input_layout_str == "BNSD" || input_layout_str == "BSND" || input_layout_str == "TND",
        "The input_layout should be BSH/SBH/BNSD/BSND/TND(case-insensitive), but got ",
        input_layout, OPS_ERROR(ErrCode::PARAM));

    int64_t B = 0;
    int64_t S0 = 0; // S for query
    int64_t S1 = 0; // S for key & value
    int64_t N_local = 0; // N for npu_fusion_attention
    int64_t D = 0;
    int64_t H = 0;
    int64_t T = 0;
    int64_t D2 = 0; // D2 for value head-dim
    c10::SmallVector<int64_t> atten_score_shape;

    if (input_layout_str == "BSH") {
        B = query.size(0);
        S0 = query.size(1);
        S1 = key.size(1);
        H = query.size(THIRD_ELEMENT);
        D = H / head_num;
        D2 = (D == 0 || !key.size(THIRD_ELEMENT)) ? 0 : value.size(THIRD_ELEMENT) / (key.size(THIRD_ELEMENT) / D);
        atten_score_shape = {B, S0, head_num * D2};
    } else if (input_layout_str == "SBH") {
        B = query.size(1);
        S0 = query.size(0);
        S1 = key.size(0);
        H = query.size(THIRD_ELEMENT);
        D = H / head_num;
        D2 = (D == 0 || !key.size(THIRD_ELEMENT)) ? 0 : value.size(THIRD_ELEMENT) / (key.size(THIRD_ELEMENT) / D);
        atten_score_shape = {S0, B, head_num * D2};
    } else if (input_layout_str == "BNSD") {
        B = query.size(0);
        N_local = query.size(1);
        S0 = query.size(THIRD_ELEMENT);
        S1 = key.size(THIRD_ELEMENT);
        D = query.size(FORTH_ELEMENT);
        D2 = value.size(FORTH_ELEMENT);
        atten_score_shape = {B, N_local, S0, D2};
    } else if (input_layout_str == "BSND") {
        B = query.size(0);
        N_local = query.size(THIRD_ELEMENT);
        S0 = query.size(1);
        S1 = key.size(1);
        D = query.size(FORTH_ELEMENT);
        D2 = value.size(FORTH_ELEMENT);
        atten_score_shape = {B, S0, N_local, D2};
    } else if (input_layout_str == "TND") {
        T = query.size(0);
        N_local = query.size(1);
        D = query.size(THIRD_ELEMENT);
        D2 = value.size(THIRD_ELEMENT);
        atten_score_shape = {T, N_local, D2};
    }

    double scale_value = scale;

    at::Tensor format_query = format_trans(query);
    at::Tensor attention_score = npu_preparation::apply_tensor_without_format(atten_score_shape, query.options());
    at::Tensor format_key = format_trans(key);
    at::Tensor format_value = format_trans(value);

    at::Tensor format_pse = format_trans(pse_const);
    at::Tensor format_padding_mask = format_trans(padding_mask_const);
    at::Tensor format_atten_mask = format_trans(atten_mask_const);
    at::Tensor format_sink = format_trans(sink_const);

    at::Tensor* seed;
    at::Tensor* offset;
    c10::SymInt numels = 0;

    if (input_layout_str == "TND") {
        numels = N_local;
        c10::SymInt accum = ac_seq_qlen[0] * ac_seq_kvlen[0];
        for (size_t i = 1; i < ac_seq_qlen.size(); i++) {
            accum += ((ac_seq_qlen[i] - ac_seq_qlen[i - 1]) * (ac_seq_kvlen[i] - ac_seq_kvlen[i - 1]));
        }
        numels *= accum;
    }
at::Tensor format_drop_mask = dropout_gen_mask(format_query, format_key, keep_prob, head_num, input_layout_str,
        gen_mask_parallel, sync, seed, offset, numels);
}
#endif
}