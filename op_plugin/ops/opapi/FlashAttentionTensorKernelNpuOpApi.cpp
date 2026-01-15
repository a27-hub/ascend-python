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
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

#include "torch_npu/csrc/logging/LogContext.h"
std::shared_ptr<npu_logging::Logger> logger = npu_logging::logging().getLogger("torch_npu.memory");

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
namespace {
DropOutStatus get_dropout_status(double keep_prob)
{
    if (std::abs(keep_prob - 0.0) < EPSILON) {
        return DropOutStatus::DROPOUT_ALL;
    }
    if (std::abs(keep_prob - 1.0) < EPSILON) {
        return DropOutStatus::DROPOUT_NONE;
    }
    return DropOutStatus::DROPOUT_NORMAL;
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

at::Tensor& stateless_dropout_gen_mask_aclop(const at::Tensor &query, double keep_prob, int64_t seed,
    const int64_t offset, const int64_t numels, at::Tensor& mask)
{
    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    c10::TensorOptions options = query.options();
    at::SmallVector<int64_t, ::N> offsetList = {0, offset};
    const int64_t seed1 = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessDropOutGenMask")
        .Input(at::IntArrayRef{numels})
        .Input(at::Scalar(keep_prob), query.scalar_type(),  at_npu::native::CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(at::Scalar(seed), at::ScalarType::Int)
        .Input(at::Scalar(seed1), at::ScalarType::Int)
        .Input(offsetList, at::kLong,  at_npu::native::CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Output(mask)
        .Run();
    return mask;
}

int64_t extract_value(const at::Tensor& cpu_tensor)
{
    TORCH_CHECK(cpu_tensor.numel() == 1,
                "Tensor must contain exactly one element",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(cpu_tensor.is_cpu(),
                "Tensor must be on CPU to call .item()",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(cpu_tensor.dtype() == at::kLong ||
                cpu_tensor.dtype() == at::kInt,
                "Expected integer dtype, got ", cpu_tensor.dtype(),
                OPS_ERROR(ErrCode::PARAM));

    return cpu_tensor.item().to<int64_t>();
}

c10_npu::CaptureStatus stream_capture_status_may_init_ctx(const c10_npu::NPUStream& stream)
{
    if (!c10_npu::acl::IsCaptureSupported() || !c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        return c10_npu::CaptureStatus::None;
    }

    aclmdlRICaptureStatus is_capturing{ACL_MODEL_RI_CAPTURE_STATUS_NONE};
    aclmdlRI model_ri;
    NPU_CHECK_ERROR(
        c10_npu::acl::AclmdlRICaptureGetInfo(stream.stream(false), &is_capturing, &model_ri));
    return c10_npu::CaptureStatus(is_capturing);
}

at::Tensor dropout_gen_mask_tensor_impl(const at::Tensor &query, double keep_prob, const at::Tensor &seed,
    const at::Tensor &offset, const int64_t numels, c10_npu::CaptureStatus is_capture, const uint64_t offset_intragraph)
{
    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    c10::TensorOptions options = query.options();
    at::Tensor mask = OpPreparation::apply_tensor_without_format(at::IntArrayRef{length}, options.dtype(at::kByte));
    c10::SmallVector<int64_t, SIZE> shapeSize = {numels};
    at::IntArrayRef shapeArray = at::IntArrayRef(shapeSize);
    double prob;
    at::Scalar probScalar = at::Scalar(float(1.0) - float(keep_prob));
    prob = probScalar.toDouble();
    aclDataType probDataType = at_npu::native::OpPreparation::convert_to_acl_data_type(query.scalar_type());
    if (is_capture == c10_npu::CaptureStatus::None) {
        int64_t seedValue = extract_value(seed);
        int64_t offsetValue = extract_value(offset);
        DO_COMPATIBILITY(aclnnDropoutGenMaskV2,
            stateless_dropout_gen_mask_aclop(query, keep_prob, seedValue, offsetValue, numels, mask));
        EXEC_NPU_CMD(aclnnDropoutGenMaskV2, shapeArray, prob, seedValue, offsetValue, probDataType, mask);
    } else {
        EXEC_NPU_CMD(aclnnDropoutGenMaskV2Tensor, shapeArray, prob, seed, offset, offset_intragraph, probDataType, mask);
    }
    return mask;
}

at::Tensor dropout_gen_mask_tensor_dispatch(const at::Tensor &query, double keep_prob, const at::Tensor &seed,
    const at::Tensor &offset, const int64_t numels, const bool gen_mask_parallel, const bool sync,
    c10_npu::CaptureStatus is_capture, const uint64_t offset_intragraph = 0)
{
    at::Tensor mask;

    logger->info("npu_flash_attention_symint dropout_gen_mask_tensor_dispatch start, is_capture is %d.", (int)is_capture);
    if (gen_mask_parallel) {
        auto original_stream = c10_npu::getCurrentNPUStream();
        auto secondary_stream = c10_npu::getCurrentSecondaryStream();
        logger->info("npu_flash_attention_symint dropout_gen_mask_tensor_dispatch true, original_stream id is %lld, secondary_stream id is %lld.",
            static_cast<int64_t>(original_stream.id()), static_cast<int64_t>(secondary_stream.id()));
        if (is_capture == c10_npu::CaptureStatus::None) {
            // During the life cycle of this raii instance, the calcu stream is set as the
            // secondary stream, and tasks are distributed to the secondary stream. At the
            // same time, according to the one-stream-one-pool principle, memory is also
            // alloced from the pool of the secondary stream.
            c10_npu::SecondaryStreamGuard guard(secondary_stream);
            mask = dropout_gen_mask_tensor_impl(query, keep_prob, seed, offset, numels, is_capture, offset_intragraph);
            if (sync) {
                OPS_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(original_stream));
            }
        } else {
            const auto gen = at_npu::detail::getDefaultNPUGenerator();
            auto gen_state_ = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_npu_state(0);

            c10_npu::CaptureStatus secondary_stream_capture_state = stream_capture_status_may_init_ctx(secondary_stream);
            logger->info("npu_flash_attention_symint dropout_gen_mask_tensor_dispatch, original_stream id is %lld, secondary_stream_capture_state is %d, secondary_stream id is %lld.",
                static_cast<int64_t>(original_stream.id()), (int)secondary_stream_capture_state, static_cast<int64_t>(secondary_stream.id()));
            const auto gen = at_npu::detail::getDefaultNPUGenerator();
            if (!gen_state_.secondary_stream_capture_state_) {
                logger->info("npu_flash_attention_symint dropout_gen_mask_tensor_dispatch, second stream record!!");
                c10_npu::NPUEvent capture_event_begin = c10_npu::NPUEvent();
                capture_event_begin.record(original_stream);
                capture_event_begin.block(secondary_stream);
                ASCEND_LOGI("Event: record and block in dropout op capture begin is successfully executed, event=%p", capture_event_begin.event());
            }
            // During the life cycle of this raii instance, the calcu stream is set as the
            // secondary stream, and tasks are distributed to the secondary stream. At the
            // same time, according to the one-stream-one-pool principle, memory is also
            // alloced from the pool of the secondary stream.
            c10_npu::SecondaryStreamGuard guard(secondary_stream);
            mask = dropout_gen_mask_tensor_impl(query, keep_prob, seed, offset, numels, is_capture, offset_intragraph);
            if (!gen_state_.secondary_stream_capture_state_) {
                ASCEND_LOGI("Event: record and block in dropout op capture end is successfully executed");
                at::check_generator<at_npu::NPUGeneratorImpl>(gen)->set_secondary_stream_capture_state(true);
            }
            if (sync) {
                OPS_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(original_stream));
            }
        }
    } else {
        auto original_stream = c10_npu::getCurrentNPUStream();
        logger->info("npu_flash_attention_symint dropout_gen_mask_tensor_dispatch false, original_stream id is %lld, is_capture is %d.",
            static_cast<int64_t>(original_stream.id()), (int)is_capture);
        mask = dropout_gen_mask_tensor_impl(query, keep_prob, seed, offset, numels, is_capture, offset_intragraph);
    }
    return mask;
}
}

at::Tensor dropout_gen_mask_tensor(const at::Tensor &query, const at::Tensor &key, double keep_prob, int64_t head_num,
    std::string input_layout, bool gen_mask_parallel, bool sync, at::Tensor& seed, at::Tensor& offset,
    int64_t& numels, c10_npu::CaptureStatus is_capture)
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
    if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        logger->info("npu_flash_attention_symint dropout_gen_mask_tensor DROPOUT_NORMAL, numels is %lld, is_capture is %d.",
            numels, (int)is_capture);
        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        uint64_t offset_intragraph = 0;
        if (is_capture == c10_npu::CaptureStatus::None) {
            auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
            auto options = at::TensorOptions().device(at::kCPU).dtype(at::kLong);
            seed = at::empty({1}, options).fill_(static_cast<int64_t>(pair.first));
            offset = at::empty({1}, options).fill_(static_cast<int64_t>(pair.second));
        } else {
            auto gen_state_ = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_npu_state(10);
            seed = gen_state_.seed_.ptr->clone();
            offset = gen_state_.offset_.ptr->clone();
            offset.add_(gen_state_.offset_intragraph_);
        }
        if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
            drop_mask = dropout_gen_mask_tensor_dispatch(query, keep_prob, seed, offset, numels,
                gen_mask_parallel, sync, is_capture, offset_intragraph);
        }
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        logger->info("npu_flash_attention_symint dropout_gen_mask_tensor DROPOUT_ALL, is_capture is %d.", (int)is_capture);
        drop_mask = at::zeros({length}, query.options().dtype(at::kByte));
    } else {
        logger->info("npu_flash_attention_symint dropout_gen_mask_tensor other, numels is %lld, is_capture is %d.",
            numels, (int)is_capture);
    }

    return drop_mask;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_flash_attention_backward(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    int64_t head_num,
    const std::string input_layout,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &drop_mask,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise,
    const c10::optional<at::Tensor> &seed,
    const c10::optional<at::Tensor> &offset,
    at::IntArrayRef prefixN,
    at::IntArrayRef ac_seq_qlen,
    at::IntArrayRef ac_seq_kvlen,
    int64_t sparse_mode,
    c10::string_view softmax_layout,
    const c10::optional<at::Tensor> &sink)
{
    double scale = scale_value;

    const at::Tensor &pse_const = pse.value_or(at::Tensor());
    const at::Tensor &drop_mask_const = drop_mask.value_or(at::Tensor());
    const at::Tensor &padding_mask_const = padding_mask.value_or(at::Tensor());
    const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
    const at::Tensor &softmax_max_const = softmax_max.value_or(at::Tensor());
    const at::Tensor &softmax_sum_const = softmax_sum.value_or(at::Tensor());
    const at::Tensor &softmax_const = softmax_in.value_or(at::Tensor());
    const at::Tensor &attention_const = attention_in.value_or(at::Tensor());
    const at::Tensor &sink_const = sink.value_or(at::Tensor());

    at::Tensor format_query = format_trans(query);
    at::Tensor format_key = format_trans(key);
    at::Tensor format_value = format_trans(value);
    at::Tensor format_dy = format_trans(dy);

    at::Tensor format_pse = format_trans(pse_const);
    at::Tensor format_drop_mask = format_trans(drop_mask_const);
    at::Tensor format_padding_mask = format_trans(padding_mask_const);
    at::Tensor format_atten_mask = format_trans(atten_mask_const);
    at::Tensor format_softmax_max = format_trans(softmax_max_const);
    at::Tensor format_softmax_sum = format_trans(softmax_sum_const);
    at::Tensor format_softmax = format_trans(softmax_const);
    at::Tensor format_attention = format_trans(attention_const);
    at::Tensor format_sink = format_trans(sink_const);
    at::Tensor dq = OpPreparation::apply_tensor_without_format(format_query);
    at::Tensor dk = OpPreparation::apply_tensor_without_format(format_key);
    at::Tensor dv = OpPreparation::apply_tensor_without_format(format_value);
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
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
        if (format_sink.defined()) { // sink is defined
            auto format_query_rope = at::Tensor();
            auto format_key_rope = at::Tensor();
            auto q_start_idx_val = at::IntArrayRef{};
            auto kv_start_idx_val = at::IntArrayRef{};
            int64_t pse_type = 1;
            at::Tensor dq_rope = at::empty({0}, query.options());
            at::Tensor dk_rope = at::empty({0}, key.options());

            if (!ac_seq_qlen.empty() && !ac_seq_kvlen.empty()) { // TND
                TORCH_CHECK(
                    check_aclnn_kernel_available("aclnnFlashAttentionUnpaddingScoreGradV5"),
                    "The param sink is not supported in this CANN version, aclnnFlashAttentionUnpaddingScoreGradV5 is not available",
                    OPS_ERROR(ErrCode::PARAM)
                );
                std::string softmax_layout_str = std::string(softmax_layout);
                softmax_layout_str = (softmax_layout_str == "TND") ? "same_as_input" : softmax_layout_str;
                char softmax_layout_char[LAYOUT_MAX_LENGTH];
                strncpy(softmax_layout_char, softmax_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
                softmax_layout_char[LAYOUT_MAX_LENGTH - 1] = '\0';
                EXEC_NPU_CMD(
                    aclnnFlashAttentionUnpaddingScoreGradV5, format_query, format_query_rope, format_key, format_key_rope, format_value, format_dy,
                    format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
                    format_softmax_sum, format_softmax, format_attention, format_sink, prefixN, ac_seq_qlen, ac_seq_kvlen, q_start_idx_val, kv_start_idx_val, // + sink
                    scale_value, keep_prob, pre_tockens, next_tockens, head_num, input_layout_char, inner_precise, sparse_mode, pse_type, softmax_layout_char, // +softmax_layout
                    dq, dq_rope, dk, dk_rope, dv, dpse, dsink); // +dsink
            } else {
                TORCH_CHECK(
                    check_aclnn_kernel_available("aclnnFlashAttentionScoreGradV3"),
                    "The param sink is not supported in this CANN version, aclnnFlashAttentionScoreGradV3 is not available",
                    OPS_ERROR(ErrCode::PARAM)
                );
                EXEC_NPU_CMD(
                    aclnnFlashAttentionScoreGradV3, format_query, format_key, format_value, format_dy,
                    format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
                    format_softmax_sum, format_softmax, format_attention, format_sink, prefixN, q_start_idx_val, kv_start_idx_val, scale_value, keep_prob, // + sink
                    pre_tockens, next_tockens, head_num, input_layout_char, inner_precise, sparse_mode, pse_type, dq, dk, dv, dpse, dsink); // +dsink
                }
        } else { // sink is undefined
            if (!ac_seq_qlen.empty() && !ac_seq_kvlen.empty()) {
                std::string softmax_layout_str = std::string(softmax_layout);
                static const bool is_fa_grad_V4_available =
                    check_aclnn_kernel_available("aclnnFlashAttentionUnpaddingScoreGradV4");
                if (softmax_layout_str == "TND" && is_fa_grad_V4_available) {
                    softmax_layout_str = "same_as_input";
                    char softmax_layout_char[LAYOUT_MAX_LENGTH];
                    strncpy(softmax_layout_char, softmax_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
                    EXEC_NPU_CMD(
                        aclnnFlashAttentionUnpaddingScoreGradV4, format_query, format_key, format_value, format_dy,
                        format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
                        format_softmax_sum, format_softmax, format_attention, prefixN, ac_seq_qlen, ac_seq_kvlen,
                        scale_value, keep_prob, pre_tockens, next_tockens, head_num, input_layout_char,
                        inner_precise, sparse_mode, dq, dk, dv, dpse, softmax_layout_char);
                } else {
                    TORCH_CHECK(softmax_layout_str == "", "The param softmax_layout is not supported",
                        OPS_ERROR(ErrCode::PARAM));
                    EXEC_NPU_CMD(
                        aclnnFlashAttentionUnpaddingScoreGrad, format_query, format_key, format_value, format_dy,
                        format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
                        format_softmax_sum, format_softmax, format_attention, prefixN, ac_seq_qlen, ac_seq_kvlen,
                        scale_value, keep_prob, pre_tockens, next_tockens, head_num, input_layout_char,
                        inner_precise, sparse_mode, dq, dk, dv, dpse);
                }
            } else {
                EXEC_NPU_CMD(
                    aclnnFlashAttentionScoreGrad, format_query, format_key, format_value, format_dy,
                    format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
                    format_softmax_sum, format_softmax, format_attention, prefixN, scale_value, keep_prob,
                    pre_tockens, next_tockens, head_num, input_layout_char,
                    inner_precise, sparse_mode, dq, dk, dv, dpse);
            }
        }
    } else { // Ascend910_95
        // TO BE DONE
    }
    FLOP_COUNT(FlopCounter::flash_attention_backward_flop, query, key, value,
        dy, head_num, input_layout, ac_seq_qlen, ac_seq_kvlen);

    if (!format_pse.defined()) {
        at::Tensor dpse_required;
        dpse = dpse_required;
    }

    return std::make_tuple(dq, dk, dv, dpse, dsink);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_flash_attention_grad_symint(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    int64_t head_num,
    c10::string_view input_layout,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise,
    const c10::optional<at::Tensor> &seed,
    const c10::optional<at::Tensor> &offset,
    c10::OptionalArrayRef<c10::SymInt> prefix,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    int64_t sparse_mode,
    bool gen_mask_parallel,
    bool sync,
    c10::string_view softmax_layout,
    const c10::optional<at::Tensor> &sink)
{
    TORCH_CHECK(c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95, "Ascend910_95 not support", OPS_ERROR(ErrCode::NOT_SUPPORT));

    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ",
        key.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "The shapes of the input value should be 3 or 4 dimensional, but got ",
        value.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dy.dim() == DIMENSION_3D || dy.dim() == DIMENSION_4D,
        "The shapes of the input dy should be 3 or 4 dimensional, but got ", dy.dim(), "-dimensional",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(keep_prob > 0 && keep_prob <= 1,
        "The keep_prob value must be in range of (0, 1], but got ", keep_prob,
        OPS_ERROR(ErrCode::PARAM));
    std::string input_layout_str = std::string(input_layout);
    if (input_layout_str == "TND") {
        TORCH_CHECK((sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode < static_cast<int64_t>(SparseMode::PREFIX)) ||
                    (sparse_mode > static_cast<int64_t>(SparseMode::PREFIX) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)),
                    "The sparse_mode value must be in range of [0,5) or (5,8], but got ",
                    sparse_mode, OPS_ERROR(ErrCode::PARAM));
    } else {
        TORCH_CHECK(sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::PREFIX_COMPRESS),
                    "The sparse_mode value must be in range of [0,6], but got ",
                    sparse_mode, OPS_ERROR(ErrCode::PARAM));
    }
    for (auto &c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" || input_layout_str == "BNSD" ||
        input_layout_str == "BSND" || input_layout_str == "TND",
        "The input_layout should be BSH/SBH/BNSD/BSND/TND(case-insensitive), but got ",
        input_layout, OPS_ERROR(ErrCode::PARAM));

    auto prefixN = prefix.has_value() ? c10::asIntArrayRefUnchecked(prefix.value()) : at::IntArrayRef{};
    auto ac_seq_qlen = actual_seq_qlen.has_value() ? c10::asIntArrayRefUnchecked(actual_seq_qlen.value()) : at::IntArrayRef{};
    auto ac_seq_kvlen = actual_seq_kvlen.has_value() ? c10::asIntArrayRefUnchecked(actual_seq_kvlen.value()) : at::IntArrayRef{};
    int64_t N_local = 0; // N for npu_fusion_attention

    if (input_layout_str == "BNSD") {
        N_local = query.size(1);
    } else if (input_layout_str == "BSND") {
        N_local = query.size(THIRD_ELEMENT);
    } else if (input_layout_str == "TND") {
        N_local = query.size(1);
    }
    int64_t numels = 0;
    if (input_layout_str == "TND") {
        numels = N_local;
        int64_t accum = ac_seq_qlen[0] * ac_seq_kvlen[0];
        for (size_t i = 1; i < ac_seq_qlen.size(); i++) {
            accum += ((ac_seq_qlen[i] - ac_seq_qlen[i - 1]) * (ac_seq_kvlen[i] - ac_seq_kvlen[i - 1]));
        }
        numels *= accum;
    } else if (input_layout == "BSH") {
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
    at::Tensor drop_mask;
    c10_npu::CaptureStatus is_capture = c10_npu::currentStreamCaptureStatusMayInitCtx();
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95 && get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        const at::Tensor &seed_const = seed.value_or(at::Tensor());
        const at::Tensor &offset_const = offset.value_or(at::Tensor());
        drop_mask = dropout_gen_mask_tensor_dispatch(query, keep_prob, seed_const, offset_const, numels, gen_mask_parallel, sync, is_capture);
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    }
    auto result = npu_flash_attention_backward(query,
        key, value, dy, head_num, input_layout_str, pse, drop_mask, padding_mask, atten_mask,
        softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob, pre_tockens,
        next_tockens, inner_precise, seed, offset, prefixN, ac_seq_qlen, ac_seq_kvlen, sparse_mode,
        softmax_layout, sink);
    if (!sync && is_capture == c10_npu::CaptureStatus::None) {
        c10_npu::NPUEvent npu_event;
        npu_event.record(c10_npu::getCurrentNPUStream());
        npu_event.block(c10_npu::getCurrentSecondaryStream());
    }

    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_flash_attention_symint(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, c10::string_view input_layout,
    const c10::optional<at::Tensor> &pse, const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise,
    c10::OptionalArrayRef<c10::SymInt> prefix, c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync,
    c10::string_view softmax_layout, const c10::optional<at::Tensor> &sink)
{
    TORCH_CHECK(c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95, "Ascend910_95 not support", OPS_ERROR(ErrCode::NOT_SUPPORT));

    std::string input_layout_str1(input_layout);
    std::string softmax_layout_str1(softmax_layout);
    logger->info("npu_flash_attention_symint start, head_num is %lld, input_layout is %s, scale is %f, keep_prob is %f, pre_tockens is %lld.",
        head_num, input_layout_str1.c_str(), scale, keep_prob, pre_tockens);
    logger->info("npu_flash_attention_symint start, next_tockens is %lld, inner_precise is %lld, sparse_mode is %lld, gen_mask_parallel is %d, sync is %d, softmax_layout is %s.",
        next_tockens, inner_precise, sparse_mode, gen_mask_parallel, sync, softmax_layout_str1.c_str());
    const at::Tensor &pse_const = pse.value_or(at::Tensor());
    const at::Tensor &padding_mask_const = padding_mask.value_or(at::Tensor());
    const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
    const at::Tensor &sink_const = sink.value_or(at::Tensor());
    auto prefixN = prefix.has_value() ? c10::asIntArrayRefUnchecked(prefix.value()) : at::IntArrayRef{};
    auto ac_seq_qlen = actual_seq_qlen.has_value() ? c10::asIntArrayRefUnchecked(actual_seq_qlen.value()) : at::IntArrayRef{};
    auto ac_seq_kvlen = actual_seq_kvlen.has_value() ? c10::asIntArrayRefUnchecked(actual_seq_kvlen.value()) : at::IntArrayRef{};

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

    at::Tensor seed ;
    at::Tensor offset;
    int64_t numels = 0;

    if (input_layout_str == "TND") {
        numels = N_local;
        int64_t accum = ac_seq_qlen[0] * ac_seq_kvlen[0];
        for (size_t i = 1; i < ac_seq_qlen.size(); i++) {
            accum += ((ac_seq_qlen[i] - ac_seq_qlen[i - 1]) * (ac_seq_kvlen[i] - ac_seq_kvlen[i - 1]));
        }
        numels *= accum;
    }
    c10_npu::CaptureStatus is_capture = c10_npu::currentStreamCaptureStatusMayInitCtx();
    logger->info("npu_flash_attention_symint dropout_gen_mask_tensor before, is_capture is %d.", (int)is_capture);
    at::Tensor format_drop_mask = dropout_gen_mask_tensor(format_query, format_key, keep_prob, head_num, input_layout_str,
        gen_mask_parallel, sync, seed, offset, numels, is_capture);

    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    at::Tensor softmax_out;

    if (input_layout_str != "TND") {
        softmax_max = OpPreparation::apply_tensor_without_format({B, head_num, S0, SOFTMAXMAX_LAST_DIMSHAPE},
            query.options().dtype(at::kFloat)); // [B, N, S0, 8]
        softmax_sum = OpPreparation::apply_tensor_without_format({B, head_num, S0, SOFTMAXMAX_LAST_DIMSHAPE},
            query.options().dtype(at::kFloat)); // [B, N, S0, 8]
    } else {
        softmax_max = OpPreparation::apply_tensor_without_format({T, N_local, SOFTMAXMAX_LAST_DIMSHAPE},
            query.options().dtype(at::kFloat)); // [T, N, 8]
        softmax_sum = OpPreparation::apply_tensor_without_format({T, N_local, SOFTMAXMAX_LAST_DIMSHAPE},
            query.options().dtype(at::kFloat)); // [T, N, 8]
    }

    softmax_out = at::empty({0}, query.options());
    char input_layout_char[LAYOUT_MAX_LENGTH];
    strncpy(input_layout_char, input_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
    static const bool is_fa_V4_available = check_aclnn_kernel_available("aclnnFlashAttentionVarLenScoreV4");
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
        if (format_sink.defined()) { // sink is defined
            logger->info("npu_flash_attention_symint aclnnFlashAttentionVarLenScoreV5, is_capture is %d.", (int)is_capture);
            auto format_query_rope = at::Tensor();
            auto format_key_rope = at::Tensor();
            auto q_start_idx_val = at::IntArrayRef{};
            auto kv_start_idx_val = at::IntArrayRef{};
            int64_t pse_type = 1;

            if (!ac_seq_qlen.empty() && !ac_seq_kvlen.empty()) { // TND
                TORCH_CHECK(
                    check_aclnn_kernel_available("aclnnFlashAttentionVarLenScoreV5"),
                    "The param sink is not supported in this CANN version, aclnnFlashAttentionVarLenScoreV5 is not available",
                    OPS_ERROR(ErrCode::PARAM)
                );
                softmax_layout_str = (softmax_layout_str == "TND") ? "same_as_input" : softmax_layout_str;
                char softmax_layout_char[LAYOUT_MAX_LENGTH];
                strncpy(softmax_layout_char, softmax_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
                softmax_layout_char[LAYOUT_MAX_LENGTH - 1] = '\0';
                EXEC_NPU_CMD(
                    aclnnFlashAttentionVarLenScoreV5, format_query, format_query_rope, format_key, format_key_rope, format_value,
                    format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_sink, prefixN, // +sink
                    ac_seq_qlen, ac_seq_kvlen, q_start_idx_val, kv_start_idx_val, scale, keep_prob, pre_tockens, next_tockens, head_num,
                    input_layout_char, inner_precise, sparse_mode, pse_type, softmax_layout_char, softmax_max, softmax_sum, // +softmax_layout
                    softmax_out, attention_score);
            } else {
                TORCH_CHECK(
                    check_aclnn_kernel_available("aclnnFlashAttentionScoreV3"),
                    "The param sink is not supported in this CANN version, aclnnFlashAttentionScoreV3 is not available",
                    OPS_ERROR(ErrCode::PARAM)
                );
                EXEC_NPU_CMD(
                    aclnnFlashAttentionScoreV3, format_query, format_key, format_value,
                    format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_sink, prefixN, q_start_idx_val, kv_start_idx_val, // +sink
                    scale, keep_prob, pre_tockens, next_tockens, head_num, input_layout_char, inner_precise,
                    sparse_mode, pse_type, softmax_max, softmax_sum, softmax_out, attention_score);
            }
        } else { // sink is undefined
            if (!ac_seq_qlen.empty() && !ac_seq_kvlen.empty()) {
                logger->info("npu_flash_attention_symint aclnnFlashAttentionVarLenScoreV4, is_capture is %d.", (int)is_capture);
                if (softmax_layout_str == "TND" && is_fa_V4_available) {
                    softmax_layout_str = "same_as_input";
                    char softmax_layout_char[LAYOUT_MAX_LENGTH];
                    strncpy(softmax_layout_char, softmax_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
                    EXEC_NPU_CMD(
                        aclnnFlashAttentionVarLenScoreV4, format_query, format_key, format_value,
                        format_pse, format_drop_mask, format_padding_mask, format_atten_mask, prefixN,
                        ac_seq_qlen, ac_seq_kvlen, scale, keep_prob, pre_tockens, next_tockens, head_num,
                        input_layout_char, inner_precise, sparse_mode, softmax_layout_char, softmax_max, softmax_sum,
                        softmax_out, attention_score);
                } else {
                    TORCH_CHECK(softmax_layout_str == "", "The param softmax_layout is not supported",
                        OPS_ERROR(ErrCode::PARAM));
                    EXEC_NPU_CMD(
                        aclnnFlashAttentionVarLenScore, format_query, format_key, format_value,
                        format_pse, format_drop_mask, format_padding_mask, format_atten_mask, prefixN,
                        ac_seq_qlen, ac_seq_kvlen, scale, keep_prob, pre_tockens, next_tockens, head_num,
                        input_layout_char, inner_precise, sparse_mode, softmax_max, softmax_sum,
                        softmax_out, attention_score);
                }
            } else {
                logger->info("npu_flash_attention_symint aclnnFlashAttentionScore, is_capture is %d.", (int)is_capture);
                EXEC_NPU_CMD(
                    aclnnFlashAttentionScore, format_query, format_key, format_value,
                    format_pse, format_drop_mask, format_padding_mask, format_atten_mask, prefixN,
                    scale, keep_prob, pre_tockens, next_tockens, head_num, input_layout_char,
                    inner_precise, sparse_mode, softmax_max, softmax_sum, softmax_out, attention_score);
            }
        }
    } else { // Ascend910_95
        // TO BE DONE, not support
    }
    FLOP_COUNT(FlopCounter::flash_attention_forward_flop, query, key, value, head_num,
               input_layout_str, ac_seq_qlen, ac_seq_kvlen);

    if (!sync && is_capture == c10_npu::CaptureStatus::None) {
        logger->info("npu_flash_attention_symint !sync, is_capture is %d.", (int)is_capture);
        c10_npu::NPUEvent npu_event;
        npu_event.record(c10_npu::getCurrentNPUStream());
        npu_event.block(c10_npu::getCurrentSecondaryStream());
    }
    
    auto original_stream = c10_npu::getCurrentNPUStream();
    auto secondary_stream = c10_npu::getCurrentSecondaryStream();
    logger->info("npu_flash_attention_symint end, original_stream id is %lld, secondary_stream id is %lld.",
        static_cast<int64_t>(original_stream.id()), static_cast<int64_t>(secondary_stream.id()));

    return std::make_tuple(attention_score, softmax_max, softmax_sum, softmax_out,
        seed, offset);
}
#endif
}