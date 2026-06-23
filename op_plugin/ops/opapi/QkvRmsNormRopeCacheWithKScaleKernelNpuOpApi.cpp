// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

#include <string>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t HEAD_NUMS_SIZE = 3;
constexpr const char* INPUT_LAYOUT_TND = "TND";
constexpr const char* INPUT_LAYOUT_NTD = "NTD";

struct ValidationParams {
    int64_t Nq;
    int64_t Nk;
    int64_t Nv;
    int64_t T;
    int64_t D;
    bool isTnd;
};

ValidationParams validate_inputs(
    const at::Tensor& qkv,
    const at::Tensor& cu_seq_len,
    const at::Tensor& actual_seq_lens,
    at::IntArrayRef head_nums,
    c10::string_view input_layout,
    const c10::optional<at::Tensor>& q_rotation,
    const c10::optional<at::Tensor>& k_rotation,
    const c10::optional<at::Tensor>& v_scale)
{
    TORCH_CHECK(c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
        "npu_qkv_rms_norm_rope_cache_with_k_scale is only supported on Ascend950." + OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(head_nums.size() == HEAD_NUMS_SIZE,
        "npu_qkv_rms_norm_rope_cache_with_k_scale: head_nums must have exactly 3 elements." + OPS_ERROR(ErrCode::PARAM));
    int64_t Nq = head_nums[0];
    int64_t Nk = head_nums[1];
    int64_t Nv = head_nums[2];
    TORCH_CHECK(Nq > 0 && Nk > 0 && Nv > 0,
        "npu_qkv_rms_norm_rope_cache_with_k_scale: all head_nums values must be positive." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(Nv == Nk,
        "npu_qkv_rms_norm_rope_cache_with_k_scale: Nv must equal Nk in current implementation, but got Nv=", Nv, ", Nk=", Nk, OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(qkv.dim() == DIM_THREE,
        "npu_qkv_rms_norm_rope_cache_with_k_scale: qkv must be 3D." + OPS_ERROR(ErrCode::PARAM));
    const std::string input_layout_str(input_layout);
    const bool is_tnd = input_layout_str == INPUT_LAYOUT_TND;
    const bool is_ntd = input_layout_str == INPUT_LAYOUT_NTD;
    TORCH_CHECK(is_tnd || is_ntd,
        "npu_qkv_rms_norm_rope_cache_with_k_scale: input_layout must be TND or NTD." + OPS_ERROR(ErrCode::PARAM));
    const int64_t head_dim = is_tnd ? DIM_ONE : DIM_ZERO;
    TORCH_CHECK(qkv.size(head_dim) == Nq + Nk + Nv,
        "npu_qkv_rms_norm_rope_cache_with_k_scale: qkv head dimension must equal Nq+Nk+Nv, but got ",
        qkv.size(head_dim), " vs expected ", Nq + Nk + Nv, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(cu_seq_len.dim() == DIM_ONE && cu_seq_len.size(DIM_ZERO) >= 2,
        "npu_qkv_rms_norm_rope_cache_with_k_scale: cu_seq_len must be 1D with length >= 2." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(actual_seq_lens.dim() == DIM_ONE && actual_seq_lens.size(DIM_ZERO) == cu_seq_len.size(DIM_ZERO) - 1,
        "npu_qkv_rms_norm_rope_cache_with_k_scale: actual_seq_lens must be 1D with length cu_seq_len.size(0)-1." + OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(q_rotation.has_value() && q_rotation.value().defined(),
        "npu_qkv_rms_norm_rope_cache_with_k_scale: q_rotation is required." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(k_rotation.has_value() && k_rotation.value().defined(),
        "npu_qkv_rms_norm_rope_cache_with_k_scale: k_rotation is required." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(v_scale.has_value() && v_scale.value().defined(),
        "npu_qkv_rms_norm_rope_cache_with_k_scale: v_scale is required." + OPS_ERROR(ErrCode::PARAM));

    int64_t T = is_tnd ? qkv.size(DIM_ZERO) : qkv.size(DIM_ONE);
    int64_t D = qkv.size(DIM_TWO);

    return {Nq, Nk, Nv, T, D, is_tnd};
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_qkv_rms_norm_rope_cache_with_k_scale(
    const at::Tensor& qkv,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    const at::Tensor& cos_sin,
    const at::Tensor& slot_mapping,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    at::Tensor& k_scale_cache,
    const at::Tensor& cu_seq_len,
    const at::Tensor& actual_seq_lens,
    at::IntArrayRef head_nums,
    c10::string_view input_layout,
    const c10::optional<at::Tensor>& q_rotation,
    const c10::optional<at::Tensor>& k_rotation,
    const c10::optional<at::Tensor>& v_scale,
    double epsilon)
{
    const auto& params = validate_inputs(qkv, cu_seq_len, actual_seq_lens, head_nums, input_layout, q_rotation, k_rotation, v_scale);
    int64_t Nq = params.Nq;
    int64_t T = params.T;
    int64_t D = params.D;
    bool is_tnd = params.isTnd;

    const at::Tensor& q_rotation_v = q_rotation.value();
    const at::Tensor& k_rotation_v = k_rotation.value();
    const at::Tensor& v_scale_v = v_scale.value();

    c10::SmallVector<int64_t, DIM_THREE> q_out_shape = is_tnd ?
        c10::SmallVector<int64_t, DIM_THREE>{T, Nq, D} :
        c10::SmallVector<int64_t, DIM_THREE>{Nq, T, D};
    c10::SmallVector<int64_t, DIM_TWO> q_scale_shape = is_tnd ?
        c10::SmallVector<int64_t, DIM_TWO>{T, Nq} :
        c10::SmallVector<int64_t, DIM_TWO>{Nq, T};
    at::Tensor q_out = npu_preparation::apply_tensor_without_format(q_out_shape, qkv.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor q_scale = npu_preparation::apply_tensor_without_format(q_scale_shape, qkv.options().dtype(at::kFloat));
    const std::string input_layout_str(input_layout);
    char* input_layout_ptr = const_cast<char*>(input_layout_str.c_str());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnQkvRmsNormRopeCacheWithKScale,
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping,
        k_cache, v_cache, k_scale_cache, cu_seq_len, actual_seq_lens,
        q_rotation_v, k_rotation_v, v_scale_v,
        head_nums, input_layout_ptr, epsilon,
        q_out, q_scale);

    return std::make_tuple(q_out, q_scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_qkv_rms_norm_rope_cache_with_k_scale_functional(
    const at::Tensor& qkv,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    const at::Tensor& cos_sin,
    const at::Tensor& slot_mapping,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& k_scale_cache,
    const at::Tensor& cu_seq_len,
    const at::Tensor& actual_seq_lens,
    at::IntArrayRef head_nums,
    c10::string_view input_layout,
    const c10::optional<at::Tensor>& q_rotation,
    const c10::optional<at::Tensor>& k_rotation,
    const c10::optional<at::Tensor>& v_scale,
    double epsilon)
{
    const auto& params = validate_inputs(qkv, cu_seq_len, actual_seq_lens, head_nums, input_layout, q_rotation, k_rotation, v_scale);
    int64_t Nq = params.Nq;
    int64_t T = params.T;
    int64_t D = params.D;
    bool is_tnd = params.isTnd;

    const at::Tensor& q_rotation_v = q_rotation.value();
    const at::Tensor& k_rotation_v = k_rotation.value();
    const at::Tensor& v_scale_v = v_scale.value();

    at::Tensor k_cache_clone = k_cache.clone();
    at::Tensor v_cache_clone = v_cache.clone();
    at::Tensor k_scale_cache_clone = k_scale_cache.clone();

    c10::SmallVector<int64_t, DIM_THREE> q_out_shape = is_tnd ?
        c10::SmallVector<int64_t, DIM_THREE>{T, Nq, D} :
        c10::SmallVector<int64_t, DIM_THREE>{Nq, T, D};
    c10::SmallVector<int64_t, DIM_TWO> q_scale_shape = is_tnd ?
        c10::SmallVector<int64_t, DIM_TWO>{T, Nq} :
        c10::SmallVector<int64_t, DIM_TWO>{Nq, T};
    at::Tensor q_out = npu_preparation::apply_tensor_without_format(q_out_shape, qkv.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor q_scale = npu_preparation::apply_tensor_without_format(q_scale_shape, qkv.options().dtype(at::kFloat));
    const std::string input_layout_str(input_layout);
    char* input_layout_ptr = const_cast<char*>(input_layout_str.c_str());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnQkvRmsNormRopeCacheWithKScale,
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping,
        k_cache_clone, v_cache_clone, k_scale_cache_clone, cu_seq_len, actual_seq_lens,
        q_rotation_v, k_rotation_v, v_scale_v,
        head_nums, input_layout_ptr, epsilon,
        q_out, q_scale);

    return std::make_tuple(q_out, q_scale, k_cache_clone, v_cache_clone, k_scale_cache_clone);
}

} // namespace op_api
