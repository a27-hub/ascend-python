// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/npu_log.h"
#include <c10/util/Logging.h>

#include <ATen/Operators.h>
#include <ATen/NativeFunctions.h>
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_v3(
    const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
    const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
    at::Tensor& kv_cache, at::Tensor& kr_cache, const c10::optional<at::Tensor>& cache_index,
    const c10::optional<at::Tensor>& dequant_scale_x, const c10::optional<at::Tensor>& dequant_scale_w_dq,
    const c10::optional<at::Tensor>& dequant_scale_w_uq_qr, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr, const c10::optional<at::Tensor>& quant_scale_ckv,
    const c10::optional<at::Tensor>& quant_scale_ckr, const c10::optional<at::Tensor>& smooth_scales_cq,
    const c10::optional<at::Tensor>& actual_seq_len, const c10::optional<at::Tensor>& k_nope_clip_alpha,
    double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
    int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale)
{
    // construct the output tensor
    auto token_x_dim = token_x.dim();
    TORCH_CHECK(token_x_dim == 2 || token_x_dim == 3, "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim, OPS_ERROR(ErrCode::PARAM));

    auto weight_uk_dim = weight_uk.dim();
    TORCH_CHECK(weight_uk_dim == 3, "weight_uk dim num should be 3, but the actual value is ", weight_uk_dim, OPS_ERROR(ErrCode::PARAM));

    auto rope_sin_dim = rope_sin.dim();
    TORCH_CHECK(rope_sin_dim == 2 || rope_sin_dim == 3, "rope_sin dim num should be 2 or 3, but the actual value is ", rope_sin_dim, OPS_ERROR(ErrCode::PARAM));

    at::Tensor query;
    at::Tensor query_rope;
    at::Tensor dequant_scale_q_nope;
    at::Tensor query_norm;
    at::Tensor dequant_scale_q_norm;

    if (token_x_dim == 3) {
        if (weight_quant_mode == 2 && kv_cache_quant_mode == 1) {
            // kvcache量化
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({token_x.size(0) * token_x.size(1), weight_uk.size(0), 1}, at::kFloat);
        } else {
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(rope_sin.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
        }
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), rope_sin.size(2)}, at::kBFloat16);
        if (query_norm_flag) {
            query_norm =  npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_dq.size(1)}, token_x.options().dtype(weight_uq_qr.dtype()));
            if (weight_uq_qr.dtype() == at::kChar) {
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({token_x.size(0) * token_x.size(1), 1}, at::kFloat);
            } else {
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
            }
        } else {
            query_norm = npu_preparation::apply_tensor_without_format({1}, token_x.options().dtype(weight_uq_qr.dtype()));
            dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
        }
    } else {
        if (weight_quant_mode == 2 && kv_cache_quant_mode == 1) {
            // kvcache量化
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), 1}, at::kFloat);
        } else {
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(rope_sin.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
        }
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), rope_sin.size(1)}, at::kBFloat16);
        if (query_norm_flag) {
            query_norm =  npu_preparation::apply_tensor_without_format({token_x.size(0), weight_dq.size(1)}, token_x.options().dtype(weight_uq_qr.dtype()));
            if (weight_uq_qr.dtype() == at::kChar) {
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({token_x.size(0), 1}, at::kFloat);
            } else {
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
            }
        } else {
            query_norm = npu_preparation::apply_tensor_without_format({1}, token_x.options().dtype(weight_uq_qr.dtype()));
            dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
        }
    }

    char *cache_mode_ptr = const_cast<char *>(cache_mode.data());

    EXEC_NPU_CMD(aclnnMlaPrologV3WeightNz, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, dequant_scale_x, dequant_scale_w_dq,
        dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len,
        k_nope_clip_alpha, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode_ptr, query_norm_flag, weight_quant_mode, kv_cache_quant_mode, query_quant_mode,
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query, query_rope,
        dequant_scale_q_nope, query_norm, dequant_scale_q_norm);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_v3_functional(
    const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
    const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
    const at::Tensor& kv_cache, const at::Tensor& kr_cache, const c10::optional<at::Tensor>& cache_index,
    const c10::optional<at::Tensor>& dequant_scale_x, const c10::optional<at::Tensor>& dequant_scale_w_dq,
    const c10::optional<at::Tensor>& dequant_scale_w_uq_qr, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr, const c10::optional<at::Tensor>& quant_scale_ckv,
    const c10::optional<at::Tensor>& quant_scale_ckr, const c10::optional<at::Tensor>& smooth_scales_cq,
    const c10::optional<at::Tensor>& actual_seq_len, const c10::optional<at::Tensor>& k_nope_clip_alpha,
    double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
    int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale)
{
    // construct the output tensor
    auto token_x_dim = token_x.dim();
    TORCH_CHECK(token_x_dim == 2 || token_x_dim == 3, "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim, OPS_ERROR(ErrCode::PARAM));

    auto weight_uk_dim = weight_uk.dim();
    TORCH_CHECK(weight_uk_dim == 3, "weight_uk dim num should be 3, but the actual value is ", weight_uk_dim, OPS_ERROR(ErrCode::PARAM));

    auto rope_sin_dim = rope_sin.dim();
    TORCH_CHECK(rope_sin_dim == 2 || rope_sin_dim == 3, "rope_sin dim num should be 2 or 3, but the actual value is ", rope_sin_dim, OPS_ERROR(ErrCode::PARAM));

    at::Tensor query;
    at::Tensor query_rope;
    at::Tensor dequant_scale_q_nope;
    at::Tensor query_norm;
    at::Tensor dequant_scale_q_norm;


    if (token_x_dim == 3) {
        if (weight_quant_mode == 2 && kv_cache_quant_mode == 1) {
            // kvcache量化
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({token_x.size(0) * token_x.size(1), weight_uk.size(0), 1}, at::kFloat);
        } else {
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(rope_sin.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
        }
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), rope_sin.size(2)}, at::kBFloat16);
        if (query_norm_flag) {
            query_norm =  npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_dq.size(1)}, token_x.options().dtype(weight_uq_qr.dtype()));
            if (weight_uq_qr.dtype() == at::kChar) {
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({token_x.size(0) * token_x.size(1), 1}, at::kFloat);
            } else {
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
            }
        } else {
            query_norm = npu_preparation::apply_tensor_without_format({1}, token_x.options().dtype(weight_uq_qr.dtype()));
            dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
        }
    } else {
        if (weight_quant_mode == 2 && kv_cache_quant_mode == 1) {
            // kvcache量化
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), 1}, at::kFloat);
        } else {
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(rope_sin.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
        }
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), rope_sin.size(1)}, at::kBFloat16);
        if (query_norm_flag) {
            query_norm =  npu_preparation::apply_tensor_without_format({token_x.size(0), weight_dq.size(1)}, token_x.options().dtype(weight_uq_qr.dtype()));
            if (weight_uq_qr.dtype() == at::kChar) {
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({token_x.size(0), 1}, at::kFloat);
            } else {
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
            }
        } else {
            query_norm = npu_preparation::apply_tensor_without_format({1}, token_x.options().dtype(weight_uq_qr.dtype()));
            dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({1}, at::kFloat);
        }
    }

    char *cache_mode_ptr = const_cast<char *>(cache_mode.data());
    at::Tensor kv_cache_inplace = kv_cache.clone();
    at::Tensor kr_cache_inplace = kr_cache.clone();

    EXEC_NPU_CMD(aclnnMlaPrologV3WeightNz, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache_inplace, kr_cache_inplace, cache_index, dequant_scale_x, dequant_scale_w_dq,
        dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len,
        k_nope_clip_alpha, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode_ptr, query_norm_flag, weight_quant_mode, kv_cache_quant_mode, query_quant_mode,
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query, query_rope,
        dequant_scale_q_nope, query_norm, dequant_scale_q_norm);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, kv_cache_inplace, kr_cache_inplace);
}

} // namespace op_api

namespace at_npu {
namespace functionalization {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mla_prolog_v3_functional_npu(c10::DispatchKeySet dispatchKeySet, const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
    const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
    at::Tensor& kv_cache, at::Tensor& kr_cache, const c10::optional<at::Tensor>& cache_index,
    const c10::optional<at::Tensor>& dequant_scale_x, const c10::optional<at::Tensor>& dequant_scale_w_dq,
    const c10::optional<at::Tensor>& dequant_scale_w_uq_qr, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr, const c10::optional<at::Tensor>& quant_scale_ckv,
    const c10::optional<at::Tensor>& quant_scale_ckr, const c10::optional<at::Tensor>& smooth_scales_cq,
    const c10::optional<at::Tensor>& actual_seq_len, const c10::optional<at::Tensor>& k_nope_clip_alpha,
    double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
    int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale) {
        at::functionalization::impl::sync(token_x);
        at::functionalization::impl::sync(weight_dq);
        at::functionalization::impl::sync(weight_uq_qr);
        at::functionalization::impl::sync(weight_uk);
        at::functionalization::impl::sync(weight_dkv_kr);
        at::functionalization::impl::sync(rmsnorm_gamma_cq);
        at::functionalization::impl::sync(rmsnorm_gamma_ckv);
        at::functionalization::impl::sync(rope_sin);
        at::functionalization::impl::sync(rope_cos);
        at::functionalization::impl::sync(kv_cache);
        at::functionalization::impl::sync(kr_cache);
        at::functionalization::impl::sync(cache_index);
        at::functionalization::impl::sync(dequant_scale_x);
        at::functionalization::impl::sync(dequant_scale_w_dq);
        at::functionalization::impl::sync(dequant_scale_w_uq_qr);
        at::functionalization::impl::sync(dequant_scale_w_dkv_kr);
        at::functionalization::impl::sync(quant_scale_ckv);
        at::functionalization::impl::sync(quant_scale_ckr);
        at::functionalization::impl::sync(smooth_scales_cq);
        at::functionalization::impl::sync(actual_seq_len);
        at::functionalization::impl::sync(k_nope_clip_alpha);
        
        auto token_x_ = at::functionalization::impl::from_functional_tensor(token_x);
        auto weight_dq_ = at::functionalization::impl::from_functional_tensor(weight_dq);
        auto weight_uq_qr_ = at::functionalization::impl::from_functional_tensor(weight_uq_qr);
        auto weight_uk_ = at::functionalization::impl::from_functional_tensor(weight_uk);
        auto weight_dkv_kr_ = at::functionalization::impl::from_functional_tensor(weight_dkv_kr);
        auto rmsnorm_gamma_cq_ = at::functionalization::impl::from_functional_tensor(rmsnorm_gamma_cq);
        auto rmsnorm_gamma_ckv_ = at::functionalization::impl::from_functional_tensor(rmsnorm_gamma_ckv);
        auto rope_sin_ = at::functionalization::impl::from_functional_tensor(rope_sin);
        auto rope_cos_ = at::functionalization::impl::from_functional_tensor(rope_cos);
        auto kv_cache_ = at::functionalization::impl::from_functional_tensor(kv_cache);
        auto kr_cache_ = at::functionalization::impl::from_functional_tensor(kr_cache);
        auto cache_index_ = at::functionalization::impl::from_functional_tensor(cache_index);
        auto dequant_scale_x_ = at::functionalization::impl::from_functional_tensor(dequant_scale_x);
        auto dequant_scale_w_dq_ = at::functionalization::impl::from_functional_tensor(dequant_scale_w_dq);
        auto dequant_scale_w_uq_qr_ = at::functionalization::impl::from_functional_tensor(dequant_scale_w_uq_qr);
        auto dequant_scale_w_dkv_kr_ = at::functionalization::impl::from_functional_tensor(dequant_scale_w_dkv_kr);
        auto quant_scale_ckv_ = at::functionalization::impl::from_functional_tensor(quant_scale_ckv);
        auto quant_scale_ckr_ = at::functionalization::impl::from_functional_tensor(quant_scale_ckr);
        auto smooth_scales_cq_ = at::functionalization::impl::from_functional_tensor(smooth_scales_cq);
        auto actual_seq_len_ = at::functionalization::impl::from_functional_tensor(actual_seq_len);
        auto k_nope_clip_alpha_ = at::functionalization::impl::from_functional_tensor(k_nope_clip_alpha);

        at::Tensor tmp_input1;
        at::Tensor tmp_input2;

        at::AutoDispatchSkipFunctionalize guard;

        auto tmp_result = at_npu::native::custom_ops::npu_mla_prolog_v3_functional(token_x_, weight_dq_, weight_uq_qr_, weight_uk_, weight_dkv_kr_, rmsnorm_gamma_cq_,
        rmsnorm_gamma_ckv_, rope_sin_, rope_cos_, kv_cache_, kr_cache_, cache_index_,
        rmsnorm_epsilon_cq_, rmsnorm_epsilon_ckv_,
        dequant_scale_x_, dequant_scale_w_dq_,
        dequant_scale_w_uq_qr_, dequant_scale_w_dkv_kr_,
        quant_scale_ckv_, quant_scale_ckr_,
        smooth_scales_cq_, actual_seq_len_, k_nope_clip_alpha_,
        cache_mode, query_norm_flag, weight_quant_mode,
        kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode,
        quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale);
        tmp_input1 = std::get<5>(tmp_result);
        tmp_input2 = std::get<6>(tmp_result);

        at::functionalization::impl::replace_(kv_cache, tmp_input1);
        at::functionalization::impl::replace_(kr_cache, tmp_input2);
        at::functionalization::impl::commit_update(kv_cache);
        at::functionalization::impl::commit_update(kr_cache);
        at::functionalization::impl::sync(kv_cache);
        at::functionalization::impl::sync(kr_cache);

        auto output_0 = at::functionalization::impl::to_functional_tensor(std::get<0>(tmp_result));
        auto output_1 = at::functionalization::impl::to_functional_tensor(std::get<1>(tmp_result));
        auto output_2 = at::functionalization::impl::to_functional_tensor(std::get<2>(tmp_result));
        auto output_3 = at::functionalization::impl::to_functional_tensor(std::get<3>(tmp_result));
        auto output_4 = at::functionalization::impl::to_functional_tensor(std::get<4>(tmp_result));

        return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(output_0, output_1, output_2, output_3, output_4);
    }
}

namespace {
TORCH_LIBRARY_IMPL(npu, Functionalize, m) {
    m.impl("npu_mla_prolog_v3", TORCH_FN(functionalization::mla_prolog_v3_functional_npu));
}
}
}