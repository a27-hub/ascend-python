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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_api_common.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& npu_nsa_nocheck(
const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& gamma_cq,
const at::Tensor& gamma_ckv, const at::Tensor& prolog_sin, const at::Tensor& prolog_cos,
const at::Tensor& cache_index, const at::Tensor& cmp_cache_index, const at::Tensor& kv_cache,
const at::Tensor& kr_cache, const at::Tensor& cmp_kv_cache, const at::Tensor& cmp_kr_cache,
const at::Tensor& block_table, const at::Tensor& cmp_block_table, const at::Tensor& act_seqs,
const at::Tensor& act_cmp_seqs, const at::Tensor& weight_mlp_k1, const at::Tensor& weight_mlp_v1,
const at::Tensor& weight_mlp_k2, const at::Tensor& weight_mlp_v2, const at::Tensor& mlp_sin,
const at::Tensor& mlp_cos, const at::Tensor& weight_avg_k, const at::Tensor& weight_avg_v,
const at::Tensor& weight_gate1, const at::Tensor& weight_gate2, const at::Tensor& weight_sim_gate1,
const at::Tensor& weight_uv, const at::Tensor& weight_o, const at::Tensor& dequant_scale_x,
const at::Tensor& dequant_scale_w_dq, const at::Tensor& dequant_scale_w_uq_qr,
const at::Tensor& dequant_scale_w_dkv_kr, const at::Tensor& smooth_scales_cq,
const at::Tensor& dequant_scale_w_gate1, const at::Tensor& dequant_scale_w_sim_gate1,
const at::Tensor& dequant_scale_w_o, const at::Tensor& smooth_scales_w_o, const at::Tensor& atten_mask,
double epsilon_cq, double epsilon_ckv, c10::string_view cache_mode, int64_t block_size, double softmax_scale,
c10::string_view cmp_mode, int64_t slc_block_size, int64_t cmp_block_size, int64_t cmp_stride, int64_t front,
int64_t near, int64_t k, int64_t win, c10::string_view gate_mode, int64_t sparse_mode, at::Tensor& nsa_out)
{
at_npu::native::OpCommand cmd;
cmd.Name("Nsa")
.Input(token_x, "token_x")
.Input(weight_dq, "weight_dq")
.Input(weight_uq_qr, "weight_uq_qr")
.Input(weight_uk, "weight_uk")
.Input(weight_dkv_kr, "weight_dkv_kr")
.Input(gamma_cq, "gamma_cq")
.Input(gamma_ckv, "gamma_ckv")
.Input(prolog_sin, "prolog_sin")
.Input(prolog_cos, "prolog_cos")
.Input(cache_index, "cache_index")
.Input(cmp_cache_index, "cmp_cache_index")
.Input(kv_cache, "kv_cache")
.Input(kr_cache, "kr_cache")
.Input(cmp_kv_cache, "cmp_kv_cache")
.Input(cmp_kr_cache, "cmp_kr_cache")
.Input(block_table, "block_table")
.Input(cmp_block_table, "cmp_block_table")
.Input(act_seqs, "act_seqs")
.Input(act_cmp_seqs, "act_cmp_seqs")
.Input(weight_mlp_k1, "weight_mlp_k1")
.Input(weight_mlp_v1, "weight_mlp_v1")
.Input(weight_mlp_k2, "weight_mlp_k2")
.Input(weight_mlp_v2, "weight_mlp_v2")
.Input(mlp_sin, "mlp_sin")
.Input(mlp_cos, "mlp_cos")
.Input(weight_avg_k, "weight_avg_k")
.Input(weight_avg_v, "weight_avg_v")
.Input(weight_gate1, "weight_gate1")
.Input(weight_gate2, "weight_gate2")
.Input(weight_sim_gate1, "weight_sim_gate1")
.Input(weight_uv, "weight_uv")
.Input(weight_o, "weight_o");

if (dequant_scale_x.defined()) {
    cmd.Input(dequant_scale_x, "dequant_scale_x");
} else {
    cmd.Input();
}

if (dequant_scale_w_dq.defined()) {
    cmd.Input(dequant_scale_w_dq, "dequant_scale_w_dq");
} else {
    cmd.Input();
}

if (dequant_scale_w_uq_qr.defined()) {
    cmd.Input(dequant_scale_w_uq_qr, "dequant_scale_w_uq_qr");
} else {
    cmd.Input();
}

if (dequant_scale_w_dkv_kr.defined()) {
    cmd.Input(dequant_scale_w_dkv_kr, "dequant_scale_w_dkv_kr");
} else {
    cmd.Input();
}

if (smooth_scales_cq.defined()) {
    cmd.Input(smooth_scales_cq, "smooth_scales_cq");
} else {
    cmd.Input();
}

if (dequant_scale_w_gate1.defined()) {
    cmd.Input(dequant_scale_w_gate1, "dequant_scale_w_gate1");
} else {
    cmd.Input();
}

if (dequant_scale_w_sim_gate1.defined()) {
    cmd.Input(dequant_scale_w_sim_gate1, "dequant_scale_w_sim_gate1");
} else {
    cmd.Input();
}

if (dequant_scale_w_o.defined()) {
    cmd.Input(dequant_scale_w_o, "dequant_scale_w_o");
} else {
    cmd.Input();
}

if (smooth_scales_w_o.defined()) {
    cmd.Input(smooth_scales_w_o, "smooth_scales_w_o");
} else {
    cmd.Input();
}

if (atten_mask.defined()) {
    cmd.Input(atten_mask, "atten_mask");
} else {
    cmd.Input();
}

cmd.Output(nsa_out, "nsa_out")
    .Attr("epsilon_cq", static_cast<float>(epsilon_cq))
    .Attr("epsilon_ckv", static_cast<float>(epsilon_ckv))
    .Attr("cache_mode", static_cast<std::string>(cache_mode))
    .Attr("block_size", static_cast<int64_t>(block_size))
    .Attr("softmax_scale", static_cast<float>(softmax_scale))
    .Attr("cmp_mode", static_cast<std::string>(cmp_mode))
    .Attr("slc_block_size", static_cast<int64_t>(slc_block_size))
    .Attr("cmp_block_size", static_cast<int64_t>(cmp_block_size))
    .Attr("cmp_stride", static_cast<int64_t>(cmp_stride))
    .Attr("front", static_cast<int64_t>(front))
    .Attr("near", static_cast<int64_t>(near))
    .Attr("k", static_cast<int64_t>(k))
    .Attr("win", static_cast<int64_t>(win))
    .Attr("gate_mode", static_cast<std::string>(gate_mode))
    .Attr("sparse_mode", static_cast<int64_t>(sparse_mode)).Run();

return nsa_out;
}
} // namespace

std::vectorat::Tensor npu_nsa(
const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& gamma_cq,
const at::Tensor& gamma_ckv, const at::Tensor& prolog_sin, const at::Tensor& prolog_cos,
const at::Tensor& cache_index, const at::Tensor& cmp_cache_index, const at::Tensor& kv_cache,
const at::Tensor& kr_cache, const at::Tensor& cmp_kv_cache, const at::Tensor& cmp_kr_cache,
const at::Tensor& block_table, const at::Tensor& cmp_block_table, const at::Tensor& act_seqs,
const at::Tensor& act_cmp_seqs, const at::Tensor& weight_mlp_k1, const at::Tensor& weight_mlp_v1,
const at::Tensor& weight_mlp_k2, const at::Tensor& weight_mlp_v2, const at::Tensor& mlp_sin,
const at::Tensor& mlp_cos, const at::Tensor& weight_avg_k, const at::Tensor& weight_avg_v,
const at::Tensor& weight_gate1, const at::Tensor& weight_gate2, const at::Tensor& weight_sim_gate1,
const at::Tensor& weight_uv, const at::Tensor& weight_o, const c10::optionalat::Tensor& dequant_scale_x,
const c10::optionalat::Tensor& dequant_scale_w_dq, const c10::optionalat::Tensor& dequant_scale_w_uq_qr,
const c10::optionalat::Tensor& dequant_scale_w_dkv_kr, const c10::optionalat::Tensor& smooth_scales_cq,
const c10::optionalat::Tensor& dequant_scale_w_gate1, const c10::optionalat::Tensor& dequant_scale_w_sim_gate1,
const c10::optionalat::Tensor& dequant_scale_w_o, const c10::optionalat::Tensor& smooth_scales_w_o,
const c10::optionalat::Tensor& atten_mask, double epsilon_cq, double epsilon_ckv, c10::string_view cache_mode,
int64_t block_size, double softmax_scale, c10::string_view cmp_mode, int64_t slc_block_size, int64_t cmp_block_size,
int64_t cmp_stride, int64_t front, int64_t near, int64_t k, int64_t win, c10::string_view gate_mode,
int64_t sparse_mode)
{
std::map<std::string, at::Tensor> mMap = {
{"token_x", token_x}, {"weight_dq", weight_dq}, {"weight_uq_qr", weight_uq_qr}, {"weight_uk", weight_uk},
{"weight_dkv_kr", weight_dkv_kr}, {"gamma_cq", gamma_cq}, {"gamma_ckv", gamma_ckv}, {"prolog_sin", prolog_sin},
{"prolog_cos", prolog_cos}, {"cache_index", cache_index}, {"cmp_cache_index", cmp_cache_index},
{"kv_cache", kv_cache}, {"kr_cache", kr_cache}, {"cmp_kv_cache", cmp_kv_cache}, {"cmp_kr_cache", cmp_kr_cache},
{"block_table", block_table}, {"cmp_block_table", cmp_block_table}, {"act_seqs", act_seqs},
{"act_cmp_seqs", act_cmp_seqs}, {"weight_mlp_k1", weight_mlp_k1}, {"weight_mlp_v1", weight_mlp_v1},
{"weight_mlp_k2", weight_mlp_k2}, {"weight_mlp_v2", weight_mlp_v2}, {"mlp_sin", mlp_sin}, {"mlp_cos", mlp_cos},
{"weight_avg_k", weight_avg_k}, {"weight_avg_v", weight_avg_v}, {"weight_gate1", weight_gate1},
{"weight_gate2", weight_gate2}, {"weight_sim_gate1", weight_sim_gate1}, {"weight_uv", weight_uv},
{"weight_o", weight_o}};

for (auto item : mMap) {
    TORCH_CHECK(item.second.numel() > 0, item.first.c_str(), " should not be null, but the actual value is null",
                OPS_ERROR(ErrCode::PARAM));
}

// construct the output tensor
// 暂时只支持3 dims
auto token_x_dim = token_x.dim();
TORCH_CHECK(token_x_dim == 3, "token_x dim num should be 3,but the actual value is ", token_x_dim,
            OPS_ERROR(ErrCode::PARAM));

auto weight_uk_dim = weight_uk.dim();
TORCH_CHECK(weight_uk_dim == 3, "weight_uk dim num should be 3,but the actual value is ", weight_uk_dim,
            OPS_ERROR(ErrCode::PARAM));

// 暂且只支持3 dims
auto prolog_sin_dim = prolog_sin.dim();
TORCH_CHECK(prolog_sin_dim == 3, "prolog_sin dim num should be 3,but the actual value is ", prolog_sin_dim,
            OPS_ERROR(ErrCode::PARAM));

at::Tensor nsa_out = npu_preparation::apply_tensor_without_format(token_x);

// optional inputs tensor
const at::Tensor& dequant_scale_x_t = c10::value_or_else(dequant_scale_x, [] {return at::Tensor();});
const at::Tensor& dequant_scale_w_dq_t = c10::value_or_else(dequant_scale_w_dq, [] {return at::Tensor(); });
const at::Tensor& dequant_scale_w_uq_qr_t = c10::value_or_else(dequant_scale_w_uq_qr, [] {return at::Tensor();});
const at::Tensor& dequant_scale_w_dkv_kr_t = c10::value_or_else(dequant_scale_w_dkv_kr, [] {return at::Tensor();});
const at::Tensor& smooth_scales_cq_t = c10::value_or_else(smooth_scales_cq, [] {return at::Tensor();});
const at::Tensor& dequant_scale_w_gate1_t = c10::value_or_else(dequant_scale_w_gate1, [] {return at::Tensor();});
const at::Tensor& dequant_scale_w_sim_gate1_t = c10::value_or_else(dequant_scale_w_sim_gate1, [] {return at::Tensor();});
const at::Tensor& dequant_scale_w_o_t = c10::value_or_else(dequant_scale_w_o, [] {return at::Tensor();});
const at::Tensor& smooth_scales_w_o_t = c10::value_or_else(smooth_scales_w_o, [] {return at::Tensor();});
const at::Tensor& atten_mask_t = c10::value_or_else(atten_mask, [] {return at::Tensor();});

// check contiguous
if (!npu_utils::check_match(&nsa_out)) {
    // 若输出非连续，创建连续tensor(contig_tensor)，接收ACLOP算子的输出。再将contig_tensor拷贝到原始输出。
    at::Tensor contiguous_nsa_out = !npu_utils::check_match(&nsa_out) ? npu_utils::format_contiguous(nsa_out) : nsa_out;
    npu_nsa_nocheck(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, gamma_cq, gamma_ckv, prolog_sin,
        prolog_cos, cache_index, cmp_cache_index, kv_cache, kr_cache, cmp_kv_cache, cmp_kr_cache, block_table,
        cmp_block_table, act_seqs, act_cmp_seqs, weight_mlp_k1, weight_mlp_v1, weight_mlp_k2, weight_mlp_v2,
        mlp_sin, mlp_cos, weight_avg_k, weight_avg_v, weight_gate1, weight_gate2, weight_sim_gate1, weight_uv,
        weight_o, dequant_scale_x_t, dequant_scale_w_dq_t, dequant_scale_w_uq_qr_t, dequant_scale_w_dkv_kr_t,
        smooth_scales_cq_t, dequant_scale_w_gate1_t, dequant_scale_w_sim_gate1_t, dequant_scale_w_o_t,
        smooth_scales_w_o_t, atten_mask_t, epsilon_cq, epsilon_ckv, cache_mode, block_size, softmax_scale, cmp_mode,
        slc_block_size, cmp_block_size, cmp_stride, front, near, k, win, gate_mode, sparse_mode, nsa_out);
    npu_utils::format_fresh_view(nsa_out, contiguous_nsa_out);
} else {
    // 若输出连续，直接调用ACLOP算子。
    npu_nsa_nocheck(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, gamma_cq, gamma_ckv, prolog_sin,
        prolog_cos, cache_index, cmp_cache_index, kv_cache, kr_cache, cmp_kv_cache, cmp_kr_cache, block_table,
        cmp_block_table, act_seqs, act_cmp_seqs, weight_mlp_k1, weight_mlp_v1, weight_mlp_k2, weight_mlp_v2,
        mlp_sin, mlp_cos, weight_avg_k, weight_avg_v, weight_gate1, weight_gate2, weight_sim_gate1, weight_uv,
        weight_o, dequant_scale_x_t, dequant_scale_w_dq_t, dequant_scale_w_uq_qr_t, dequant_scale_w_dkv_kr_t,
        smooth_scales_cq_t, dequant_scale_w_gate1_t, dequant_scale_w_sim_gate1_t, dequant_scale_w_o_t,
        smooth_scales_w_o_t, atten_mask_t, epsilon_cq, epsilon_ckv, cache_mode, block_size, softmax_scale, cmp_mode,
        slc_block_size, cmp_block_size, cmp_stride, front, near, k, win, gate_mode, sparse_mode, nsa_out);
}
std::vector<at::Tensor> result;
result.push_back(nsa_out);
return result;
}
} // namespace acl_op