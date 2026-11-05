// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
//
// Why this file exists:
//   On Ascend950 the SDPA kernel falls back from npu_fusion_attention_v3 (whose seed/offset are
//   Tensor outputs and therefore survive torch.compile functionalization) to npu_fusion_attention
//   (v1), whose seed/offset/numels are plain int outputs. Under torch.compile the meta function of
//   v1 returns constants (0), so the traced backward does not receive the runtime RNG state and
//   re-generates the dropout mask with seed=0/offset=0. This file provides a thin wrapper that
//   exposes the v1 RNG state as int64 tensors (forward) and consumes it again (backward), matching
//   the v3 contract so that AOTAutograd threads the values through the graph.
//
//   TODO: remove once Ascend950 supports npu_fusion_attention_v3 (or the v1 aclnn interface exposes
//   seed/offset as Tensor attributes).

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
using fusion_attention_rng_output = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                               at::Tensor, at::Tensor, at::Tensor>;

static inline at::Tensor rng_scalar_to_tensor(int64_t value, const at::Tensor &like)
{
    return at::tensor({value}, at::TensorOptions().dtype(at::kLong).device(like.device()));
}

fusion_attention_rng_output npu_fusion_attention_tensor_rng(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, int64_t head_num,
    c10::string_view input_layout, const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise,
    c10::OptionalIntArrayRef prefix, c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync,
    c10::string_view softmax_layout, const c10::optional<at::Tensor> &sink,
    const c10::optional<at::Tensor> &dropout_mask, int64_t seed, int64_t offset)
{
    auto output = npu_fusion_attention(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask,
                                       scale, keep_prob, pre_tockens, next_tockens, inner_precise, prefix,
                                       actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync,
                                       softmax_layout, dropout_mask, seed, offset);
    return {std::get<0>(output), std::get<1>(output), std::get<2>(output), std::get<3>(output),
            rng_scalar_to_tensor(std::get<4>(output), query),
            rng_scalar_to_tensor(std::get<5>(output), query),
            rng_scalar_to_tensor(std::get<6>(output), query)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_grad_tensor_rng(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &dy,
    int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max, const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in, const c10::optional<at::Tensor> &attention_in,
    double scale_value, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise,
    const c10::optional<at::Tensor> &seed, const c10::optional<at::Tensor> &offset,
    const c10::optional<at::Tensor> &numels, c10::OptionalIntArrayRef prefix,
    c10::OptionalIntArrayRef actual_seq_qlen, c10::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode,
    bool gen_mask_parallel, bool sync, c10::string_view softmax_layout, const c10::optional<at::Tensor> &sink)
{
    int64_t seed_value = seed.has_value() ? seed->item<int64_t>() : 0;
    int64_t offset_value = offset.has_value() ? offset->item<int64_t>() : 0;
    int64_t numels_value = numels.has_value() ? numels->item<int64_t>() : 0;
    auto output = npu_fusion_attention_grad(query, key, value, dy, head_num, input_layout, pse, padding_mask,
                                            atten_mask, softmax_max, softmax_sum, softmax_in, attention_in,
                                            scale_value, keep_prob, pre_tockens, next_tockens, inner_precise,
                                            seed_value, offset_value, numels_value, prefix, actual_seq_qlen,
                                            actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, softmax_layout,
                                            sink);
    return {std::get<0>(output), std::get<1>(output), std::get<2>(output)};
}
#endif
}  // namespace op_api
