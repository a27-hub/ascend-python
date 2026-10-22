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

#include <cmath>
#include <limits>
#include <map>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/FlashAttention.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace op_api {
#if VERSION_BETWEEN(V2R16, VERSION_NEWEST)
namespace {

constexpr int64_t kUnlimitedTokens = std::numeric_limits<int32_t>::max();
constexpr int64_t kCausalMaskSize = 2048;
constexpr int64_t kStatLanes = 8;

at::Tensor causal_mask(const at::Tensor& query) {
  // Keep initialization and reuse on the same stream, including concurrent callers.
  thread_local std::map<std::pair<c10::DeviceIndex, c10::StreamId>, at::Tensor> masks;
  auto stream = c10_npu::getCurrentNPUStream(query.device().index());
  auto key = std::make_pair(stream.device_index(), stream.id());
  auto it = masks.find(key);
  if (it == masks.end()) {
    auto mask = at::ones({kCausalMaskSize, kCausalMaskSize}, query.options().dtype(at::kBool)).triu(1);
    it = masks.emplace(key, std::move(mask)).first;
  }
  return it->second;
}

std::vector<int64_t> sequence_ends(const at::Tensor& cumulative) {
  TORCH_CHECK(
      cumulative.dim() == 1 && cumulative.numel() >= 2,
      "npu flash attention: cumulative lengths must be a 1D tensor with at least two entries",
      OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      cumulative.scalar_type() == at::kInt || cumulative.scalar_type() == at::kLong,
      "npu flash attention: cumulative lengths must be int32 or int64",
      OPS_ERROR(ErrCode::TYPE));
  auto host = cumulative.slice(0, 1).to(at::kCPU, at::kLong).contiguous();
  const auto* data = host.const_data_ptr<int64_t>();
  return {data, data + host.numel()};
}

struct SequenceEnds {
  std::vector<int64_t> query;
  std::vector<int64_t> key;
  bool packed;
  bool shared;

  SequenceEnds(const at::Tensor& q, const at::Tensor& k) : packed(q.defined()), shared(false) {
    TORCH_CHECK(
        q.defined() == k.defined(),
        "npu flash attention: Q/K cumulative lengths must both be set or both be absent",
        OPS_ERROR(ErrCode::PARAM));
    if (packed) {
      query = sequence_ends(q);
      shared = q.is_same(k);
      if (!shared) {
        key = sequence_ends(k);
      }
    }
  }

  at::OptionalIntArrayRef q() const {
    return packed ? at::OptionalIntArrayRef(query) : std::nullopt;
  }
  at::OptionalIntArrayRef k() const {
    return packed ? at::OptionalIntArrayRef(shared ? query : key) : std::nullopt;
  }
};

struct SparseArgs {
  int64_t mode;
  int64_t pre;
  int64_t next;
};

SparseArgs sparse_args(bool causal, std::optional<int64_t> window_left, std::optional<int64_t> window_right) {
  auto left = window_left.value_or(-1);
  auto right = window_right.value_or(-1);
  TORCH_CHECK_NOT_IMPLEMENTED(
      right == -1 || right == 0,
      "npu flash attention: window_size_right must be -1 or 0",
      OPS_ERROR(ErrCode::NOT_SUPPORT));
  if (left >= 0) {
    return {4, left, (causal || right == 0) ? 0 : kUnlimitedTokens};
  }
  return (causal || right == 0) ? SparseArgs{3, kUnlimitedTokens, 0}
                                : SparseArgs{0, kUnlimitedTokens, kUnlimitedTokens};
}

at::Tensor lse_from_stats(const at::Tensor& maximum, const at::Tensor& sum, bool packed) {
  auto lse = maximum.select(-1, 0) + sum.select(-1, 0).log();
  return packed ? lse.transpose(0, 1).contiguous() : lse;
}

std::pair<at::Tensor, at::Tensor> stats_from_lse(const at::Tensor& lse, const at::Tensor& query, bool packed) {
  at::Tensor maximum;
  if (packed) {
    maximum = lse.transpose(0, 1).unsqueeze(-1).expand({query.size(0), query.size(1), kStatLanes}).contiguous();
  } else {
    maximum = lse.reshape({query.size(0), query.size(2), query.size(1), 1})
                  .expand({query.size(0), query.size(2), query.size(1), kStatLanes})
                  .contiguous();
  }
  // exp(scores - max) / sum == exp(scores - LSE), so (LSE, 1) is sufficient.
  return {maximum, at::ones_like(maximum)};
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> _flash_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& cum_seq_q,
    const std::optional<at::Tensor>& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<at::Tensor>& seqused_k,
    const std::optional<at::Tensor>& alibi_slopes,
    const std::optional<at::Tensor>& block_table,
    std::optional<int64_t> num_splits) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      dropout_p == 0., "npu flash attention: dropout_p must be 0", OPS_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK_NOT_IMPLEMENTED(
      !alibi_slopes.has_value() && !block_table.has_value() && !seqused_k.has_value(),
      "npu flash attention: alibi_slopes/block_table/seqused_k unsupported",
      OPS_ERROR(ErrCode::NOT_SUPPORT));
  SequenceEnds ends(cum_seq_q.value_or(at::Tensor()), cum_seq_k.value_or(at::Tensor()));
  auto sparse = sparse_args(is_causal, window_size_left, window_size_right);
  auto mask = sparse.mode == 0 ? std::optional<at::Tensor>() : causal_mask(query);
  auto result = flash_attention::forward(
      query,
      key,
      value,
      query.size(ends.packed ? 1 : 2),
      ends.packed ? "TND" : "BSND",
      std::nullopt,
      std::nullopt,
      mask,
      scale.value_or(1. / std::sqrt(query.size(-1))),
      1.,
      sparse.pre,
      sparse.next,
      0,
      std::nullopt,
      ends.q(),
      ends.k(),
      sparse.mode,
      false,
      true,
      ends.packed ? "TND" : "",
      std::nullopt,
      std::nullopt,
      0,
      0);
  auto lse = lse_from_stats(std::get<1>(result), std::get<2>(result), ends.packed);
  auto rng = at::zeros({2}, query.options().dtype(at::kLong));
  auto empty = at::empty({0}, query.options());
  return {std::get<0>(result), lse, rng, empty, empty};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _flash_attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& rng_state,
    const at::Tensor& unused,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      dropout_p == 0., "npu flash attention: dropout_p must be 0", OPS_ERROR(ErrCode::NOT_SUPPORT));
  SequenceEnds ends(cum_seq_q, cum_seq_k);
  auto sparse = sparse_args(is_causal, window_size_left, window_size_right);
  auto mask = sparse.mode == 0 ? std::optional<at::Tensor>() : causal_mask(query);
  auto stats = stats_from_lse(logsumexp, query, ends.packed);
  auto result = flash_attention::backward(
      query,
      key,
      value,
      grad_out.contiguous(),
      query.size(ends.packed ? 1 : 2),
      ends.packed ? "TND" : "BSND",
      std::nullopt,
      std::nullopt,
      mask,
      stats.first,
      stats.second,
      std::nullopt,
      out,
      scale.value_or(1. / std::sqrt(query.size(-1))),
      1.,
      sparse.pre,
      sparse.next,
      0,
      0,
      0,
      0,
      std::nullopt,
      ends.q(),
      ends.k(),
      sparse.mode,
      false,
      true,
      ends.packed ? "TND" : "",
      std::nullopt);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
}

#endif
} // namespace op_api
