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

#pragma once

#include <ATen/ATen.h>
#include "op_plugin/utils/Version.h"

#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
namespace op_api {
namespace flash_attention {

// Internal CANN-backed implementation shared by native ATen and legacy NPU APIs.
// These functions are not registered operators and do not redispatch through npu_* APIs.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& dy,
    int64_t head_num,
    c10::string_view input_layout,
    const c10::optional<at::Tensor>& pse,
    const c10::optional<at::Tensor>& padding_mask,
    const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& softmax_max,
    const c10::optional<at::Tensor>& softmax_sum,
    const c10::optional<at::Tensor>& softmax_in,
    const c10::optional<at::Tensor>& attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise,
    int64_t seed,
    int64_t offset,
    int64_t numels,
    c10::OptionalIntArrayRef prefix,
    c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen,
    int64_t sparse_mode,
    bool gen_mask_parallel,
    bool sync,
    c10::string_view softmax_layout,
    const c10::optional<at::Tensor>& sink);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t> forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    int64_t head_num,
    c10::string_view input_layout,
    const c10::optional<at::Tensor>& pse,
    const c10::optional<at::Tensor>& padding_mask,
    const c10::optional<at::Tensor>& atten_mask,
    double scale,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise,
    c10::OptionalIntArrayRef prefix,
    c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen,
    int64_t sparse_mode,
    bool gen_mask_parallel,
    bool sync,
    c10::string_view softmax_layout,
    const c10::optional<at::Tensor>& sink,
    const c10::optional<at::Tensor>& dropout_mask,
    int64_t seed,
    int64_t offset);

} // namespace flash_attention
} // namespace op_api
#endif
