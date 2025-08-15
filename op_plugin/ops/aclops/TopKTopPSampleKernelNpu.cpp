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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {

std::tuple<at::Tensor, at::Tensor> topk_topp_sample_nocheck(
    const at::Tensor& logits,
    const at::Tensor& top_ks,
    const at::Tensor& top_ps,
    const at::Tensor& q,
    float eps,
    bool is_need_logits,
    at::Tensor& logits_select_idx,
    at::Tensor& logits_top_kp_select,
) 
{
  at_npu::native::OpCommand cmd;
  cmd.Name("TopKTopPSample")
      .Input(logits)
      .Input(top_ks)
      .Input(top_ps)
      .Input(q)
      .Output(logits_select_idx)
      .Output(logits_top_kp_select)
      .Attr("eps", eps)
      .Attr("is_need_logits", is_need_logits)
      .Run();

  return std::tie(logits_select_idx, logits_top_kp_select);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> topk_topp_sample(
    const at::Tensor& logits,
    const at::Tensor& top_ks,
    const at::Tensor& top_ps,
    const at::Tensor& q,
    float eps,
    bool is_need_logits)
{
  at::Tensor self_cp = npu_preparation::CastBackToOriFormat(logits);
  c10::SmallVector<int64_t, SIZE> logits_select_idx_size = {top_ks.size(0), 1};
  c10::SmallVector<int64_t, SIZE> logits_top_kp_select_size = {logits.size(0), logits.size(1)};

  at::Tensor logits_top_kp_select = npu_preparation::apply_tensor(self_cp, logits_top_kp_select_size);
  at::Tensor logits_select_idx = npu_preparation::apply_tensor_with_format(
      logits_select_idx_size, self_cp.options().dtype(at::kLong), ACL_FORMAT_ND);
  topk_topp_sample_nocheck(logits, top_ks, top_ps, q, eps, is_need_logits, logits_select_idx, logits_top_kp_select);

  // logits_select_idx dtype transform Int64
  logits_select_idx = at_npu::native::custom_ops::npu_dtype_cast(logits_select_idx, at::kLong);

  return std::tie(logits_select_idx, logits_top_kp_select);
}
} // namespace acl_op
