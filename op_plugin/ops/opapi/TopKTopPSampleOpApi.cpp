// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> topk_topp_sample(
    const at::Tensor& logits,
    const at::Tensor& top_ks,
    const at::Tensor& top_ps,
    const at::Tensor& q,
    float eps,
    bool is_need_logits)
{
    DO_COMPATIBILITY(aclnnTopKTopPSample, acl_op::topk_topp_sample(logits, top_ks, top_ps, q, eps, is_need_logits));
    //auto output_size = op_infer::topk_topp_sample_npu_output_size(logits, top_ks, top_ps);
    c10::SmallVector<int64_t, SIZE> logits_select_idx_size = {top_ks.size(0), 1};
    c10::SmallVector<int64_t, SIZE> logits_top_kp_select_size = {logits.size(0), logits.size(1)};

    at::Tensor logits_select_idx;
    at::Tensor logits_top_kp_select;
    if (!is_need_logits){
        logits_top_kp_select = at::zeros_like(logits).fill_(logits);
    }else{
        logits_top_kp_select = npu_preparation::apply_tensor_without_format(logits_top_kp_select_size, logits.options());
    }
    if (logits.dim() == 0 && top_ks == 0 && top_ps == 0 && q.dim() == 0) {
        logits_select_idx = at::zeros_like(top_ks);
        return std::tuple<at::Tensor, at::Tensor>(logits_select_idx, logits_top_kp_select);
    }

    logits_select_idx = npu_preparation::apply_tensor_without_format(logits_select_idx_size, logits.options().dtype(at::kLong));
    //EXEC_NPU_CMD(aclnnTopKTopPSample, logits, top_ks, top_ps, q, eps, is_need_logits, logits_select_idx, logits_top_kp_select);
    return std::tuple<at::Tensor, at::Tensor>(logits_select_idx, logits_top_kp_select);
}

} // namespace op_api
