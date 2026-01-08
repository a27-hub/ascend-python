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
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor>;

tensor_list npu_top_k_top_p_sample(const at::Tensor &logits, const at::Tensor &top_k, const at::Tensor &top_p, const c10::optional<at::Tensor> &q_option, const c10::optional<at::Tensor> &min_ps_option, double eps, bool is_need_logits, int64_t top_k_guess, c10::optional<int64_t> ks_max_potion, c10::optional<bool> input_is_logits_option,  c10::optional<c10::string_view> post_sample_option)
{
    const at::Tensor &q = c10::value_or_else(q_option, [] { return at::Tensor(); });
    const at::Tensor &min_ps = c10::value_or_else(min_ps_option, [] { return at::Tensor(); });
    int64_t ks_max = c10::value_or_else(ks_max_potion, [] {return 1024;});
    bool input_is_logits = c10::value_or_else(input_is_logits_option, [] {return true; });
    c10::string_view post_sample = post_sample_option.value_or("qSample");
    char *post_sample_ptr = const_cast<char *>(post_sample.data());

    auto logits_size = logits.sizes();
    auto batch = logits_size[0];
    auto voc_size = logits_size[1];

    at::Tensor logtis_select_idx = npu_preparation::apply_tensor_without_format({batch, 1}, logits.options().dtype(at::kInt));
    at::Tensor logtis_top_kp_select = npu_preparation::apply_tensor_without_format({batch, voc_size}, logits.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnTopKTopPSample, logits, top_k, top_p, q, min_ps, eps, is_need_logits, top_k_guess, ks_max, input_is_logits, post_sample_ptr, logtis_select_idx, logtis_top_kp_select);
    return std::tie(logtis_select_idx, logtis_top_kp_select);
}

}
