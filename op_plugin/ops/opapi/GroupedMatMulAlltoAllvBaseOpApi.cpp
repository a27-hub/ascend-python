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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    void check_gmm_all_to_all_v_param(const at::Tensor &gmm_x, const at::Tensor &gmm_weight, const at::Tensor &expert_token_num,
                            const c10::optional<at::Tensor> &mm_x, const c10::optional<at::Tensor> &mm_weight)
    {
        TORCH_CHECK(gmm_x.dim() == 2,
            "The dim num of input gmm_x should be 2, but got ", gmm_x.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_weight.dim() == 3,
           "The dim num of input gmm_weight should be 3, but got ", gmm_weight.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(expert_token_num.dim() == 1,
          "The dim num of input expert_token_num should be 1, but got ", expert_token_num.dim(), OPS_ERROR(ErrCode::PARAM));
          TORCH_CHECK((gmm_x.scalar_type() == at::kBFloat16) || (gmm_x.scalar_type() == at::kHalf),
          "The data type of input gmm_x should be bfloat16 or float16, but got ", gmm_x.scalar_type(), OPS_ERROR(ErrCode::PARAM));
       TORCH_CHECK((gmm_weight.scalar_type() == at::kBFloat16) || (gmm_weight.scalar_type() == at::kHalf),
         "The data type of input gmm_weight should be bfloat16 or float16, but got ", gmm_weight.scalar_type(), OPS_ERROR(ErrCode::PARAM));
       TORCH_CHECK(expert_token_num.scalar_type() == at::kLong,
         "The data type of input expert_token_num should be Long, but got ", expert_token_num.scalar_type(), OPS_ERROR(ErrCode::PARAM));
       
       if (mm_x.has_value()) {
           TORCH_CHECK(mm_x.value().dim() == 2,
              "The dim num of input mm_x should be 2, but got ", mm_x.value().dim(), OPS_ERROR(ErrCode::PARAM));
           TORCH_CHECK((mm_x.value().scalar_type() == at::kBFloat16) || (mm_x.value().scalar_type() == at::kHalf),
             "The data type of input mm_x should be bfloat16 or float16, but got ", mm_x.value().scalar_type(), OPS_ERROR(ErrCode::PARAM));
           TORCH_CHECK(mm_weight.has_value(),
            "The input mm_weight should not be empty when mm_x is not empty", OPS_ERROR(ErrCode::PARAM));
           TORCH_CHECK(mm_weight.value().dim() == 2,
            "The dim num of input mm_weight should be 2, but got ", mm_weight.value().dim(), OPS_ERROR(ErrCode::PARAM));
           TORCH_CHECK((mm_weight.value().scalar_type() == at::kBFloat16) || (mm_weight.value().scalar_type() == at::kHalf),
           "The data type of input mm_weight should be bfloat16 or float16, but got ", mm_weight.value().scalar_type(), OPS_ERROR(ErrCode::PARAM));
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_grouped_matmul_alltoallv_base(const at::Tensor &gmm_x, const at::Tensor &gmm_weight,
                                                                                        const at::Tensor &expert_token_num, const c10::optional<at::Tensor> &mm_x,
                                                                                        const c10::optional<at::Tensor> &mm_weight, c10::string_view group, int64_t ep_world_size,
                                                                                        at::IntArrayRef send_counts, at::IntArrayRef recv_counts, bool trans_gmm_weight,
                                                                                        bool trans_mm_weight, bool all_gather_flag, bool gmm_out_flag)
    {
        check_gmm_all_to_all_v_param(gmm_x, gmm_weight, expert_token_num, mm_x, mm_weight);
        at::Tensor y {nullptr};
        at::Tensor mm_y {nullptr};
        at::Tensor all_gather_out {nullptr};
        at::Tensor gmm_out {nullptr};
        int64_t bsk = 0;
        for (auto &i : recv_counts) {
            bsk = bsk + i;
        }
        int64_t n1 = gmm_weight.size(2);
        if (mm_x.has_value()) {
            const at::Tensor &mm_x_value = mm_x.value();
            const at::Tensor &mm_weight_value = mm_weight.value();
            int64_t bs = mm_x_value.size(0);
            int64_t n2 = mm_weight_value.size(1);
            mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, mm_x_value.options());
        }
        if (all_gather_flag) {
            int64_t e = gmm_weight.size(0);
            all_gather_out = at_npu::native::OpPreparation::apply_tensor_without_format({ep_world_size, e * ep_world_size}, expert_token_num.options());
        }
        if (gmm_out_flag) {
            gmm_out = at_npu::native::OpPreparation::apply_tensor_without_format({bsk, n1}, gmm_x.options());
        }
        y = at_npu::native::OpPreparation::apply_tensor_without_format({bsk, n1}, gmm_x.options());
        const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());
        const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
        const char *group_ptr = (char *)(group.data());

        EXEC_NPU_CMD(aclnnGroupedMatMulAlltoAllv, gmm_x, gmm_weight, expert_token_num, mm_x_real, mm_weight_real, group_ptr, ep_world_size,
                send_counts, recv_counts, trans_gmm_weight, trans_mm_weight, all_gather_flag, gmm_out_flag, y, mm_y, all_gather_out, gmm_out);
        return std::tie(y, mm_y, all_gather_out, gmm_out);
    }
}