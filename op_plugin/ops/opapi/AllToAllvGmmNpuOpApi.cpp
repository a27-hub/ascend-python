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

namespace op_api
{
    void check_alltoallv_gmm_param(const at::Tensor &gmm_x, const at::Tensor &gmm_weight,
                                   const c10::optional<at::Tensor> &mm_x, const c10::optional<at::Tensor> &mm_weight)
    {
        TORCH_CHECK(
            gmm_x.dim() == 2, "The dim num of input gmm_x should be 2, but got ", gmm_x.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_weight.dim() == 3,
                    "The dim num of input gmm_weight should be 3, but got ", gmm_weight.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((gmm_x.scalar_type() == at::kBFloat16) || (gmm_x.scalar_type() == at::kHalf),
                    "The dtype of input gmm_x should be float16 or bfloat16, but got ", gmm_x.scalar_type(),
                    OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((gmm_weight.scalar_type() == at::kBFloat16) || (gmm_weight.scalar_type() == at::kHalf),
                    "The dtype of input gmm_weight should be float16 or bfloat16, but got ", gmm_weight.scalar_type(),
                    OPS_ERROR(ErrCode::PARAM));

        if (mm_x.has_value())
        {
            const at::Tensor &mm_x_value = mm_x.value();
            TORCH_CHECK(mm_x_value.dim() == 2,
                        "The dim num of input mm_x should be 2, but got ", mm_x_value.dim(), OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK((mm_x_value.scalar_type() == at::kBFloat16) || (mm_x_value.scalar_type() == at::kHalf),
                        "The dtype of input mm_x should be float16 or bfloat16, but got ", mm_x_value.scalar_type(),
                        OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(mm_weight.has_value(),
                        "The mm_weight should be provided when mm_x is provided", OPS_ERROR(ErrCode::PARAM));
            const at::Tensor &mm_weight_value = mm_weight.value();
            TORCH_CHECK(mm_weight_value.dim() == 2,
                        "The dim num of input mm_weight should be 2, but got ", mm_weight_value.dim(),
                        OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK((mm_weight_value.scalar_type() == at::kBFloat16) || (mm_weight_value.scalar_type() == at::kHalf),
                        "The dtype of input mm_weight should be float16 or bfloat16, but got ", mm_weight_value.scalar_type(),
                        OPS_ERROR(ErrCode::PARAM));
        }
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_alltoallv_gmm(const at::Tensor &gmm_x, const at::Tensor &gmm_weight,
                                                                     c10::string_view group, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts,
                                                                     const c10::optional<at::Tensor> &send_counts_tensor, const c10::optional<at::Tensor> &recv_counts_tensor,
                                                                     const c10::optional<at::Tensor> &mm_x, const c10::optional<at::Tensor> &mm_weight,
                                                                     bool trans_gmm_weight, bool trans_mm_weight, bool permute_out_flag)
    {
        check_alltoallv_gmm_param(gmm_x, gmm_weight, mm_x, mm_weight);
        at::Tensor mm_y{nullptr};
        at::Tensor permute_out{nullptr};
        int64_t a = 0;
        for (auto &i : recv_counts)
        {
            a += i;
        }
        if (mm_x.has_value())
        {
            const at::Tensor &mm_x_value = mm_x.value();
            const at::Tensor &mm_weight_value = mm_weight.value();
            int64_t bs = mm_x_value.size(0);
            int64_t n2 = trans_mm_weight ? mm_weight_value.size(0) : mm_weight_value.size(1);
            mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, mm_x_value.options());
        }
        if (permute_out_flag)
        {
            int64_t h1 = gmm_x.size(1);
            permute_out = at_npu::native::OpPreparation::apply_tensor_without_format({a, h1}, gmm_x.options());
        }
        int64_t n1 = trans_gmm_weight ? gmm_weight.size(1) : gmm_weight.size(2);
        auto gmm_y = at_npu::native::OpPreparation::apply_tensor_without_format({a, n1}, gmm_x.options());
        const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());
        const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
        const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
        const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
        const char *group_ptr = (char *)group.data();
        EXEC_NPU_CMD(aclnnAlltoallvGmm, gmm_x,
                     gmm_x,
                     gmm_weight,
                     send_count_tensor_real,
                     recv_count_tensor_real,
                     mm_x_real,
                     mm_weight_real,
                     group_ptr,
                     ep_world_size,
                     send_counts,
                     recv_counts,
                     trans_gmm_weight,
                     trans_mm_weight,
                     permute_out_flag,
                     gmm_y,
                     mm_y,
                     permute_out);
        return std::make_tuple(gmm_y, mm_y, permute_out);
    }
} // namespace op_api