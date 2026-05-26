// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
at::Tensor q_work_tensor_for_r_mode(const at::Tensor& self)
{
    auto q_size = op_infer::array_to_small_vector(self.sizes());
    q_size.back() = std::min<int64_t>(self.size(-2), self.size(-1));
    return npu_preparation::apply_tensor(self, q_size);
}

std::tuple<at::Tensor&, at::Tensor&> qr_out_npu_nocheck(
    at::Tensor& Q,
    at::Tensor& R,
    const at::Tensor& self,
    bool some)
{
    bool full_matrices = !some;
    at_npu::native::OpCommand cmd;
    cmd.Name("Qr")
        .Input(self)
        .Output(Q)
        .Output(R)
        .Attr("full_matrices", full_matrices)
        .Run();
    return std::tie(Q, R);
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> linalg_qr_out(
    const at::Tensor& self,
    c10::string_view mode,
    at::Tensor& Q,
    at::Tensor& R)
{
    bool some = (mode == "complete") ? false : true;
    at::Tensor q_work_tensor;
    at::Tensor& q_for_exec = (mode == "r") ? (q_work_tensor = q_work_tensor_for_r_mode(self)) : Q;
    bool q_match = npu_utils::check_match(&q_for_exec);
    bool r_match = npu_utils::check_match(&R);
    if (!(q_match && r_match)) {
        at::Tensor contiguous_q = q_match ? q_for_exec : npu_utils::format_contiguous(q_for_exec);
        at::Tensor contiguous_r = r_match ? R : npu_utils::format_contiguous(R);
        qr_out_npu_nocheck(contiguous_q, contiguous_r, self, some);
        if (!q_match && mode != "r") {
            npu_utils::format_fresh_view(Q, contiguous_q);
        }
        if (!r_match) {
            npu_utils::format_fresh_view(R, contiguous_r);
        }
    } else {
        qr_out_npu_nocheck(q_for_exec, R, self, some);
    }
    return std::tie(Q, R);
}
} // namespace acl_op
