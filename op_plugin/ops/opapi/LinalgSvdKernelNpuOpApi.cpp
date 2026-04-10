// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> _linalg_svd(
    const at::Tensor& A,
    bool full_matrices,
    bool compute_uv,
    c10::optional<c10::string_view> driver)
{
    int64_t MIN_DIM = 2;
    TORCH_CHECK(A.dtype() == at::kFloat, "svd_npu only supported Float, but get", A.dtype(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(A.dim() >= MIN_DIM, "The dim of input tensor must larger than two.", OPS_ERROR(ErrCode::VALUE));

    int64_t m = A.size(-2);
    int64_t n = A.size(-1);

    // Handle empty tensor case
    if (A.numel() == 0) {
        if (compute_uv && full_matrices) {
            if (U.numel() != 0) {
                U.zero_();
                U.diagonal(0, -2, -1).fill_(1.);
            }
            if (Vh.numel() != 0) {
                Vh.zero_();
                Vh.diagonal(0, -2, -1).fill_(1.);
            }
        }
        return std::make_tuple(U, S, Vh);
    }

    return op_api::_linalg_svd_out(A, full_matrices, compute_uv, driver, U, S, Vh);
}
}