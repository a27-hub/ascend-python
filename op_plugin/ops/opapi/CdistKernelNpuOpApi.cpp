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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
at::Tensor _cdist_forward(const at::Tensor &x1, const at::Tensor &x2, const double p, c10::optional<int64_t> compute_mode)
{
    std::cout << "======================================" << std::endl;
    std::cout << "CDIST: ENTERED op_api::_cdist_forward" << std::endl;
    std::cout << "======================================" << std::endl;
    DO_COMPATIBILITY(aclnnCdist, acl_op::cdist(x1, x2, p, compute_mode));
    std::cout << "**************************************" << std::endl;
    std::cout << "*** CDIST: USING ACLNN PATH! ***" << std::endl;
    std::cout << "**************************************" << std::endl;
    int64_t mode = compute_mode.value_or(0);
    float p_float = static_cast<float>(p);
    auto output_size = op_infer::cdist_npu_output_size(x1, x2);
    auto type = x1.scalar_type();
    at::Tensor out = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, x1.options().dtype(type));
    EXEC_NPU_CMD(aclnnCdist, x1, x2, p_float, mode, out);
    return out;
}

at::Tensor cdist(const at::Tensor &x1, const at::Tensor &x2, const double p, c10::optional<int64_t> compute_mode)
{
    return at::_cdist_forward(x1, x2, p, compute_mode);
}
}