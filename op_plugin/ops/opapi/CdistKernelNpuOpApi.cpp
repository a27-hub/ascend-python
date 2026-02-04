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

at::Tensor _cdist_forward(const at::Tensor &x1, const at::Tensor &x2, const double p, c10::optional<int64_t> compute_mode)
{
    auto soc_version = c10_npu::GetSocVersion();
    bool is_target_chip_series = 
        (soc_version >= c10_npu::SocVersion::Ascend910B1 && soc_version <  c10_npu::SocVersion::Ascend310B1) ||
        (soc_version >= c10_npu::SocVersion::Ascend910_9391 && soc_version <  c10_npu::SocVersion::Ascend910_95) ||
        soc_version == c10_npu::SocVersion::Ascend910B ||
        soc_version == c10_npu::SocVersion::Ascend910ProB;
    if (is_target_chip_series) {
        int64_t mode = compute_mode.value_or(0);
        float p_float;
        p_float = static_cast<float>(p);
        auto output_size = op_infer::cdist_npu_output_size(x1, x2);
        auto type = x1.scalar_type();
        at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, x1.options().dtype(type));
        EXEC_NPU_CMD(aclnnCdist, x1, x2, p_float, mode, out);
        return out;
    } else {
        return acl_op::cdist(x1, x2, p, compute_mode);
    }
}

at::Tensor cdist(const at::Tensor &x1, const at::Tensor &x2, const double p, c10::optional<int64_t> compute_mode)
{
    return at::_cdist_forward(x1, x2, p, compute_mode);
}
}