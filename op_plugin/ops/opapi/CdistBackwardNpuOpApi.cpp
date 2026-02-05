// Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
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

at::Tensor _cdist_backward(
    const at::Tensor& grad,
    const at::Tensor& x1,
    const at::Tensor& x2,
    double p,
    const at::Tensor& cdist)
{
    auto soc_version = c10_npu::GetSocVersion();

    float p_cast;
    
    if (std::isinf(p)) {
        p_cast = -1;
    } else {
        TORCH_CHECK(
            p <= std::numeric_limits<float>::max(),
            "npu does not support float64" + OPS_ERROR(ErrCode::TYPE));
        p_cast = static_cast<float>(p);
    }
    bool p_is_integer = std::floor(p_cast) == p_cast;
    if (p_is_integer) {
        return acl_op::_cdist_backward(grad, x1, x2, p, cdist);
    }
    bool soc_need_aclnn =
        (soc_version >= c10_npu::SocVersion::Ascend910B1 &&
         soc_version <  c10_npu::SocVersion::Ascend310B1) ||
        (soc_version >= c10_npu::SocVersion::Ascend910_9391 &&
         soc_version <  c10_npu::SocVersion::Ascend910_95) ||
        soc_version == c10_npu::SocVersion::Ascend910B ||
        soc_version == c10_npu::SocVersion::Ascend910ProB;
    if (!soc_need_aclnn) {
        return acl_op::_cdist_backward(grad, x1, x2, p, cdist);
    }

    auto output_size = x1.sizes();
    auto output_dtype = grad.scalar_type();

    at::Tensor out = at_npu::native::OpPreparation::apply_tensor_without_format(
        output_size,
        grad.options().dtype(output_dtype));

    EXEC_NPU_CMD(aclnnCdistBackward, grad, x1, x2, cdist, p_cast, out);

    return out;
}
}