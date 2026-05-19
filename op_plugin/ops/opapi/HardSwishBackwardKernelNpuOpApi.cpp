// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
#if VERSION_BETWEEN(V1R11, V2R7)
at::Tensor hardswish_backward(const at::Tensor & grad_output, const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnHardswishBackward, acl_op::hardswish_backward(grad_output, self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                  grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHardswishBackward, grad_output, self, out);
    return out;
}
#endif

#if VERSION_BETWEEN(V2R8, VERSION_NEWEST)
at::Tensor hardswish_backward(const at::Tensor & grad_output, const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnHardswishBackwardV2, acl_op::hardswish_backward(grad_output, self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                  grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHardswishBackwardV2, grad_output, self, out);
    return out;
}
#endif
}
