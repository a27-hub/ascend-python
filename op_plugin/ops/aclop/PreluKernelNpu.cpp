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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
at::Tensor prelu_common_nocheck(const at::Tensor& self, const at::Tensor& weight) {
    at::Tensor result = npu_preparation::apply_tensor(self);

    at_npu::native::OpCommand cmd;
    cmd.Name("PRelu")
        .Input(self)
        .Input(weight)
        .Output(result)
        .Run();
    return result;
}

#if (defined VERSION_V1R11)
at::Tensor prelu(const at::Tensor& self, const at::Tensor& weight) {
    return prelu_common_nocheck(self, weight);
}
#endif


#if (defined VERSION_V2R0) || (defined VERSION_V2R1) || (defined VERSION_V2R2) || (defined VERSION_V2R3) || (defined VERSION_V2R4)
at::Tensor _prelu_kernel(const at::Tensor& self, const at::Tensor& weight) {
    return prelu_common_nocheck(self, weight);
}
#endif
} // namespace acl_op
