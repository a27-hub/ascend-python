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

at::Tensor& npu_broadcast_out(const at::Tensor& self, at::IntArrayRef size, at::Tensor& out)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_broadcast_out(self, size, out);
    }

    DO_COMPATIBILITY(aclnnBroadcastTo, acl_op::npu_broadcast_out(self, size, out));
    npu_preparation::check_tensor({self}, out, self.scalar_type(), size);
    EXEC_NPU_CMD(aclnnBroadcastTo, self, size, out);
    return out;
}

at::Tensor npu_broadcast(const at::Tensor& self, at::IntArrayRef size)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_broadcast(self, size);
    }

    DO_COMPATIBILITY(aclnnBroadcastTo, acl_op::npu_broadcast(self, size));
    at::Tensor out = npu_preparation::apply_tensor_without_format(size, self.options());
    EXEC_NPU_CMD(aclnnBroadcastTo, self, size, out);
    return out;
}
} // namespace op_api
