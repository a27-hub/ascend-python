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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {

using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor trace(const at::Tensor &self)
{
    TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ",
        self.dim(), OPS_ERROR(ErrCode::PARAM));
    // aclnnTrace is not implemented on some SoCs (e.g. Ascend950PR raises error 561103),
    // and the DO_COMPATIBILITY weak-symbol guard cannot cover SoC-level kernel absence.
    // Compose trace with diagonal + sum, which are universally available and consistent
    // with CPU semantics.
    return self.diagonal(0, -2, -1).sum();
}
#endif

} // namespace op_plugin
