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

using namespace op_plugin::utils;
using namespace op_infer;
using npu_preparation = at_npu::native::OpPreparation;

namespace op_api {

static bool bool_clamp_pair(const at::Tensor & a, const at::Tensor & b)
{
    return a.scalar_type() == at::ScalarType::Bool && b.scalar_type() == at::ScalarType::Bool;
}

at::Tensor clamp_max(const at::Tensor & self, const at::Tensor & max)
{
    // aclnnClampMaxTensor has no BOOL support on 910B4; clamp_max(x, max) == x & max for bool.
    if (bool_clamp_pair(self, max)) {
        return at::logical_and(self, max);
    }
    DO_COMPATIBILITY(aclnnClampMaxTensor, acl_op::clamp_max(self, max));
    auto output_size_0 = op_infer::broadcast_ops_npu_output_size(self, max);
    auto output_dtype_0 = is_gte_cann_version_820rc1()?at::native::result_type(self, max):c10::typeMetaToScalarType(self.dtype());
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnClampMaxTensor, self, max, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & clamp_max_out(const at::Tensor & self, const at::Tensor & max, at::Tensor & out)
{
    if (bool_clamp_pair(self, max)) {
        out.copy_(at::logical_and(self, max));
        return out;
    }
    DO_COMPATIBILITY(aclnnClampMaxTensor, acl_op::clamp_max_out(self, max, out));
    auto output_size_0 = op_infer::broadcast_ops_npu_output_size(self, max);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, max}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnClampMaxTensor, self, max, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

}  // namespace op_api
