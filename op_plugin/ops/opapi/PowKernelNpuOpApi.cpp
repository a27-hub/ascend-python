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
#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"

namespace op_api {
at::Tensor & pow_(at::Tensor & self, const at::Scalar & exponent)
{
    at_npu::native::assert_no_internal_overlap(self);
    DO_COMPATIBILITY(aclnnInplacePowTensorScalar, acl_op::pow_(self, exponent));
    EXEC_NPU_CMD(aclnnInplacePowTensorScalar, self, exponent);
    return self;
}

at::Tensor & pow_(at::Tensor & self, const at::Tensor & exponent)
{
    at_npu::native::assert_no_internal_overlap(self);
    DO_COMPATIBILITY(aclnnInplacePowTensorTensor, acl_op::pow_(self, exponent));
    std::vector<at::Tensor> tensor_list = {self, exponent};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    EXEC_NPU_CMD(aclnnInplacePowTensorTensor, self, exponent);
    at::namedinference::propagate_names_if_nonempty(self, maybe_names);
    return self;
}
}
