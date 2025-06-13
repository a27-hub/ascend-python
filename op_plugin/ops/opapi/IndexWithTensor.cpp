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

at::Tensor npu_index_with_tensor(const at::Tensor &x, const at::Tensor &indices0, const at::Tensor &indices1)
{
    auto indices0_size = indices1.sizes();
    int n = indices1_size[0];
    int k = indices1_size[1];
    at::Tensor y = npu_preparation::apply_tensor_without_format(x, {n, k});
    EXEC_NPU_CMD(aclnnIndexWithTensor, x, indices0, indices1, y);
    return y;
}
}  // namespace op_api