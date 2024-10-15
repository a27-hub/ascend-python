// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/scalar_op_api.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_mul(const at::TensorList self, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(self);
    return at::native::foreach_tensor_mul_scalar_kernel_slow(self, scalar);
}

void _foreach_mul_(const at::TensorList self, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(self);
    return at::native::foreach_tensor_mul_scalar_kernel_slow_(self, scalar);
}
}
