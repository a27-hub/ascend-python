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

namespace op_api {

at::Tensor bitwise_right_shift_Tensor_Scalar(const at::Tensor& self, const at::Scalar& other) {
  return at::bitwise_right_shift(self, at::native::wrapped_scalar_tensor(other, self.device().type()));
}

at::Tensor bitwise_right_shift_Tensor_Scalar(const at::Tensor& self, const at::Scalar& other, const at::Tensor& out) {
  return at::bitwise_right_shift_out(self, at::native::wrapped_scalar_tensor(other, self.device().type()), out);
}

at::Tensor bitwise_right_shift(const at::Scalar& self, const at::Tensor& other) {
  return at::bitwise_right_shift(at::native::wrapped_scalar_tensor(self, other.device().type()), other);
}

} // namespace op_api
