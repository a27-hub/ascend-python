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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace{
bool check_optim(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  if (index.dim() == 2 && index.stride(0) == 1 && index.stride(1) == 0 && dim == 0 && self.dim() == 2 &&
      self.size(1) == index.size(1)) {
      return true;
    }
  return false;
}
}

at::Tensor& gather_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGather, acl_op::gather_out(self, dim, index, sparse_grad, result));
  auto output_size = index.sizes();
  npu_preparation::check_tensor(
      {self},
      result,
      self.scalar_type(),
      output_size);
  if (!check_optim){
    EXEC_NPU_CMD(aclnnGather, self, dim, index, result);
  } else{
    at::Tensor sub_index = index.select(1,0);
    at::TensorList indices = {sub_index};
    EXEC_NPU_CMD(aclnnIndex, self, indices, result);
  }
  return result;
}

at::Tensor& gather_out(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGather, acl_op::gather_out(self, dim, index, sparse_grad, result));
  const int64_t real_dim = dimname_to_position(self, dim);
  op_api::gather_out(self, real_dim, index, sparse_grad, result);
  return result;
}

at::Tensor gather(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {
  DO_COMPATIBILITY(aclnnGather, acl_op::gather(self, dim, index, sparse_grad));
  auto outputSize = index.sizes();
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, outputSize);
  op_api::gather_out(self, dim, index, sparse_grad, result);
  return result;
}

at::Tensor gather(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad) {
  DO_COMPATIBILITY(aclnnGather, acl_op::gather(self, dim, index, sparse_grad));
  auto outputSize = index.sizes();
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, outputSize);
  op_api::gather_out(self, dim, index, sparse_grad, result);
  return result;
}
} // namespace op_api
