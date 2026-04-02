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

inline void resize_output_check(const at::Tensor &out, at::IntArrayRef expect_size)
{
    if (out.sizes().equals(expect_size)) {
        return;
    }
    if (out.numel() != 0){
        TORCH_WARN(
            "An output with one or more elements was resized since it had ",
            "shape ", out.sizes(), ", which does not match the required ",
            "output shape ", expect_size, ". ",
            "This behavior is deprecated, and in a future PyTorch release outputs ",
            "will not be resized unless they have zero elements. You can explicitly ",
            "reuse an out tensor t by resizing it, inplace, to zero elements with ",
            "t.resize_(0).");
        }
    return;
}

at::Tensor ceil(const at::Tensor &self)
{
    DO_COMPATIBILITY(aclnnCeil, acl_op::ceil(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size_0, self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCeil, self, result);
    at::namedinference::propagate_names(result, self);
    return result;
}

at::Tensor &ceil_out(const at::Tensor &self, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnCeil, acl_op::ceil_out(self, result));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    resize_output_check(result, output_size_0);
    npu_preparation::check_tensor({self}, result, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnCeil, self, result);
    at::namedinference::propagate_names(result, self);
    return result;
}
}