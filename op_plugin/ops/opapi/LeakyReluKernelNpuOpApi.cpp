// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

namespace {
at::Tensor leaky_relu_forward_fp32(const at::Tensor & self, const at::Scalar & negative_slope, bool inplace = false)
{
    auto self_fp32 = self.to(at::kFloat);
    if (inplace) {
        EXEC_NPU_CMD(aclnnInplaceLeakyRelu, self_fp32, negative_slope);
        return self_fp32;
    }

    at::Tensor out_fp32 = npu_preparation::apply_tensor_without_format(self.sizes(), self_fp32.options());
    EXEC_NPU_CMD(aclnnLeakyRelu, self_fp32, negative_slope, out_fp32);
    return out_fp32;
}

at::Tensor leaky_relu_backward_fp32(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result)
{
    auto grad_output_fp32 = grad_output.to(at::kFloat);
    auto self_fp32 = self.to(at::kFloat);
    at::Tensor out_fp32 = npu_preparation::apply_tensor_without_format(self.sizes(), grad_output_fp32.options());
    EXEC_NPU_CMD(aclnnLeakyReluBackward, grad_output_fp32, self_fp32, negative_slope, self_is_result, out_fp32);
    return out_fp32;
}
} // namespace

at::Tensor leaky_relu(const at::Tensor & self, const at::Scalar & negative_slope)
{
    DO_COMPATIBILITY(aclnnLeakyRelu, acl_op::leaky_relu(self, negative_slope));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    if (output_dtype_0 == at::kHalf || output_dtype_0 == at::kBFloat16) {
        return leaky_relu_forward_fp32(self, negative_slope).to(output_dtype_0);
    }

    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLeakyRelu, self, negative_slope, out);
    return out;
}

at::Tensor & leaky_relu_out(const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLeakyRelu, acl_op::leaky_relu_out(self, negative_slope, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    if (output_dtype_0 == at::kHalf || output_dtype_0 == at::kBFloat16) {
        out.copy_(leaky_relu_forward_fp32(self, negative_slope));
        return out;
    }

    EXEC_NPU_CMD(aclnnLeakyRelu, self, negative_slope, out);
    return out;
}

at::Tensor & leaky_relu_(at::Tensor & self, const at::Scalar & negative_slope)
{
    DO_COMPATIBILITY(aclnnInplaceLeakyRelu, acl_op::leaky_relu_(self, negative_slope));
    if (self.scalar_type() == at::kHalf || self.scalar_type() == at::kBFloat16) {
        self.copy_(leaky_relu_forward_fp32(self, negative_slope, true));
        return self;
    }

    EXEC_NPU_CMD(aclnnInplaceLeakyRelu, self, negative_slope);
    return self;
}

at::Tensor leaky_relu_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result)
{
    DO_COMPATIBILITY(aclnnLeakyReluBackward, acl_op::leaky_relu_backward(grad_output, self, negative_slope, self_is_result));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    if (self.scalar_type() == at::kHalf || self.scalar_type() == at::kBFloat16) {
        return leaky_relu_backward_fp32(grad_output, self, negative_slope, self_is_result).to(output_dtype_0);
    }

    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLeakyReluBackward, grad_output, self, negative_slope, self_is_result, out);
    return out;
}

at::Tensor & leaky_relu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnLeakyReluBackward, acl_op::leaky_relu_backward_out(grad_output, self, negative_slope, self_is_result, grad_input));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    if (self.scalar_type() == at::kHalf || self.scalar_type() == at::kBFloat16) {
        grad_input.copy_(leaky_relu_backward_fp32(grad_output, self, negative_slope, self_is_result));
        return grad_input;
    }

    EXEC_NPU_CMD(aclnnLeakyReluBackward, grad_output, self, negative_slope, self_is_result, grad_input);
    return grad_input;
}
} // namespace op_api
