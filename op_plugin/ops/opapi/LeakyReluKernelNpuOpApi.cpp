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

#include <cmath>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
constexpr double kLeakyReluFp16Max = 65504.0;
constexpr double kLeakyReluBf16Max = 3.3895313892515355e38;

inline bool should_use_leaky_relu_fp32_fallback(const at::Tensor& self, const at::Scalar& negative_slope)
{
    const auto dtype = self.scalar_type();
    if (dtype != at::kHalf && dtype != at::kBFloat16) {
        return false;
    }

    const double slope = negative_slope.to<double>();
    if (!std::isfinite(slope)) {
        return true;
    }

    const double limit = dtype == at::kHalf ? kLeakyReluFp16Max : kLeakyReluBf16Max;
    return std::abs(slope) > limit;
}

inline at::Tensor cast_leaky_relu_tensor(const at::Tensor& tensor, at::ScalarType dtype)
{
    return tensor.scalar_type() == dtype ? tensor : tensor.to(dtype);
}

at::Tensor& leaky_relu_backward_out_fallback(const at::Tensor& grad_output,
                                             const at::Tensor& self,
                                             const at::Scalar& negative_slope,
                                             bool self_is_result,
                                             at::Tensor& grad_input)
{
    grad_input.copy_(acl_op::leaky_relu_backward(grad_output, self, negative_slope, self_is_result));
    return grad_input;
}
}  // namespace

at::Tensor leaky_relu(const at::Tensor& self, const at::Scalar& negative_slope)
{
    DO_COMPATIBILITY(aclnnLeakyRelu, acl_op::leaky_relu(self, negative_slope));
    if (should_use_leaky_relu_fp32_fallback(self, negative_slope)) {
        auto self_fp32 = cast_leaky_relu_tensor(self, at::kFloat);
        auto out_fp32 = op_api::leaky_relu(self_fp32, negative_slope);
        return cast_leaky_relu_tensor(out_fp32, self.scalar_type());
    }

    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                  self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLeakyRelu, self, negative_slope, out);
    return out;
}

at::Tensor& leaky_relu_out(const at::Tensor& self, const at::Scalar& negative_slope, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnLeakyRelu, acl_op::leaky_relu_out(self, negative_slope, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    if (should_use_leaky_relu_fp32_fallback(self, negative_slope)) {
        out.copy_(op_api::leaky_relu(self, negative_slope));
        return out;
    }

    EXEC_NPU_CMD(aclnnLeakyRelu, self, negative_slope, out);
    return out;
}

at::Tensor& leaky_relu_(at::Tensor& self, const at::Scalar& negative_slope)
{
    DO_COMPATIBILITY(aclnnInplaceLeakyRelu, acl_op::leaky_relu_(self, negative_slope));
    if (should_use_leaky_relu_fp32_fallback(self, negative_slope)) {
        self.copy_(op_api::leaky_relu(self, negative_slope));
        return self;
    }

    EXEC_NPU_CMD(aclnnInplaceLeakyRelu, self, negative_slope);
    return self;
}

at::Tensor leaky_relu_backward(const at::Tensor& grad_output,
                               const at::Tensor& self,
                               const at::Scalar& negative_slope,
                               bool self_is_result)
{
    DO_COMPATIBILITY(aclnnLeakyReluBackward,
                     acl_op::leaky_relu_backward(grad_output, self, negative_slope, self_is_result));
    if (should_use_leaky_relu_fp32_fallback(self, negative_slope)) {
        auto grad_output_fp32 = cast_leaky_relu_tensor(grad_output, at::kFloat);
        auto self_fp32 = cast_leaky_relu_tensor(self, at::kFloat);
        auto out_fp32 = op_api::leaky_relu_backward(grad_output_fp32, self_fp32, negative_slope, self_is_result);
        return cast_leaky_relu_tensor(out_fp32, self.scalar_type());
    }

    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                  grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLeakyReluBackward, grad_output, self, negative_slope, self_is_result, out);
    return out;
}

at::Tensor& leaky_relu_backward_out(const at::Tensor& grad_output,
                                    const at::Tensor& self,
                                    const at::Scalar& negative_slope,
                                    bool self_is_result,
                                    at::Tensor& grad_input)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    DO_COMPATIBILITY(aclnnLeakyReluBackward,
                     leaky_relu_backward_out_fallback(grad_output, self, negative_slope, self_is_result, grad_input));
    if (should_use_leaky_relu_fp32_fallback(self, negative_slope)) {
        grad_input.copy_(op_api::leaky_relu_backward(grad_output, self, negative_slope, self_is_result));
        return grad_input;
    }

    EXEC_NPU_CMD(aclnnLeakyReluBackward, grad_output, self, negative_slope, self_is_result, grad_input);
    return grad_input;
}
}  // namespace op_api
