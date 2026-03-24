// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#ifndef __TORCH_NPU_OP_PLUGIN_lazy_fusion_INTERFACE__
#define __TORCH_NPU_OP_PLUGIN_lazy_fusion_INTERFACE__

#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include "op_plugin/utils/Export.h"

namespace lazy_fusion {

// Cast
OP_PLUGIN_HIDDEN at::Tensor npu_dtype_cast(const at::Tensor & self, at::ScalarType dtype);
OP_PLUGIN_HIDDEN at::Tensor _npu_dtype_cast(const at::Tensor & self, at::ScalarType dtype);

// Unary
OP_PLUGIN_HIDDEN at::Tensor abs(const at::Tensor & self);
OP_PLUGIN_HIDDEN at::Tensor neg(const at::Tensor & self);
OP_PLUGIN_HIDDEN at::Tensor sqrt(const at::Tensor & self);
OP_PLUGIN_HIDDEN at::Tensor exp(const at::Tensor & self);
OP_PLUGIN_HIDDEN at::Tensor & exp_(at::Tensor & self);
OP_PLUGIN_HIDDEN at::Tensor reciprocal(const at::Tensor & self);

// Binary: add/sub/mul/div
OP_PLUGIN_HIDDEN at::Tensor add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
OP_PLUGIN_HIDDEN at::Tensor & add_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
OP_PLUGIN_HIDDEN at::Tensor sub(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
OP_PLUGIN_HIDDEN at::Tensor & sub_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
OP_PLUGIN_HIDDEN at::Tensor mul(const at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor mul(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor & mul_(at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor & mul_(at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor div(const at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor div(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor & div_(at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor & div_(at::Tensor & self, const at::Tensor & other);

// Comparison
OP_PLUGIN_HIDDEN at::Tensor eq(const at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor eq(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor ne(const at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor ne(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor gt(const at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor gt(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor ge(const at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor ge(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor lt(const at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor lt(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor le(const at::Tensor & self, const at::Scalar & other);
OP_PLUGIN_HIDDEN at::Tensor le(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor maximum(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor minimum(const at::Tensor & self, const at::Tensor & other);

// Activation
OP_PLUGIN_HIDDEN at::Tensor sigmoid(const at::Tensor & self);
OP_PLUGIN_HIDDEN at::Tensor sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & output);
OP_PLUGIN_HIDDEN at::Tensor gelu(const at::Tensor & self, c10::string_view approximate);
OP_PLUGIN_HIDDEN at::Tensor gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate);
OP_PLUGIN_HIDDEN at::Tensor relu(const at::Tensor & self);
OP_PLUGIN_HIDDEN at::Tensor & relu_(at::Tensor & self);
OP_PLUGIN_HIDDEN at::Tensor silu(const at::Tensor & self);
OP_PLUGIN_HIDDEN at::Tensor silu_backward(const at::Tensor & grad_output, const at::Tensor & self);

// BatchNorm
OP_PLUGIN_HIDDEN ::std::tuple<at::Tensor, at::Tensor> batch_norm_stats(const at::Tensor & input, double eps);
OP_PLUGIN_HIDDEN ::std::tuple<at::Tensor, at::Tensor> batch_norm_gather_stats_with_counts(
    const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd,
    const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var,
    double momentum, double eps, const at::Tensor & counts);
OP_PLUGIN_HIDDEN at::Tensor batch_norm_elemt(const at::Tensor & input, const c10::optional<at::Tensor> & weight,
    const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps);
OP_PLUGIN_HIDDEN at::Tensor batch_norm_backward_elemt(const at::Tensor & grad_out, const at::Tensor & input,
    const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight,
    const at::Tensor & sum_dy, const at::Tensor & sum_dy_xmu, const at::Tensor & count);

// Reduce
OP_PLUGIN_HIDDEN at::Tensor sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
OP_PLUGIN_HIDDEN at::Tensor sum(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim,
    c10::optional<at::ScalarType> dtype);

// MatMul
OP_PLUGIN_HIDDEN at::Tensor matmul(const at::Tensor & self, const at::Tensor & other);
OP_PLUGIN_HIDDEN at::Tensor mm(const at::Tensor & self, const at::Tensor & mat2);
OP_PLUGIN_HIDDEN at::Tensor bmm(const at::Tensor & self, const at::Tensor & mat2);

// Foreach
OP_PLUGIN_HIDDEN std::vector<at::Tensor> _foreach_sqrt(at::TensorList tensors);
OP_PLUGIN_HIDDEN void _foreach_sqrt_(at::TensorList tensors);
OP_PLUGIN_HIDDEN void _foreach_mul_(const at::TensorList self, const at::Scalar & scalar);
OP_PLUGIN_HIDDEN void _foreach_add_(const at::TensorList self, const at::Scalar & scalar);
OP_PLUGIN_HIDDEN void _foreach_div_(at::TensorList self, const at::Scalar & scalar);
OP_PLUGIN_HIDDEN void _foreach_div_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars);
OP_PLUGIN_HIDDEN void _foreach_addcmul_(const at::TensorList input, const at::TensorList tensors1,
    const at::TensorList tensors2, const at::Scalar & scalar);
OP_PLUGIN_HIDDEN void _foreach_addcdiv_(const at::TensorList input, const at::TensorList tensors1,
    const at::TensorList tensors2, const at::Scalar & scalar);
OP_PLUGIN_HIDDEN void _foreach_addcdiv_(const at::TensorList input, const at::TensorList tensors1,
    const at::TensorList tensors2, at::ArrayRef<at::Scalar> scalars);

}  // namespace lazy_fusion
#endif  // __TORCH_NPU_OP_PLUGIN_lazy_fusion_INTERFACE__
