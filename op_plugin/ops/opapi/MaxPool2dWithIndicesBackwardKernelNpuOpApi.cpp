// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Copyright (c) 2025, Facebook CORPORATION.
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

#include <limits>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// aclnnMaxPool2dWithIndicesBackward only accepts int32/int8 indices on products that are not
// aclnn-only, while aten::max_pool2d_with_indices reports Long. Narrow here instead of weakening
// the ATen dtype.
//
// A max_pool2d index produced by a valid forward pass is a flat offset inside one input plane,
// so its largest possible value is height * width - 1. That lets the shape alone bound the
// conversion without any device synchronisation. The bound says nothing about indices a caller
// fabricated outside that range; policing those stays with the ATen contract.
at::Tensor narrow_indices_for_cann(const at::Tensor& self, const at::Tensor& indices)
{
    if (indices.scalar_type() != at::kLong) {
        return indices;
    }
    const int64_t height = self.size(-2);
    const int64_t width = self.size(-1);
    // Largest offset is height * width - 1, so the plane may hold up to INT32_MAX + 1 elements.
    // Compare by division to avoid overflowing the product while computing the bound.
    constexpr int64_t max_plane_numel = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    TORCH_CHECK(width <= 0 || height <= max_plane_numel / width,
                "max_pool2d_with_indices_backward: cannot narrow Long indices to the int32 index "
                "range required by this device, input plane is ", height, "x", width,
                OPS_ERROR(ErrCode::VALUE));
    return indices.to(at::kInt);
}
}  // namespace

at::Tensor max_pool2d_with_indices_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices)
{
    at::Tensor grad_input;
    static auto maxpool2d_sc = at_npu::native::env::CheckCompatibleImpl();
    static const bool is_aclnn_only = c10_npu::IsAclnnOnly();
    if (!is_aclnn_only && !maxpool2d_sc) {
        DO_COMPATIBILITY(aclnnMaxPool2dWithMaskBackward,
                         acl_op::max_pool2d_with_indices_backward(grad_output, self, kernel_size,
                                                                  stride, padding, dilation, ceil_mode, indices));
        grad_input = npu_preparation::apply_tensor_without_format(self);

        EXEC_NPU_CMD(aclnnMaxPool2dWithMaskBackward, grad_output, self, indices, kernel_size,
                     stride, padding, dilation, ceil_mode, grad_input);
    } else {
        DO_COMPATIBILITY(aclnnMaxPool2dWithIndicesBackward,
                         acl_op::max_pool2d_with_indices_backward(grad_output, self, kernel_size,
                                                                  stride, padding, dilation, ceil_mode, indices));
        grad_input = npu_preparation::apply_tensor_without_format(self);

        at::Tensor cann_indices = is_aclnn_only ? indices : narrow_indices_for_cann(self, indices);
        EXEC_NPU_CMD(aclnnMaxPool2dWithIndicesBackward, grad_output, self, cann_indices, kernel_size,
                     stride, padding, dilation, ceil_mode, grad_input);
    }

    return grad_input;
}

at::Tensor& max_pool2d_with_indices_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices,
    at::Tensor& grad_input)
{
    static auto maxpool2d_sc = at_npu::native::env::CheckCompatibleImpl();
    static const bool is_aclnn_only = c10_npu::IsAclnnOnly();
    if (!is_aclnn_only && !maxpool2d_sc) {
        DO_COMPATIBILITY(aclnnMaxPool2dWithMaskBackward,
                         acl_op::max_pool2d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding,
                                                                      dilation, ceil_mode, indices, grad_input));
        EXEC_NPU_CMD(aclnnMaxPool2dWithMaskBackward, grad_output, self, indices, kernel_size,
                     stride, padding, dilation, ceil_mode, grad_input);
    } else {
        DO_COMPATIBILITY(aclnnMaxPool2dWithIndicesBackward,
                         acl_op::max_pool2d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding,
                                                                      dilation, ceil_mode, indices, grad_input));
        at::Tensor cann_indices = is_aclnn_only ? indices : narrow_indices_for_cann(self, indices);
        EXEC_NPU_CMD(aclnnMaxPool2dWithIndicesBackward, grad_output, self, cann_indices, kernel_size,
                     stride, padding, dilation, ceil_mode, grad_input);
    }

    return grad_input;
}

}
