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

static void check_resize_output(const at::Tensor& output, at::IntArrayRef shape)
{
    if (output.sizes().equals(shape)) {
        return;
    }
    if (output.numel() != 0) {
        TORCH_WARN(
            "An output with one or more elements was resized since it had ",
            "shape ", output.sizes(), ", which does not match the required ",
            "output shape ", shape, ". ",
            "This behavior is deprecated, and in a future PyTorch release outputs ",
            "will not be resized unless they have zero elements. You can explicitly ",
            "reuse an out tensor t by resizing it, inplace, to zero elements with ",
            "t.resize_(0).");
    }
}

at::Tensor& nanmean_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                        c10::optional<c10::ScalarType> dtype, at::Tensor& out)
{
    // Check if dtype is an integral type or Bool and raise an error
    TORCH_CHECK(
        !at::isIntegralType(self.scalar_type(), /*includeBool=*/true),
        "nanmean(): integral types and 'Bool' are not supported for nanmean, even for empty tensors.");
    TORCH_CHECK(
        self.is_floating_point() || self.is_complex(),
        "nanmean(): expected input to have floating point or complex dtype but got ",
        self.scalar_type());
    
    // Compute expected output size
    at::IntArrayRef dimArray;
    c10::SmallVector<int64_t, N> dimlist;
    if (dim.has_value()) {
        dimArray = dim.value();
    } else {
        dimlist = op_plugin::utils::get_dimlist_for_tensor(self);
        dimArray = dimlist;
    }
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dimArray, keepdim);

    // Emit resize warning if out tensor needs resizing (required by test_out_warning)
    check_resize_output(out, output_size);
    
    // Calculate the factor: count of non-NaN values
    const auto factor = at::isnan(self).logical_not_().sum(dim, keepdim);

    // Compute nansum and divide by factor in-place
    op_api::nansum_out(self, dim, keepdim, dtype, out).div_(factor);

    return out;
}

at::Tensor nanmean(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                   c10::optional<c10::ScalarType> dtype)
{
    TORCH_CHECK(
        self.is_floating_point() || self.is_complex(),
        "nanmean(): expected input to have floating point or complex dtype but got ",
        self.scalar_type());

    // Calculate output size
    at::IntArrayRef dimArray;
    c10::SmallVector<int64_t, N> dimlist;
    if (dim.has_value()) {
        dimArray = dim.value();
    } else {
        dimlist = op_plugin::utils::get_dimlist_for_tensor(self);
        dimArray = dimlist;
    }

    // Determine output dtype
    c10::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dimArray, keepdim);

    // Create result tensor
    at::Tensor result =
        at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options().dtype(dstType));

    // Call nanmean_out to compute the result
    op_api::nanmean_out(self, dim, keepdim, dtype, result);
    return result;
}
}
