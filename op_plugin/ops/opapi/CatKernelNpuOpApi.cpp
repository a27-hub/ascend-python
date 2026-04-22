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

#include <ATen/NamedTensorUtils.h>
#include <ATen/native/TypeProperties.h>
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
const int64_t FP4_IN_UINT8_NUM = 2;

static c10::SmallVector<at::Tensor, op_infer::N> cat_dest_tensor_list_opapi(
    const at::MaterializedITensorListRef& tensors)
{
    c10::SmallVector<at::Tensor, op_infer::N> dst_tensor_list;
    // pytorch supports empty tensors, which needs to be removed from the NPU.
    for (at::Tensor tensor : tensors) {
        if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
            continue;
        }
        dst_tensor_list.emplace_back(tensor);
    }
    return dst_tensor_list;
}

static c10::SmallVector<int64_t, op_infer::SIZE> cat_npu_output_size_opapi(
    c10::SmallVector<at::Tensor, op_infer::N>& tensors, int64_t dimension)
{
    bool allSkipped = true;
    int64_t nDims = 0;
    at::Tensor* notSkippedTensor;
    int numInputs = static_cast<int64_t>(tensors.size());
    auto should_skip = [](const at::Tensor* t) { return t->nbytes() == 0 && t->dim() == 1; };
    for (int i = 0; i < numInputs; i++) {
        if (should_skip(static_cast<at::Tensor*>(&tensors[i]))) {
            continue;
        }
        // found a non-empty tensor
        allSkipped = false;
        notSkippedTensor = static_cast<at::Tensor*>(&tensors[i]);
        nDims = notSkippedTensor->dim();
        break;
    }

    if (allSkipped) {
        c10::SmallVector<int64_t, op_infer::SIZE> size = {0};
        return size;
    }

    // Compute size of the result in the cat dimension
    int64_t cat_dim_size = 0;
    for (int i = 0; i < numInputs; i++) {
        at::Tensor* tensor = static_cast<at::Tensor*>(&tensors[i]);
        if (should_skip(tensor)) {
            continue;
        }
        cat_dim_size += tensor->size(dimension);
    }

    // Compute the size of the result
    c10::SmallVector<int64_t, op_infer::SIZE> size;
    size.resize(nDims);
    for (int dim = 0; dim < nDims; dim++) {
        int64_t result_dim_size = notSkippedTensor->size(dim);
        if (dim == dimension) {
            result_dim_size = cat_dim_size;
        }
        size[dim] = result_dim_size;
    }

    return size;
}

inline void cat_check_no_zero_dim(const at::MaterializedITensorListRef& tensors)
{
    size_t i = 0;
    for (const at::Tensor& t : tensors) {
        TORCH_CHECK(
            t.dim() > 0,
            "zero-dimensional tensor (at position ", i, ") cannot be concatenated" + OPS_ERROR(ErrCode::PARAM));
        i++;
    }
}

at::Tensor& cat_out(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnCat, acl_op::cat_out(tensors, dim, out));
    auto materialized = tensors.materialize();
    cat_check_no_zero_dim(materialized);
    c10::SmallVector<at::Tensor, op_infer::N> inputTensors = cat_dest_tensor_list_opapi(materialized);
    at::TensorList tensor_list(inputTensors.begin(), inputTensors.end());
    int64_t dim_post_expr = 0;
    if (inputTensors.size() > 0) {
        dim_post_expr = inputTensors[0].dim();
    } else {
        npu_preparation::check_tensor({materialized[0].get()}, out, at::IntArrayRef({0}));
        return out;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
    // Checking names before the actual dimensions.
    auto maybe_outnames = at::namedinference::compute_cat_outnames(materialized);

    auto outputSize = cat_npu_output_size_opapi(inputTensors, dim);
    npu_preparation::check_tensor({materialized[0].get()}, out, at::IntArrayRef(outputSize));
    EXEC_NPU_CMD(aclnnCat, tensor_list, dim, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_outnames);
    return out;
}

at::Tensor cat(const at::ITensorListRef& tensors, int64_t dim)
{
    DO_COMPATIBILITY(aclnnCat, acl_op::cat(tensors, dim));
    auto materialized = tensors.materialize();
    cat_check_no_zero_dim(materialized);
    c10::SmallVector<at::Tensor, op_infer::N> inputTensors = cat_dest_tensor_list_opapi(materialized);
    at::TensorList tensor_list(inputTensors.begin(), inputTensors.end());
    at::ScalarType high_type = at::native::result_type(materialized);

    int64_t dim_post_expr = 0;
    if (inputTensors.size() > 0) {
        dim_post_expr = inputTensors[0].dim();
    } else {
        at::Tensor result = npu_preparation::apply_tensor_without_format(materialized[0]);
        return result;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
    // Checking names before the actual dimensions.
    auto maybe_outnames = at::namedinference::compute_cat_outnames(materialized);

    // calculate the output size
    auto outputSize = cat_npu_output_size_opapi(inputTensors, dim);
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(outputSize, inputTensors[0].options().dtype(high_type));
    EXEC_NPU_CMD(aclnnCat, tensor_list, dim, result);
    at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
    return result;
}

at::Tensor& cat_out(at::TensorList tensors, at::Dimname dim, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnCat, acl_op::cat_out(tensors, dim, out));
    return at::cat_out(out, tensors, dimname_to_position(tensors[0], dim));
}

at::Tensor cat(at::TensorList tensors, at::Dimname dim)
{
    DO_COMPATIBILITY(aclnnCat, acl_op::cat(tensors, dim));
    return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

static bool is_fp4_dtype(int64_t dtype)
{
    return dtype == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
           dtype == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2);
}

static c10::SmallVector<at::Tensor, op_infer::N> cat_dest_tensor_list_from_tensorlist(
    at::TensorList tensors)
{
    c10::SmallVector<at::Tensor, op_infer::N> dst_tensor_list;
    for (const at::Tensor& tensor : tensors) {
        if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
            continue;
        }
        dst_tensor_list.emplace_back(tensor);
    }
    return dst_tensor_list;
}

at::Tensor npu_cat(at::TensorList tensors, int64_t dim, c10::optional<int64_t> input_dtype)
{
    // If input_dtype is not specified or not a FP4 type, fall back to standard cat
    if (!input_dtype.has_value() || !is_fp4_dtype(input_dtype.value())) {
        return at::cat(tensors, dim);
    }

    // FP4 path: input tensors are stored as uint8, but logically they are FP4
    int64_t dtype_val = input_dtype.value();

    // Filter out empty tensors
    c10::SmallVector<at::Tensor, op_infer::N> inputTensors = cat_dest_tensor_list_from_tensorlist(tensors);
    TORCH_CHECK(inputTensors.size() > 0,
        "npu_cat: all input tensors are empty" + OPS_ERROR(ErrCode::PARAM));

    // Verify all input tensors are stored as uint8 (Byte)
    for (size_t i = 0; i < inputTensors.size(); i++) {
        TORCH_CHECK(inputTensors[i].scalar_type() == c10::ScalarType::Byte,
            "npu_cat: when input_dtype is FP4, all input tensors must be of dtype uint8 (Byte), "
            "but tensor at position ", i, " has dtype ", inputTensors[i].dtype(),
            OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(inputTensors[i].dim() > 0,
            "zero-dimensional tensor (at position ", i, ") cannot be concatenated",
            OPS_ERROR(ErrCode::PARAM));
    }

    int64_t dim_post_expr = inputTensors[0].dim();
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);

    // Calculate output size based on storage shapes (uint8)
    // For FP4 packed in uint8, directly concatenating on storage shapes is correct:
    //   - Non-last-dim cat: last dim unchanged, other dims add up normally
    //   - Last-dim cat: packed sizes add up correctly (sum of packed = packed of sum)
    auto outputSize = cat_npu_output_size_opapi(inputTensors, dim);

    // Allocate output as uint8 (Byte) tensor
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, c10::ScalarType::Byte);

    // Wrap input/output with the logical FP4 ACL data type
    at::TensorList tensor_list(inputTensors.begin(), inputTensors.end());
    aclDataType acl_dtype = c10_npu::GetAclDataType(dtype_val);
    TensorListWrapper input_wrapper = {tensor_list, acl_dtype};
    TensorWrapper output_wrapper = {result, acl_dtype};

    EXEC_NPU_CMD(aclnnCat, input_wrapper, dim, output_wrapper);
    return result;
}

} // namespace op_api