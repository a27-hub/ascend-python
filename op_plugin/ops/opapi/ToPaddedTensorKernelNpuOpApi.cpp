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

#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <torch/library.h>
#include <vector>

namespace op_api {

namespace {

// Compute the number of elements of a component given its sizes,
// assuming contiguous layout.
int64_t num_bytes(at::IntArrayRef sizes) {
  int64_t result = 1;
  int64_t stride = 1;
  for (int64_t ii = static_cast<int64_t>(sizes.size()) - 1; ii >= 0; --ii) {
    result += (sizes[ii] - 1) * stride;
    stride *= sizes[ii];
  }
  return result;
}

// Pad tensor t to goal_shape with constant value, then reshape.
at::Tensor pad_tensor_to_shape(
    const at::Tensor& t,
    at::IntArrayRef goal_shape,
    double value) {
  std::vector<int64_t> padding;
  auto tup = t.sizes();
  for (int64_t i = static_cast<int64_t>(tup.size()) - 1; i >= 0; i--) {
    padding.push_back(0);
    padding.push_back(goal_shape[i] - tup[i]);
  }
  at::Tensor new_tensor = at::constant_pad_nd(t, at::IntArrayRef(padding), value);
  new_tensor = new_tensor.reshape(goal_shape);
  return new_tensor;
}

} // namespace

at::Tensor to_padded_tensor(
    const at::Tensor& self,
    double padding,
    c10::OptionalArrayRef<int64_t> output_size) {
  // Reimplement NestedTensor_to_padded_tensor_generic using dense NPU ops
  // on the buffer directly, avoiding NestedTensor device transfer which is
  // unsupported on NPU (no NestedTensorPrivateUse1 kernel for _to_copy).
  TORCH_CHECK(self.is_nested(), "to_padded_tensor expects a nested tensor");
  const auto* nt = at::native::get_nested_tensor_impl(self);
  TORCH_CHECK(nt != nullptr, "to_padded_tensor expects a nested tensor");

  const auto sizes = nt->get_nested_sizes();
  if (sizes.numel() == 0 || sizes.dim() == 0) {
    return nt->get_buffer().clone();
  }
  TORCH_CHECK(
      self.numel() > 0,
      "to_padded_tensor: at least one constituent tensor should have non-zero numel");

  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_num_columns = sizes.sizes()[1];
  const auto* sizes_data_ptr = sizes.const_data_ptr<int64_t>();

  // Compute max size per dimension from nested_sizes.
  std::vector<int64_t> max_size(sizes_num_columns, 0);
  for (int64_t i = 0; i < sizes_num_rows; i++) {
    for (int64_t j = 0; j < sizes_num_columns; j++) {
      max_size[j] = std::max(
          max_size[j], sizes_data_ptr[i * sizes_num_columns + j]);
    }
  }

  // Compute split sizes for each component.
  std::vector<int64_t> split_sizes;
  split_sizes.reserve(sizes_num_rows);
  for (int64_t i = 0; i < sizes_num_rows; i++) {
    split_sizes.push_back(num_bytes(
        at::IntArrayRef(sizes_data_ptr + i * sizes_num_columns, sizes_num_columns)));
  }

  // Split the buffer into components.
  std::vector<int64_t> nonzero_split_sizes;
  for (const auto s : split_sizes) {
    if (s > 0) {
      nonzero_split_sizes.push_back(s);
    }
  }
  const auto buffer = nt->get_buffer();
  std::vector<at::Tensor> buffers_;
  if (!nonzero_split_sizes.empty()) {
    buffers_ = at::split_with_sizes(buffer, nonzero_split_sizes, 0);
  }

  // Pad each component to max_size and collect.
  std::vector<at::Tensor> buffers;
  buffers.reserve(split_sizes.size());
  int64_t next_buffer = 0;
  for (int64_t i = 0; i < sizes_num_rows; i++) {
    at::IntArrayRef tensor_sizes(
        sizes_data_ptr + i * sizes_num_columns, sizes_num_columns);
    at::Tensor to_pad;
    if (split_sizes[i] > 0) {
      to_pad = buffers_[next_buffer++].reshape(tensor_sizes);
    } else {
      to_pad = at::empty(tensor_sizes, buffer.options());
    }
    buffers.push_back(pad_tensor_to_shape(to_pad, max_size, padding));
  }

  at::Tensor ret_val = at::stack(buffers);

  // Optionally pad to output_size.
  if (output_size.has_value()) {
    auto output_size_ = output_size.value();
    TORCH_CHECK(
        static_cast<int64_t>(output_size_.size()) == ret_val.dim(),
        "Length of output_size does not match NestedTensor dims. "
        "Broadcasting is not supported.");
    for (int64_t i = 0; i < ret_val.dim(); i++) {
      TORCH_CHECK(
          output_size_[i] >= ret_val.size(i),
          "Value in output_size is less than NestedTensor padded size. "
          "Truncation is not supported.");
    }
    return pad_tensor_to_shape(ret_val, output_size_, padding);
  }
  return ret_val;
}

} // namespace op_api

TORCH_LIBRARY_IMPL(aten, NestedTensorPrivateUse1, m) {
  m.impl("to_padded_tensor", TORCH_FN(op_api::to_padded_tensor));
}