// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm_quant(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const at::Tensor& scales1,
    const c10::optional<at::Tensor>& scales2,
    const c10::optional<at::Tensor>& zero_points1,
    const c10::optional<at::Tensor>& zero_points2,
    const c10::optional<at::Tensor>& beta,
    int64_t axis,
    double epsilon,
    bool div_mode,
    bool output_res,
    c10::optional<int64_t> dst_type) {

  // Check kernel availability
  static const bool is_available =
      check_aclnn_kernel_available("aclnnAddRmsNormQuantV2");
  TORCH_CHECK(
      is_available,
      "Current CANN version does not support this api. Please try to update the version of CANN." +
          OPS_ERROR(ErrCode::PARAM));

  // Validate rank: x1 must be 1-8 dimensional
  TORCH_CHECK(
      x1.dim() >= 1 && x1.dim() <= 8,
      "x1 must be 1-8 dimensional, but got ", x1.dim(), "D",
      OPS_ERROR(ErrCode::PARAM));

  // Validate x2 shape must match x1 (cross-parameter consistency)
  TORCH_CHECK(
      x2.sizes() == x1.sizes(),
      "x2 shape must match x1 shape, but got x1 shape ",
      x1.sizes(), " and x2 shape ", x2.sizes(),
      OPS_ERROR(ErrCode::PARAM));

  // Validate gamma dtype must match x1
  TORCH_CHECK(
      gamma.scalar_type() == x1.scalar_type(),
      "gamma dtype must match x1 dtype, but got x1 dtype ",
      x1.scalar_type(), " and gamma dtype ", gamma.scalar_type(),
      OPS_ERROR(ErrCode::PARAM));

  // Validate axis (currently only -1 supported)
  TORCH_CHECK(
      axis == -1,
      "axis currently only supports -1, but got ", axis,
      OPS_ERROR(ErrCode::PARAM));

  // Validate beta shape/dtype must match gamma when provided
  if (beta.has_value() && beta.value().defined()) {
    TORCH_CHECK(
        beta.value().scalar_type() == gamma.scalar_type(),
        "beta dtype must match gamma dtype, but got gamma dtype ",
        gamma.scalar_type(), " and beta dtype ",
        beta.value().scalar_type(),
        OPS_ERROR(ErrCode::PARAM));
  }

  // Resolve output quant dtype
  at::ScalarType y_scalar_dtype;
  aclDataType y_acltype;
  if (dst_type.has_value()) {
    int64_t dst_type_value = dst_type.value();
    y_acltype = c10_npu::GetAclDataType(dst_type_value);
    y_scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
  } else {
    y_acltype = aclDataType::ACL_INT8;
    y_scalar_dtype = at::ScalarType::Char;
  }

  // Output shape: y1, y2 same as x1
  auto y_out_shape = op_infer::array_to_small_vector(x1.sizes());

  // Create y1 output tensor
  at::Tensor y1 = npu_preparation::apply_tensor_without_format(
      y_out_shape, c10::dtype(y_scalar_dtype));

  // Create y2 output tensor (same shape and dtype as y1)
  at::Tensor y2 = npu_preparation::apply_tensor_without_format(
      y_out_shape, c10::dtype(y_scalar_dtype));

  // Create x_out tensor (x1 + x2 result, same shape and dtype as x1)
  at::Tensor x_out = npu_preparation::apply_tensor_without_format(
      x1.sizes(), x1.options());

  // Create rms_norm_out tensor.
  // When output_res is False, the aclnn kernel expects nullptr.
  // For the Python return, always provide a tensor (empty {0} when unused).
  at::Tensor rms_norm_out;
  at::Tensor rms_norm_aclnn;  // undefined → nullptr for aclnn
  if (output_res) {
    rms_norm_out = npu_preparation::apply_tensor_without_format(
        x1.sizes(), x1.options());
    rms_norm_aclnn = rms_norm_out;
  } else {
    rms_norm_out = at::empty({0}, x1.options());
    // rms_norm_aclnn stays undefined → nullptr for aclnn
  }

  // Wrap y1, y2 with their aclDataType
  TensorWrapper y1_wrapper = {y1, y_acltype};
  TensorWrapper y2_wrapper = {y2, y_acltype};

  // Execute aclnn operator
  EXEC_NPU_CMD(
      aclnnAddRmsNormQuantV2,
      x1,
      x2,
      gamma,
      scales1,
      scales2,
      zero_points1,
      zero_points2,
      beta,
      axis,
      epsilon,
      div_mode,
      y1_wrapper,
      y2_wrapper,
      x_out,
      rms_norm_aclnn);

  return std::make_tuple(y1, y2, x_out, rms_norm_out);
}

}  // namespace op_api
