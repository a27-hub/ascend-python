// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include <cstdint>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr int64_t BLOCK_SIZE_BASE_NUM = 32;
constexpr int64_t BLOCK_SIZE_MAX_NUM = 1024;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t FP4_IN_UINT8_NUM = 2;
constexpr int64_t MIN_INPUT_DIM = 2;
constexpr double DEFAULT_DST_TYPE_MAX = 0.0;

struct TensorByteRange {
  uintptr_t begin;
  uintptr_t end;
};

TensorByteRange get_tensor_byte_range(const at::Tensor& tensor) {
  auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  return {begin, begin + tensor.numel() * tensor.element_size()};
}

bool byte_ranges_overlap(const TensorByteRange& first, const TensorByteRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

void check_tensors_do_not_overlap(
    const at::Tensor& first,
    const char* first_name,
    const at::Tensor& second,
    const char* second_name) {
  TORCH_CHECK(
      !byte_ranges_overlap(get_tensor_byte_range(first), get_tensor_byte_range(second)),
      first_name,
      " and ",
      second_name,
      " must not overlap",
      OPS_ERROR(ErrCode::PARAM));
}
}; // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_dynamic_mx_quant_with_dual_axis(
    const at::Tensor& input,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t scale_alg,
    c10::optional<double> dst_type_max) {
  at::Tensor y1;
  at::Tensor mxscale1;
  at::Tensor y2;
  at::Tensor mxscale2;

  TORCH_CHECK(input.dim() >= MIN_INPUT_DIM, "The input should be at least 2D" + OPS_ERROR(ErrCode::PARAM));
  auto y1_shape = op_infer::array_to_small_vector(input.sizes());
  auto mxscale1_shape = op_infer::array_to_small_vector(input.sizes());
  mxscale1_shape.emplace_back(ALIGN_NUM);
  auto y2_shape = op_infer::array_to_small_vector(input.sizes());
  auto mxscale2_shape = op_infer::array_to_small_vector(input.sizes());
  mxscale2_shape.emplace_back(ALIGN_NUM);
  double dst_type_max_real = dst_type_max.has_value() ? dst_type_max.value() : DEFAULT_DST_TYPE_MAX;

  int64_t last_axis = -1;
  int64_t second_to_last_axis = -2;
  int64_t last_axis_change = last_axis + input.dim();
  int64_t second_to_last_axis_change = second_to_last_axis + input.dim();
  int64_t block_size = 32;
  int64_t last_dim_size = op_infer::CeilDiv(mxscale1_shape[last_axis_change], block_size);
  int64_t second_to_dim_size = op_infer::CeilDiv(mxscale2_shape[second_to_last_axis_change], block_size);
  last_dim_size = (last_dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
  second_to_dim_size = (second_to_dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
  mxscale1_shape[last_axis_change] = last_dim_size;
  mxscale2_shape[second_to_last_axis_change] = second_to_dim_size;
  char* round_mode_ptr = const_cast<char*>(round_mode.data());

  aclDataType y_acltype;
  bool special_output_type =
      (dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
       dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
  ASCEND_LOGI(
      "[npu_dynamic_mx_quant_with_dual_axis]: Getting aclTensor y1 and y2 dtype by Parameter(dst_type): %ld", dst_type);
  if (special_output_type) {
    int64_t y1_last_dim_val = y1_shape[input.dim() - 1];
    int64_t y2_last_dim_val = y2_shape[input.dim() - 1];
    TORCH_CHECK(
        y1_last_dim_val % FP4_IN_UINT8_NUM == 0,
        "The last dim input shape must be divisible by 2 if "
        "y1 dtype is torch_npu.float4_e2m1fn_x2 or torch_npu.float4_e1m2" +
            OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        y2_last_dim_val % FP4_IN_UINT8_NUM == 0,
        "The last dim input shape must be divisible by 2 if "
        "y2 dtype is torch_npu.float4_e2m1fn_x2 or torch_npu.float4_e1m2" +
            OPS_ERROR(ErrCode::PARAM));
    y1_shape[input.dim() - 1] = y1_last_dim_val / FP4_IN_UINT8_NUM;
    y2_shape[input.dim() - 1] = y1_last_dim_val / FP4_IN_UINT8_NUM;
    y1 = npu_preparation::apply_tensor_without_format(y1_shape, c10::ScalarType::Byte);
    y2 = npu_preparation::apply_tensor_without_format(y2_shape, c10::ScalarType::Byte);
    y_acltype = c10_npu::GetAclDataType(dst_type);
  } else {
    y_acltype = c10_npu::GetAclDataType(dst_type);
    at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    y1 = npu_preparation::apply_tensor_without_format(y1_shape, c10::dtype(scalar_dtype));
    y2 = npu_preparation::apply_tensor_without_format(y2_shape, c10::dtype(scalar_dtype));
  }

  mxscale1 = npu_preparation::apply_tensor_without_format(mxscale1_shape, c10::dtype(at::ScalarType::Byte));
  mxscale2 = npu_preparation::apply_tensor_without_format(mxscale2_shape, c10::dtype(at::ScalarType::Byte));
  TensorWrapper y1_wrapper = {y1, y_acltype};
  TensorWrapper y2_wrapper = {y2, y_acltype};
  TensorWrapper mxscale1_wrapper = {mxscale1, aclDataType::ACL_FLOAT8_E8M0};
  TensorWrapper mxscale2_wrapper = {mxscale2, aclDataType::ACL_FLOAT8_E8M0};
  static bool npu_support_v2 = check_aclnn_kernel_available("aclnnDynamicMxQuantWithDualAxisV2");

  if (npu_support_v2) {
    EXEC_NPU_CMD(
        aclnnDynamicMxQuantWithDualAxisV2,
        input,
        round_mode_ptr,
        y_acltype,
        scale_alg,
        dst_type_max_real,
        y1_wrapper,
        mxscale1_wrapper,
        y2_wrapper,
        mxscale2_wrapper);
  } else {
    if (dst_type_max.has_value()) {
      TORCH_NPU_WARN_ONCE(
          "CAUTION: The operator aten::npu_dynamic_mx_quant_with_dual_axis does not support the dst_type_max "
          "parameter and this parameter is currently ignored. "
          "If you want to use the functionality of this parameter, please try to update your CANN version.");
    }
    EXEC_NPU_CMD(
        aclnnDynamicMxQuantWithDualAxis,
        input,
        round_mode_ptr,
        y_acltype,
        scale_alg,
        y1_wrapper,
        mxscale1_wrapper,
        y2_wrapper,
        mxscale2_wrapper);
  }

  return std::make_tuple(y1, mxscale1, y2, mxscale2);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
npu_dynamic_mx_quant_with_dual_axis_out(
    const at::Tensor& input,
    at::Tensor& y1_out,
    at::Tensor& mxscale1_out,
    at::Tensor& y2_out,
    at::Tensor& mxscale2_out,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t scale_alg,
    double dst_type_max) {
  TORCH_CHECK(input.dim() >= MIN_INPUT_DIM, "The input should be at least 2D" + OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "npu_dynamic_mx_quant_with_dual_axis_out only supports a float16 or bfloat16 input, but got ",
      input.scalar_type(),
      OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      input.is_contiguous() && y1_out.is_contiguous() && y2_out.is_contiguous() && mxscale1_out.is_contiguous() &&
          mxscale2_out.is_contiguous(),
      "input and all output tensors must be contiguous" + OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      torch_npu::utils::is_npu(input),
      "input and all output tensors must be NPU tensors" + OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      input.device() == y1_out.device() && input.device() == y2_out.device() &&
          input.device() == mxscale1_out.device() && input.device() == mxscale2_out.device(),
      "input and all output tensors must be on the same device" + OPS_ERROR(ErrCode::PARAM));

  auto y_shape = op_infer::array_to_small_vector(input.sizes());
  const aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
  const bool special_output_type =
      dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
      dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2);
  at::ScalarType y_scalar_type;
  if (special_output_type) {
    TORCH_CHECK(
        y_shape[input.dim() - 1] % FP4_IN_UINT8_NUM == 0,
        "The last dim input shape must be divisible by 2 for a float4 output" + OPS_ERROR(ErrCode::PARAM));
    y_shape[input.dim() - 1] /= FP4_IN_UINT8_NUM;
    y_scalar_type = at::kByte;
  } else {
    y_scalar_type = npu_preparation::convert_to_scalar_type(y_acltype);
  }
  TORCH_CHECK(
      y1_out.scalar_type() == y_scalar_type && y2_out.scalar_type() == y_scalar_type,
      "y1_out and y2_out must use the storage dtype selected by dst_type" + OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      y1_out.sizes().equals(y_shape) && y2_out.sizes().equals(y_shape),
      "y1_out and y2_out have an invalid shape" + OPS_ERROR(ErrCode::PARAM));

  auto mxscale1_shape = op_infer::array_to_small_vector(input.sizes());
  auto mxscale2_shape = op_infer::array_to_small_vector(input.sizes());
  mxscale1_shape.emplace_back(ALIGN_NUM);
  mxscale2_shape.emplace_back(ALIGN_NUM);
  const int64_t last_axis = input.dim() - 1;
  const int64_t second_to_last_axis = input.dim() - 2;
  const int64_t last_dim_size = op_infer::CeilDiv(mxscale1_shape[last_axis], BLOCK_SIZE_BASE_NUM);
  const int64_t second_to_last_dim_size =
      op_infer::CeilDiv(mxscale2_shape[second_to_last_axis], BLOCK_SIZE_BASE_NUM);
  mxscale1_shape[last_axis] = (last_dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
  mxscale2_shape[second_to_last_axis] = (second_to_last_dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
  TORCH_CHECK(
      mxscale1_out.sizes().equals(mxscale1_shape),
      "mxscale1_out has an invalid shape" + OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      mxscale2_out.sizes().equals(mxscale2_shape),
      "mxscale2_out has an invalid shape" + OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      mxscale1_out.scalar_type() == at::kByte && mxscale2_out.scalar_type() == at::kByte,
      "mxscale1_out and mxscale2_out must use uint8 storage for FLOAT8_E8M0 values" +
          OPS_ERROR(ErrCode::PARAM));

  check_tensors_do_not_overlap(input, "input", y1_out, "y1_out");
  check_tensors_do_not_overlap(input, "input", mxscale1_out, "mxscale1_out");
  check_tensors_do_not_overlap(input, "input", y2_out, "y2_out");
  check_tensors_do_not_overlap(input, "input", mxscale2_out, "mxscale2_out");
  check_tensors_do_not_overlap(y1_out, "y1_out", mxscale1_out, "mxscale1_out");
  check_tensors_do_not_overlap(y1_out, "y1_out", y2_out, "y2_out");
  check_tensors_do_not_overlap(y1_out, "y1_out", mxscale2_out, "mxscale2_out");
  check_tensors_do_not_overlap(mxscale1_out, "mxscale1_out", y2_out, "y2_out");
  check_tensors_do_not_overlap(mxscale1_out, "mxscale1_out", mxscale2_out, "mxscale2_out");
  check_tensors_do_not_overlap(y2_out, "y2_out", mxscale2_out, "mxscale2_out");

  char* round_mode_ptr = const_cast<char*>(round_mode.data());
  TensorWrapper y1_wrapper = {y1_out, y_acltype};
  TensorWrapper y2_wrapper = {y2_out, y_acltype};
  TensorWrapper mxscale1_wrapper = {mxscale1_out, aclDataType::ACL_FLOAT8_E8M0};
  TensorWrapper mxscale2_wrapper = {mxscale2_out, aclDataType::ACL_FLOAT8_E8M0};
  static bool npu_support_v2 = check_aclnn_kernel_available("aclnnDynamicMxQuantWithDualAxisV2");
  if (npu_support_v2) {
    EXEC_NPU_CMD(
        aclnnDynamicMxQuantWithDualAxisV2,
        input,
        round_mode_ptr,
        y_acltype,
        scale_alg,
        dst_type_max,
        y1_wrapper,
        mxscale1_wrapper,
        y2_wrapper,
        mxscale2_wrapper);
  } else {
    if (dst_type_max != DEFAULT_DST_TYPE_MAX) {
      TORCH_NPU_WARN_ONCE(
          "CAUTION: npu_dynamic_mx_quant_with_dual_axis_out does not support dst_type_max with this CANN "
          "version, so the parameter is ignored. Please update CANN to use dst_type_max.");
    }
    EXEC_NPU_CMD(
        aclnnDynamicMxQuantWithDualAxis,
        input,
        round_mode_ptr,
        y_acltype,
        scale_alg,
        y1_wrapper,
        mxscale1_wrapper,
        y2_wrapper,
        mxscale2_wrapper);
  }

  return std::tie(y1_out, mxscale1_out, y2_out, mxscale2_out);
}

} // namespace op_api
