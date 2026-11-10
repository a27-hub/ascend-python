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
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// 与 aclnnAsStrided 的 dtype 列表对齐（HIFLOAT8 在 torch 侧表现为 Byte，已含在内）；
// FLOAT64、COMPLEX128 不在该列表中，落回 InplaceCopy 兜底。
bool isAsStridedSupportedDtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kLong:
    case at::kUInt64:
    case at::kInt:
    case at::kUInt32:
    case at::kFloat:
    case at::kHalf:
    case at::kChar:
    case at::kByte:
    case at::kBFloat16:
    case at::kShort:
    case at::kUInt16:
    case at::kBool:
    case at::kComplexHalf:
    case at::kComplexFloat:
    case at::kFloat8_e5m2:
    case at::kFloat8_e4m3fn:
      return true;
    default:
      return false;
  }
}

bool canUseAsStrided(const at::Tensor& self, at::IntArrayRef shape, at::IntArrayRef stride) {
  // 新接口尚未随 CANN 发布，符号不存在时必须回退到 aclnnInplaceCopy 路径
  static const bool is_available = check_aclnn_kernel_available("aclnnAsStrided");
  const int64_t itemsize = self.element_size();
  const bool storage_alignable = itemsize > 0 && self.storage().nbytes() % itemsize == 0;
  return is_available && !shape.empty() && shape.size() == stride.size() &&
      shape.size() <= 8 && storage_alignable && isAsStridedSupportedDtype(self.scalar_type());
}

// CANN 侧 GM 地址已含视图偏移、storageOffset 相对视图起点计算：必须传入从 storage 起址
// 对齐的视图，才能保持 npu_stride_copy 相对 storage 起址的语义，否则偏移会叠加两次。
at::Tensor alignToStorageBase(const at::Tensor& self) {
  if (self.storage_offset() == 0) {
    return self;
  }
  return self.as_strided({static_cast<int64_t>(self.storage().nbytes()) / self.element_size()}, {1}, 0);
}
} // namespace

// A2/A3 等支持 acl_op 的老 SOC 保持原有 acl_op(GE AsStrided) 实现，行为不变；
// 仅 Ascend950(A5) 等 aclnn-only 设备改走 op_api：优先专用 aclnnAsStrided，
// 接口缺失或参数超出其约束时，回退到 as_strided 元数据视图 + aclnnInplaceCopy 物化。
at::Tensor npu_stride_copy(
    const at::Tensor& self,
    at::IntArrayRef shape,
    at::IntArrayRef stride,
    const at::Scalar& storage_offset) {
  if (!c10_npu::IsAclnnOnly()) {
    return acl_op::npu_stride_copy(self, shape, stride, storage_offset);
  }
  // EXEC_NPU_CMD 的 ConvertTypes 只接受左值，偏移量先落到局部变量。
  const int64_t storage_offset_value = storage_offset.toLong();
  if (canUseAsStrided(self, shape, stride)) {
    at::Tensor x = alignToStorageBase(self);
    at::Tensor result = npu_preparation::apply_tensor_without_format(shape, self.options());
    EXEC_NPU_CMD(aclnnAsStrided, x, shape, stride, storage_offset_value, result);
    return result;
  }
  DO_COMPATIBILITY(aclnnInplaceCopy, acl_op::npu_stride_copy(self, shape, stride, storage_offset));
  at::Tensor view = self.as_strided(shape, stride, storage_offset.toLong());
  at::Tensor result = npu_preparation::apply_tensor_without_format(shape, self.options());
  EXEC_NPU_CMD(aclnnInplaceCopy, result, view);
  return result;
}

at::Tensor& npu_stride_copy_out(
    const at::Tensor& self,
    at::IntArrayRef shape,
    at::IntArrayRef stride,
    const at::Scalar& storage_offset,
    at::Tensor& out) {
  if (!c10_npu::IsAclnnOnly()) {
    return acl_op::npu_stride_copy_out(self, shape, stride, storage_offset, out);
  }
  // EXEC_NPU_CMD 的 ConvertTypes 只接受左值，偏移量先落到局部变量。
  const int64_t storage_offset_value = storage_offset.toLong();
  npu_preparation::check_tensor({self}, out, shape);
  if (canUseAsStrided(self, shape, stride)) {
    at::Tensor x = alignToStorageBase(self);
    EXEC_NPU_CMD(aclnnAsStrided, x, shape, stride, storage_offset_value, out);
    return out;
  }
  DO_COMPATIBILITY(aclnnInplaceCopy, acl_op::npu_stride_copy_out(self, shape, stride, storage_offset, out));
  at::Tensor view = self.as_strided(shape, stride, storage_offset.toLong());
  EXEC_NPU_CMD(aclnnInplaceCopy, out, view);
  return out;
}

} // namespace op_api
