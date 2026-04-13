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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static std::tuple<at::Tensor, at::Tensor> prepare_binary_tensors(
    const at::Tensor &self, const at::Tensor &other)
{
    at::Tensor self_device = self;
    at::Tensor other_device = other;
    if (torch_npu::utils::is_npu(self)) {
        if (!torch_npu::utils::is_npu(other)) {
            other_device = other.to(self.device());
        }
    } else {
        TORCH_CHECK(torch_npu::utils::is_npu(other),
            "when self is a CPU tensor, other must be an NPU tensor", OPS_ERROR(ErrCode::PARAM));
        self_device = self.to(other.device());
    }
    return std::make_tuple(self_device, other_device);
}

at::Tensor &le_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnLeTensor, acl_op::le_out(self, other, result));
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        auto [self_device, other_device] = prepare_binary_tensors(self, other);
        TORCH_CHECK(torch_npu::utils::is_npu(result),
            "result tensor must be on NPU device", OPS_ERROR(ErrCode::PARAM));
        auto outputSize = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
        npu_preparation::check_tensor({self_device, other_device}, result, outputSize);
        EXEC_NPU_CMD(aclnnLeTensor, self_device, other_device, result);
        return result;
    }
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::check_tensor({self}, result, result.scalar_type(), outputSize);
    EXEC_NPU_CMD(aclnnLeTensor, self, other, result);
    return result;
}

at::Tensor &le_out(const at::Tensor &self, const at::Scalar &other, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnLeScalar, acl_op::le_out(self, other, result));
    auto outputSize = self.sizes();
    npu_preparation::check_tensor({self}, result, result.scalar_type(), outputSize);

    EXEC_NPU_CMD(aclnnLeScalar, self, other, result);
    return result;
}

at::Tensor le(const at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnLeTensor, acl_op::le(self, other));
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        auto [self_device, other_device] = prepare_binary_tensors(self, other);
        auto outputSize = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
        at::Tensor result = npu_preparation::apply_tensor_without_format(
            outputSize, self_device.options().dtype(at::kBool));
        EXEC_NPU_CMD(aclnnLeTensor, self_device, other_device, result);
        return result;
    }
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kBool));
    if (npu_preparation::IsCPUScalar(other)) {
        const at::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnLeScalar, self, other_scalar, result);
    } else {
        EXEC_NPU_CMD(aclnnLeTensor, self, other, result);
    }
    return result;
}

at::Tensor le(const at::Tensor &self, const at::Scalar &other)
{
    DO_COMPATIBILITY(aclnnLeScalar, acl_op::le(self, other));
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        auto outputSize = op_infer::input_same_output_size(self);
        at::Tensor result = npu_preparation::apply_tensor_without_format(
            outputSize, self.options().dtype(at::kBool));
        EXEC_NPU_CMD(aclnnLeScalar, self, other, result);
        return result;
    }
    auto outputSize = op_infer::input_same_output_size(self);
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnLeScalar, self, other, result);
    return result;
}

at::Tensor &le_(at::Tensor &self, const at::Scalar &other)
{
    DO_COMPATIBILITY(aclnnInplaceLeScalar, acl_op::le_(self, other));
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        TORCH_CHECK(torch_npu::utils::is_npu(self),
            "Inplace tensor self must be NPU-Tensor.", OPS_ERROR(ErrCode::PARAM));
        EXEC_NPU_CMD(aclnnInplaceLeScalar, self, other);
        return self;
    }
    EXEC_NPU_CMD(aclnnInplaceLeScalar, self, other);
    return self;
}

at::Tensor &le_(at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnInplaceLeTensor, acl_op::le_(self, other));
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        TORCH_CHECK(torch_npu::utils::is_npu(self),
            "Inplace tensor self must be NPU-Tensor.", OPS_ERROR(ErrCode::PARAM));
        at::Tensor other_device = other;
        if (!torch_npu::utils::is_npu(other)) {
            other_device = other.to(self.device());
        }
        npu_preparation::CheckMemory({self, other_device}, {self});
        EXEC_NPU_CMD(aclnnInplaceLeTensor, self, other_device);
        return self;
    }
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        return op_api::le_(self, other.item());
    } else {
        TORCH_CHECK(self.device() == other.device(),
                    "Expected all tensors to be on the same device, but found at least two devices", OPS_ERROR(ErrCode::INTERNAL));
        npu_preparation::CheckMemory({self, other}, {self});
        EXEC_NPU_CMD(aclnnInplaceLeTensor, self, other);
        return self;
    }
}

}
