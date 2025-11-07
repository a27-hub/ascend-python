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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// 定义常量
constexpr uint64_t UNIFORM_INCREMENT = 4;
constexpr float CURAND_2POW32_INV = float(2.3283064e-10)

at::Tensor& uniform_(at::Tensor& self, double from, double to, c10::optional<at::Generator> generator)
{
    DO_COMPATIBILITY(aclnnInplaceUniform, acl_op::uniform_(self, from, to, generator));
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
    auto is_capture = c10_npu::currentStreamCaptureStatusMayInitCtx();
    if (is_capture == c10_npu::CaptureStatus::None) {
        auto pair = gen->philox_engine_inputs(UNIFORM_INCREMENT);
        int64_t seed = static_cast<int64_t>(pair.first);
        int64_t offset = static_cast<int64_t>(pair.second);
        ASCEND_LOGI("uniform_ exec with seed:%lld, offset:%lld, philox_engine_inputs:%llu", seed, offset, UNIFORM_INCREMENT);
        EXEC_NPU_CMD(aclnnInplaceUniform, self, from, to, seed, offset);
    } else {
#if VERSION_BETWEEN(V2R5, VERSION_NEWEST)
        auto gen_state_ = gen->philox_npu_state(10);
        const at::Tensor* seed_ptr = gen_state_.seed_.ptr;
        const at::Tensor* offset_ptr = gen_state_.offset_.ptr;
        const uint64_t offset_intragraph = gen_state_.offset_intragraph_;
        EXEC_NPU_CMD(aclnnInplaceUniformTensor, self, from, to, *seed_ptr, *offset_ptr, offset_intragraph);
#endif
    }
    return self;
}

// 算子返回UInt32的值
at::Tensor& uniform_2(at::Tensor& self, double from, double to, c10::optional<at::Generator> generator)
{
    // 创建与self相同的temp张量，数据类型为UInt32
    at::Tensor temp = at::empty_like(self, at::kUInt32);

    DO_COMPATIBILITY(aclnnInplaceUniform, acl_op::uniform_(self, from, to, generator));
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
    auto is_capture = c10_npu::currentStreamCaptureStatusMayInitCtx();
    if (is_capture == c10_npu::CaptureStatus::None) {
        auto pair = gen->philox_engine_inputs(UNIFORM_INCREMENT);
        int64_t seed = static_cast<int64_t>(pair.first);
        int64_t offset = static_cast<int64_t>(pair.second);
        ASCEND_LOGI("uniform_ exec with seed:%lld, offset:%lld, philox_engine_inputs:%llu", seed, offset, UNIFORM_INCREMENT);
        EXEC_NPU_CMD(aclnnInplaceUniform, temp, from, to, seed, offset);

        // uniform_算子返回值转换
        at::Tensor temp_fp = temp.to(at::kFloat);
        temp_fp.mul_(CURAND_2POW32_INV).add_(CURAND_2POW32_INV / 2.0);
        self.copy_(temp_fp.to(self.dtype));
    } else {
#if VERSION_BETWEEN(V2R5, VERSION_NEWEST)
        auto gen_state_ = gen->philox_npu_state(10);
        const at::Tensor* seed_ptr = gen_state_.seed_.ptr;
        const at::Tensor* offset_ptr = gen_state_.offset_.ptr;
        const uint64_t offset_intragraph = gen_state_.offset_intragraph_;
        EXEC_NPU_CMD(aclnnInplaceUniformTensor, self, from, to, *seed_ptr, *offset_ptr, offset_intragraph);
#endif
    }
    return self;
}

}
