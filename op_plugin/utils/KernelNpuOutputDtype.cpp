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

#include <ATen/native/TypeProperties.h>
#include "op_plugin/utils/KernelNpuOutputDtype.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace op_infer {

at::ScalarType angle_out_dtype(const at::Tensor& self)
{
    auto out_dtype = self.scalar_type();
    if (self.is_complex()) {
        out_dtype = self.scalar_type() == at::kComplexFloat ? at::kFloat : at::kDouble;
    } else if (at::isIntegralType(out_dtype, true)) {
        out_dtype = at::kFloat;
    }
    return out_dtype;
}

at::ScalarType polar_out_dtype(const at::Tensor& abs, const at::Tensor& angle)
{
    at::ScalarType high_type = at::native::result_type(abs, angle);
    if (high_type == at::ScalarType::Float) {
        high_type = at::ScalarType::ComplexFloat;
    } else if (high_type == at::ScalarType::Double) {
        high_type = at::ScalarType::ComplexDouble;
    } else if (high_type == at::ScalarType::Half) {
        high_type = at::ScalarType::ComplexHalf;
    }
    return high_type;
}

at::ScalarType clamp_out_dtype(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max)
{
    // 对输入参数进行校验，保证min和max至少有一个不为None
    TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp:At least one of 'min' or 'max' must be not None!");
    // 判断芯片版本号是否为910_95，自定义输出类型推导只针对910_95芯片生效
    bool isRegBaseSoc = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910_95;
    // 定义类型推导保存状态
    at::native::ResultTypeState state = {};
    state = at::native::update_result_type_state(self, state);

    if (isRegBaseSoc) {
        // 如果min为None，则max不为None，最后推导的输出类型为：self和max的推导类型，反之亦然。
        // 如果min，max都不为None，则最后推导的输出类型为：self，min，max的推导类型
        if (!min.has_value()) {
            state = at::native::update_result_type_state(max.value(), state);
        } else if (!max.has_value()){
            state = at::native::update_result_type_state(min.value(), state);
        } else {
            state = at::native::update_result_type_state(max.value(), state);
            state = at::native::update_result_type_state(min.value(), state);
        }
        
    }
    return at::native::result_type(state);
}

at::ScalarType clamp_scalar_out_dtype(const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max)
{
    // 对输入参数进行校验，保证min和max至少有一个不为None
    TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp:At least one of 'min' or 'max' must be not None!");
    // 判断芯片版本号是否为910_95，自定义输出类型推导只针对910_95芯片生效
    bool isRegBaseSoc = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910_95;
    // 定义类型推导保存状态
    at::native::ResultTypeState state = {};
    state = at::native::update_result_type_state(self, state);

    if (isRegBaseSoc) {
        // 如果min为None，则max不为None，最后推导的输出类型为：self和max的推导类型，反之亦然。
        // 如果min，max都不为None，则最后推导的输出类型为：self，min，max的推导类型
        if (!min.has_value()) {
            state = at::native::update_result_type_state(max.value(), state);
        } else if (!max.has_value()){
            state = at::native::update_result_type_state(min.value(), state);
        } else {
            state = at::native::update_result_type_state(max.value(), state);
            state = at::native::update_result_type_state(min.value(), state);
        }
        
    }
    return at::native::result_type(state);
}

} // namespace op_infer
