// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "op_plugin/utils/AccumulateType.h"

namespace at_npu {

c10::ScalarType toAccumulateType(c10::ScalarType type, c10::DeviceType device)
{
    switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum)                                                                                       \
        case c10::ScalarType::TypeNum:                                                                                       \
            switch (device) {                                                                                                \
                case torch_npu::utils::get_npu_device_type():                                                                           \
                    return c10::CppTypeToScalarType<at_npu::acc_type_device<scalar_t, torch_npu::utils::get_npu_device_type()>>::value; \
                default:                                                                                                     \
                    return c10::CppTypeToScalarType<at_npu::acc_type_device<scalar_t, c10::DeviceType::CPU>>::value;         \
            }

        AT_NPU_SCALAR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE

        default: TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
    }
}

c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_npu)
{
    return is_npu ? toAccumulateType(type, torch_npu::utils::get_npu_device_type()) : toAccumulateType(type, c10::DeviceType::CPU);
}

}  // namespace at_npu
