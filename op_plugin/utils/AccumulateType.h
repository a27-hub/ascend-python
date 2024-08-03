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

#pragma once
#include <ATen/Config.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include "torch_npu/csrc/core/npu/DeviceUtils.h"

namespace at_npu {

template <typename T, c10::DeviceType D>
struct AccumulateTypeDevice {};

template <typename T, bool>
struct AccumulateType {};

template <typename T>
struct AccumulateType<T, false> {
    using type = typename AccumulateTypeDevice<T, c10::DeviceType::CPU>::type;
};

template <typename T>
struct AccumulateType<T, true> {
    using type = typename AccumulateTypeDevice<T, torch_npu::utils::get_npu_device_type()>::type;
};

template <typename T, c10::DeviceType device>
using acc_type_device = typename AccumulateTypeDevice<T, device>::type;

template <typename T, bool is_npu>
using acc_type = typename AccumulateType<T, is_npu>::type;

#define ACC_TYPE(t, acc_t, device_type)         \
    template <>                                   \
    struct AccumulateTypeDevice<t, device_type> { \
        using type = acc_t;                         \
    };

#define NPU_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, torch_npu::utils::get_npu_device_type())
#define CPU_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::CPU)

NPU_ACC_TYPE(c10::Half, float);
NPU_ACC_TYPE(float, float);
NPU_ACC_TYPE(double, double);
NPU_ACC_TYPE(int8_t, int64_t);
NPU_ACC_TYPE(uint8_t, int64_t);
NPU_ACC_TYPE(char, int64_t);
NPU_ACC_TYPE(int16_t, int64_t);
NPU_ACC_TYPE(int32_t, int64_t);
NPU_ACC_TYPE(int64_t, int64_t);
NPU_ACC_TYPE(bool, bool);
NPU_ACC_TYPE(c10::complex<float>, c10::complex<float>);
NPU_ACC_TYPE(c10::complex<double>, c10::complex<double>);

CPU_ACC_TYPE(c10::Half, float);
CPU_ACC_TYPE(float, double);
CPU_ACC_TYPE(double, double);
CPU_ACC_TYPE(int8_t, int64_t);
CPU_ACC_TYPE(uint8_t, int64_t);
CPU_ACC_TYPE(char, int64_t);
CPU_ACC_TYPE(int16_t, int64_t);
CPU_ACC_TYPE(int32_t, int64_t);
CPU_ACC_TYPE(int64_t, int64_t);
CPU_ACC_TYPE(bool, bool);
CPU_ACC_TYPE(c10::complex<float>, c10::complex<double>);
CPU_ACC_TYPE(c10::complex<double>, c10::complex<double>);

#define AT_NPU_SCALAR_TYPES(_) \
 _(uint8_t, Byte)              \
 _(int8_t, Char)               \
 _(int16_t, Short)             \
 _(int64_t, Long)              \
 _(float, Float)               \
 _(double, Double)             \
 _(at::Half, Half)

TORCH_API c10::ScalarType toAccumulateType(
    c10::ScalarType type,
    c10::DeviceType device);
TORCH_API c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_npu);

} // namespace at_npu
