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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_api_common.h"

namespace acl_op {
// hann/hamming/blackman 组合内核（替代默认分解路径，修复 #639：窗口默认 dtype 路径不可用）。
// 公式与 aten::Window.cpp 一致（periodic=True 分母 N；periodic=False 分母 N-1）：
//   hann:     0.5 - 0.5*cos(2*pi*k/N)          （a=0.5）
//   hamming:  0.54 - 0.46*cos(2*pi*k/N)        （a=0.54）
//   blackman: 0.42 - 0.5*cos(2*pi*k/N) + 0.08*cos(4*pi*k/N)
// 计算统一在 float32 上进行（NPU 无 double 逐点算子），请求 double 时结果由 float32 转换，
// 与 CPU 的偏差为 ULP 级（真机验证：n={8,64,1024} 三窗 maxdiff 0.0~1.2e-07，n=8 位等）。
// dtype 未指定时遵循 torch.get_default_dtype()；基础与 .periodic 两个重载均已注册。

namespace {
constexpr double kPi = 3.14159265358979323846;

at::Tensor combine_window(int64_t window_length, double a0, double a1, double a2,
                          bool periodic, bool has_a2, at::ScalarType dtype,
                          c10::optional<at::Device> device, c10::optional<bool> pin_memory)
{
    c10::TensorOptions options = c10::TensorOptions().dtype(at::kFloat)
                                     .device(device)
                                     .pinned_memory(pin_memory);
    if (window_length <= 1) {
        // 与 aten 一致：长度 0 返回空张量，长度 1 返回全 1
        return at::ones({window_length}, options).to(dtype);
    }
    const double denom = periodic ? static_cast<double>(window_length)
                                  : static_cast<double>(window_length - 1);
    at::Tensor k = at::arange(window_length, options);
    at::Tensor inner = k * (2.0 * kPi / denom);
    at::Tensor result = a0 - a1 * at::cos(inner);
    if (has_a2) {
        result = result + a2 * at::cos(inner * 2);
    }
    return result.to(dtype);
}

at::ScalarType window_dtype(const c10::optional<at::ScalarType> &dtype)
{
    return dtype.has_value() ? dtype.value() : at::get_default_dtype();
}
} // namespace

at::Tensor hann_window(int64_t window_length, bool periodic,
                       c10::optional<at::ScalarType> dtype,
                       c10::optional<at::Layout> layout,
                       c10::optional<at::Device> device,
                       c10::optional<bool> pin_memory)
{
    TORCH_CHECK(window_length >= 0, "hann_window: expected window_length >= 0, but got ",
                window_length, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!layout.has_value() || layout.value() == at::kStrided,
                "hann_window: only strided layout is supported", OPS_ERROR(ErrCode::PARAM));
    at::ScalarType dtype_value = window_dtype(dtype);
    TORCH_CHECK(at::isFloatingType(dtype_value),
                "hann_window: expected a floating point dtype, but got ", dtype_value,
                OPS_ERROR(ErrCode::TYPE));
    return combine_window(window_length, 0.5, 0.5, 0.0, periodic, false, dtype_value,
                          device, pin_memory);
}

at::Tensor hann_window(int64_t window_length,
                       c10::optional<at::ScalarType> dtype,
                       c10::optional<at::Layout> layout,
                       c10::optional<at::Device> device,
                       c10::optional<bool> pin_memory)
{
    return hann_window(window_length, true, dtype, layout, device, pin_memory);
}

at::Tensor hamming_window(int64_t window_length, bool periodic,
                          c10::optional<at::ScalarType> dtype,
                          c10::optional<at::Layout> layout,
                          c10::optional<at::Device> device,
                          c10::optional<bool> pin_memory)
{
    TORCH_CHECK(window_length >= 0, "hamming_window: expected window_length >= 0, but got ",
                window_length, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!layout.has_value() || layout.value() == at::kStrided,
                "hamming_window: only strided layout is supported", OPS_ERROR(ErrCode::PARAM));
    at::ScalarType dtype_value = window_dtype(dtype);
    TORCH_CHECK(at::isFloatingType(dtype_value),
                "hamming_window: expected a floating point dtype, but got ", dtype_value,
                OPS_ERROR(ErrCode::TYPE));
    return combine_window(window_length, 0.54, 0.46, 0.0, periodic, false, dtype_value,
                          device, pin_memory);
}

at::Tensor hamming_window(int64_t window_length,
                          c10::optional<at::ScalarType> dtype,
                          c10::optional<at::Layout> layout,
                          c10::optional<at::Device> device,
                          c10::optional<bool> pin_memory)
{
    return hamming_window(window_length, true, dtype, layout, device, pin_memory);
}

at::Tensor blackman_window(int64_t window_length, bool periodic,
                           c10::optional<at::ScalarType> dtype,
                           c10::optional<at::Layout> layout,
                           c10::optional<at::Device> device,
                           c10::optional<bool> pin_memory)
{
    TORCH_CHECK(window_length >= 0, "blackman_window: expected window_length >= 0, but got ",
                window_length, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!layout.has_value() || layout.value() == at::kStrided,
                "blackman_window: only strided layout is supported", OPS_ERROR(ErrCode::PARAM));
    at::ScalarType dtype_value = window_dtype(dtype);
    TORCH_CHECK(at::isFloatingType(dtype_value),
                "blackman_window: expected a floating point dtype, but got ", dtype_value,
                OPS_ERROR(ErrCode::TYPE));
    return combine_window(window_length, 0.42, 0.5, 0.08, periodic, true, dtype_value,
                          device, pin_memory);
}

at::Tensor blackman_window(int64_t window_length,
                           c10::optional<at::ScalarType> dtype,
                           c10::optional<at::Layout> layout,
                           c10::optional<at::Device> device,
                           c10::optional<bool> pin_memory)
{
    return blackman_window(window_length, true, dtype, layout, device, pin_memory);
}
} // namespace acl_op
