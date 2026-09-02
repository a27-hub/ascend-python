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

// Fallback 实现：将 NPU 上的 nested tensor 转换到 CPU 执行原生实现，
// 再将 padded 结果转回 NPU 设备。
// 参考模式：op-plugin/ops/opapi/VarKernelNpuOpApi.cpp 与 ScatterReduceKernelNpuOpApi.cpp
at::Tensor nested_to_padded_tensor(
    const at::Tensor& self,
    double padding,
    c10::OptionalIntArrayRef output_size) {
    // 将 nested tensor 的 buffer 与 sizes 一并转移到 CPU
    at::Tensor self_cpu = self.cpu();
    // 在 CPU 上调用原生 nested_to_padded_tensor（dispatch 到 at::native::nested_to_padded_tensor）
    at::Tensor result_cpu = at::nested_to_padded_tensor(self_cpu, padding, output_size);
    // 将 padded 结果转回原设备
    return result_cpu.to(self.options());
}
} // namespace op_api
