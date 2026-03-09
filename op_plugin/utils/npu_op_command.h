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

#ifndef OP_PLUGIN_UTILS_NPU_OP_COMMAND_H_
#define OP_PLUGIN_UTILS_NPU_OP_COMMAND_H_

#include <cstdint>
#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace op_plugin {

enum NpuParamType : int32_t {
    kTensor = 0,
    kScalar = 1,
    kIntArrayRef = 2,
    kBool = 3,
    kInt64 = 4,
    kDouble = 5,
    kOptionalTensor = 6,
    kOptionalIntArrayRef = 7,
    kOptionalScalar = 8,
    kScalarType = 9,
    kString = 10,
    kTensorList = 11,
    kScalarList = 12,
    kBoolArray = 13,
    kSymIntArrayRef = 14,
    kOptionalSymIntArrayRef = 15,
    kOptionalIntArrayRefV2 = 16,
    kCharArray = 17,
};

struct NpuParam {
    const void* data;
    NpuParamType type;
};

inline NpuParam MakeNpuParam(const at::Tensor& v) { return {&v, NpuParamType::kTensor}; }
inline NpuParam MakeNpuParam(const at::Scalar& v) { return {&v, NpuParamType::kScalar}; }
inline NpuParam MakeNpuParam(const at::IntArrayRef& v) { return {&v, NpuParamType::kIntArrayRef}; }
inline NpuParam MakeNpuParam(const bool& v) { return {&v, NpuParamType::kBool}; }
inline NpuParam MakeNpuParam(const int64_t& v) { return {&v, NpuParamType::kInt64}; }
inline NpuParam MakeNpuParam(const double& v) { return {&v, NpuParamType::kDouble}; }
inline NpuParam MakeNpuParam(const c10::optional<at::Tensor>& v) { return {&v, NpuParamType::kOptionalTensor}; }
inline NpuParam MakeNpuParam(const c10::optional<at::IntArrayRef>& v) { return {&v, NpuParamType::kOptionalIntArrayRef}; }
inline NpuParam MakeNpuParam(const c10::optional<at::Scalar>& v) { return {&v, NpuParamType::kOptionalScalar}; }
inline NpuParam MakeNpuParam(const at::ScalarType& v) { return {&v, NpuParamType::kScalarType}; }
inline NpuParam MakeNpuParam(const c10::string_view& v) { return {&v, NpuParamType::kString}; }
inline NpuParam MakeNpuParam(const std::string& v) { return {&v, NpuParamType::kString}; }
inline NpuParam MakeNpuParam(const at::TensorList& v) { return {&v, NpuParamType::kTensorList}; }
inline NpuParam MakeNpuParam(const at::ArrayRef<at::Scalar>& v) { return {&v, NpuParamType::kScalarList}; }
inline NpuParam MakeNpuParam(const at::ArrayRef<bool>& v) { return {&v, NpuParamType::kBoolArray}; }
inline NpuParam MakeNpuParam(const at::ArrayRef<c10::SymInt>& v) { return {&v, NpuParamType::kSymIntArrayRef}; }
inline NpuParam MakeNpuParam(const c10::OptionalArrayRef<c10::SymInt>& v) { return {&v, NpuParamType::kOptionalSymIntArrayRef}; }
inline NpuParam MakeNpuParam(const c10::OptionalIntArrayRef& v) { return {&v, NpuParamType::kOptionalIntArrayRefV2}; }
inline NpuParam MakeNpuParam(char* v) { return {v, NpuParamType::kCharArray}; }
inline NpuParam MakeNpuParam(const char* v) { return {v, NpuParamType::kCharArray}; }

}

extern "C" {
    TORCH_NPU_API void ExecNpuCmdExtImpl(const char* api_name, const op_plugin::NpuParam* params, int nparams);
}

namespace op_plugin {

template<typename... Args>
inline void ExecNpuCmd(const char* api_name, const Args&... args) {
    NpuParam params[] = {MakeNpuParam(args)...};
    ExecNpuCmdExtImpl(api_name, params, sizeof...(Args));
}

}

#define EXEC_NPU_CMD_EXT(aclnn_api, ...)                           \
    do {                                                           \
        op_plugin::ExecNpuCmd(#aclnn_api, __VA_ARGS__);            \
    } while(false)

#endif
