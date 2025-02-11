// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#ifndef OPPLUGIN_UTILS_ATB_COMMON_H
#define OPPLUGIN_UTILS_ATB_COMMON_H
#include <torch_npu/csrc/framework/OpCommand.h>
#include "op_plugin/third_party/atb/inc/atb_infer.h"
#include "op_plugin/utils/op_api_common.h"


atb::Tensor AtTensor2Tensor(const at::Tensor atTensor);
atb::Context* GetContext();
at::Tensor GetWorkspaceTensor(uint64_t workspaceSize);
uint64_t OperationSetup(atb::VariantPack variantPack, atb::Operation *operation, atb::Context* contextPtr);
class ParamSetter {
public:
    ParamSetter& Input(const at::Tensor &tensor);
    ParamSetter& Input(const c10::optional<at::Tensor> &tensor);
    ParamSetter& Output(at::Tensor &tensor);
    atb::VariantPack variantPack;
};

void RunAtbCmd(atb::Operation *op, const ParamSetter &paramsetter, const std::string &name);

#endif