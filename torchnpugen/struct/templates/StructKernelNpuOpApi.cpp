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

{% if env_aclnn_extension_switch %}
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include "op_plugin/include/npu_cpp_extension.h"
{% else %}
#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
{% endif %}

namespace op_api {
{% if env_aclnn_extension_switch %}
using npu_preparation = at_npu::native::OpPreparation;
using namespace op_plugin::utils;
using namespace op_infer;
{% else %}
using npu_preparation = at_npu::native::OpPreparation;
using namespace op_plugin::utils;
using namespace op_infer;
{% endif %}

${op_api_definition}

}  // namespace op_api
