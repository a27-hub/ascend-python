// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
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

at::Tensor npu_mhc_post(const at::Tensor &x, const at::Tensor &h_res, const at::Tensor &h_out, const at::Tensor &h_post)
{
    at::Tensor y = npu_preparation::apply_tensor_without_format(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnMhcPost, x, h_res, h_out, h_post, y);
    return y;
}

}