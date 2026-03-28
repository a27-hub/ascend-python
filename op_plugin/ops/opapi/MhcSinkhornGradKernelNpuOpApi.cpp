// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
using npu_utils = at_npu::native::NpuUtils;

at::Tensor npu_mhc_sinkhorn_grad(const at::Tensor &x, const at::Tensor &norm_out, const at::Tensor &sum_out)
{

    at::Tensor result = npu_preparation::apply_tensor_with_format(x.sizes(), x.options(), ACL_FORMAT_ND);

    EXEC_NPU_CMD(aclnnMhcSinkhornGrad, x, norm_out, sum_out, result);

    return result;
}

}  // namespace op_api