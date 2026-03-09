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

#include <torch/csrc/autograd/custom_function.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_mhc_sinkhorn(const at::Tensor& h_res, int64_t out_flag, float eps, int64_t num_iters)
{
    at::Tensor result = npu_preparation::apply_tensor_without_format(h_res);
    at::Tensor norm_out = npu_preparation::apply_tensor_without_format(h_res);
    at::Tensor sum_out = npu_preparation::apply_tensor_without_format(h_res);
    EXEC_NPU_CMD(aclnnMhcSinkhorn, h_res, out_flag, eps, num_iters, result, norm_out, sum_out);
    return result, norm_out, sum_out;
}
} // namespace op_api