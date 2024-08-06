// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_fake_ascend_quant(
    const at::Tensor &x,
    double scale,
    double offset)
{
    at::Tensor y = npu_preparation::apply_tensor(x, x.options().dtype(at::kChar));

    bool sqrt_mode = false;
    string round_mode = "Round";
    int64_t ge_dtype = 2;

    at_npu::native::OpCommand cmd;
    cmd.Name("AscendQuant")
        .Input(x, "x")
        .Output(y, "y")
        .Attr("scale", static_cast<float>(scale))
        .Attr("offset", static_cast<float>(offset))
        .Attr("sqrt_mode", sqrt_mode)
        .Attr("round_mode", round_mode)
        .Attr("dst_type", ge_dtype)
        .Run();
    return y;
}
} // namespace acl_op
