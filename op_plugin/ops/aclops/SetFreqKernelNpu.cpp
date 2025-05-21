// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

void npu_set_freq_check(const at::Tensor &freq)
{
    TORCH_CHECK(freq.numel() == 1, "freq must be a 1-element tensor" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(freq.scalar_type() == at::ScalarType::Int, "freq must be an int tensor" + OPS_ERROR(ErrCode::TYPE));
}

at::Tensor npu_set_freq(const at::Tensor &freq)
{
    npu_set_freq_check(freq);
    at::Tensor result = at::empty({1}, freq.options());
    int64_t device_id = freq.get_device();

    at_npu::native::OpCommand cmd;
    cmd.Name("SetFreq")
        .Input(freq)
        .Output(result)
        .Attr("device", (int64_t)device_id)
        .Run();
    return result;
}

} // namespace acl_op
