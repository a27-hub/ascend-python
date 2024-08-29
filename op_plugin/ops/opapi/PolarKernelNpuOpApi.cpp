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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &polar_out(const at::Tensor &abs, const at::Tensor &angle, at::Tensor &result)
{
    auto outputSize = op_infer::broadcast_ops_npu_output_size(abs, angle);
    npu_preparation::check_tensor({abs}, result, result.scalar_type(), outputSize);
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnPolar, abs, angle, result);
    return result;
}

at::Tensor polar(const at::Tensor &abs, const at::Tensor &angle)
{
    at::ScalarType high_type = at::native::result_type(abs, angle);
    if (high_type == at::ScalarType::Float) {
        high_type = at::ScalarType::ComplexFloat;
    } else if (high_type == at::ScalarType::Double) {
        high_type = at::ScalarType::ComplexDouble;
    } else if (high_type == at::ScalarType::Half) {
        high_type = at::ScalarType::ComplexHalf;
    }
    auto outputSize = op_infer::broadcast_ops_npu_output_size(abs, angle);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, abs.options().dtype(high_type));
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnPolar, abs, angle, result);
    return result;
}
} // namespace op_api
