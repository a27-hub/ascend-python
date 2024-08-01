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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_instance_norm_v3(const at::Tensor &x, const at::Tensor &gamma, const at::Tensor &beta, c10::string_view data_format, double epsilon)
{
    DO_COMPATIBILITY(aclnnInstanceNormV3, acl_op::npu_instance_norm_v3(x, gamma, beta, data_format, epsilon));

    at::SmallVector<int64_t, SIZE> shape;
    string format = std::string(data_format);
    if (format == "NHWC") {
        for (int64_t index = 0; index < x.dim(); index++) {
            if (index == 0 || index == x.dim() - 1) {
                shape.emplace_back(x.size(index));
            } else {
                shape.emplace_back(1);
            }
        }
    } else if (format == "NCHW") {
        for (int64_t index = 0; index < x.dim(); index++) {
            if (index == 0 || index == 1) {
                shape.emplace_back(x.size(index));
            } else {
                shape.emplace_back(1);
            }
        }
    } else {
        for (int64_t index = 0; index < x.dim(); index++) {
            shape.emplace_back(x.size(index));
        }
    }

    at::Tensor y = at_npu::native::OpPreparation::apply_tensor(x);
    at::Tensor mean = at_npu::native::OpPreparation::apply_tensor(x, shape);
    at::Tensor variance = at_npu::native::OpPreparation::apply_tensor(x, shape);

    const char* format_chars = format.c_str();
    EXEC_NPU_CMD(aclnnInstanceNormV3, x, gamma, beta, format_chars, epsilon, y, mean, variance);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, mean, variance);
}
} // namespace op_api