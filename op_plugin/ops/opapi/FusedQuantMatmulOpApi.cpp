// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include <vector>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {

at::Tensor npu_fused_quant_matmul(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &scale,
                            const c10::optional<at::Tensor> &offset, const c10::optional<at::Tensor> &pertoken_scale,
                            const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &x3,
                            c10::string_view fused_op_type, c10::optional<int64_t> output_dtype,
                            c10::optional<int64_t> x1_dtype, c10::optional<int64_t> x2_dtype,
                            c10::optional<int64_t> pertoken_scale_dtype, c10::optional<int64_t> scale_dtype,
                            c10::optional<int64_t> x3_dtype, c10::OptionalIntArrayRef group_sizes,
                            const c10::optional<at::Tensor> &y_scale)
{
    at::Tensor result = npu_preparation::apply_tensor(x1);
}
}