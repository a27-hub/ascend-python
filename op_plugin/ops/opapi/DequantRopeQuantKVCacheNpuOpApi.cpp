// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api
{
    using npu_preparation = at_npu::native::OpPreparation;

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> dequant_rope_quant_kvcache(
        const at::Tensor &x,
        const at::Tensor &cos,
        const at::Tensor &sin,
        at::Tensor &k_cache_ref,
        at::Tensor &v_cache_ref,
        const at::Tensor &indices,
        const at::Tensor &scale_k_optional,
        const at::Tensor &scale_v_optional,
        const at::Tensor &offset_k_optional,
        const at::Tensor &offset_v_optional,
        const at::Tensor &weight_scale_optional,
        const at::Tensor &activation_scale_optional,
        const at::Tensor &bias_optional,
        const std::vector &size_splits,
        const std::string quant_mode,
        const std::string input_layout,
        const bool kv_output)
    {
        const int64_t b = x.size(0);
        const int64_t n = x.size(1);
        const int64_t d = cos.size(3);
        c10::SmallVector<int64_t, SIZE> q_shape = {b, n, size_splits[0], d};
        c10::SmallVector<int64_t, SIZE> k_shape = {b, n, size_splits[1], d};
        c10::SmallVector<int64_t, SIZE> v_shape = {b, n, size_splits[2], d};
        at::Tensor q_output = npu_preparation::apply_tensor_without_format(q_shape);
        at::Tensor k_output = npu_preparation::apply_tensor_without_format(k_shape);
        at::Tensor v_output = npu_preparation::apply_tensor_without_format(v_shape);

        EXEC_NPU_CMD(aclnnDequantRopeQuantKVCache, x, cos, sin, k_cache_ref, v_cache_ref, indices, scale_k_optional, scale_v_optional, offset_k_optional, offset_v_optional, weight_scale_optional, activation_scale_optional, bias_optional);
        return std::tie(q_output, k_output, v_output, k_cache_ref, v_cache_ref);
    }
} // namespace op_api