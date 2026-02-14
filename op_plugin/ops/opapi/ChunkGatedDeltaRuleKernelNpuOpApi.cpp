// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <tuple>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
    constexpr int VALUE_DIM_NUM = 3;
    constexpr int OUT_DIM_NUM = 3;

    using npu_preparation = at_npu::native::OpPreparation;

    std::tuple<at::Tensor, at::Tensor> npu_chunk_gated_delta_rule(
        const at::Tensor &query,
        const at::Tensor &key,
        const at::Tensor &value,
        const c10::optional<at::Tensor> &beta,
        const c10::optional<at::Tensor> &initial_state,
        const c10::optional<at::Tensor> &actual_seq_lengths,
        const c10::optional<double> scale,
        const c10::optional<at::Tensor> &g)
    {
        TORCH_CHECK(value.dim() == VALUE_DIM_NUM, "value dim should be ", VALUE_DIM_NUM, OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(initial_state.has_value(), "initial state cannot be empty", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(scale.has_value(), "scale cannot be empty", OPS_ERROR(ErrCode::PARAM));

        auto t_dim = value.size(0);
        auto nv_dim = value.size(1);
        auto dv_dim = value.size(2);
        c10::SmallVector<int64_t, OUT_DIM_NUM> out;
        out.push_back(t_dim);
        out.push_back(nv_dim);
        out.push_back(dv_dim);
        c10::TensorOptions options = value.options().dtype(at::ScalarType::BFloat16);
        at::Tensor outResult = npu_preparation::apply_tensor_without_format(out, options);

        at::Tensor finalStateResult = initial_state.value().clone();

        float scale_real = static_cast<float>(scale.value());

        EXEC_NPU_CMD(aclnnChunkGatedDeltaRule,
            query, key, value, beta, initial_state, actual_seq_lengths, g, scale_real, outResult, finalStateResult);

        return std::tuple<at::Tensor, at::Tensor>(outResult, finalStateResult);
    }
}  // namespace op_api
