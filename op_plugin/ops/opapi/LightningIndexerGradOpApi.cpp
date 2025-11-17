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

#include <cstring>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;
const int LAYOUT_MAX_LENGTH = 20;

namespace {
at::Tensor format_trans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
            "Expected all tensors to be on the same device. "
            "Expected NPU tensor, please check whether the input tensor device is correct.",
            OPS_ERROR(ErrCode::TYPE));
        return custom_ops::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_lightning_indexer_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &dy,
    const at::Tensor &sparse_indices,
    const at::Tensor &weights,
    int64_t head_num,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    c10::optional<c10::string_view> layout,
    c10::optional<int64_t> sparse_mode,
    int64_t pre_tockens, 
    int64_t next_tockens,
    c10::optional<bool> deterministic)
{
    at::Tensor format_query = format_trans(query);
    at::Tensor format_key = format_trans(key);
    at::Tensor format_dy = format_trans(dy);
    at::Tensor format_sparse_indices = format_trans(sparse_indices);
    at::Tensor format_weights = format_trans(weights);
    const at::Tensor &actual_seq_lengths_query_const = actual_seq_lengths_query.value_or(at::Tensor());
    const at::Tensor &actual_seq_lengths_key_const = actual_seq_lengths_key.value_or(at::Tensor());

    at::Tensor d_query = OpPreparation::apply_tensor_without_format(format_query);
    at::Tensor d_key = OpPreparation::apply_tensor_without_format(format_key);
    at::Tensor d_weights = OpPreparation::apply_tensor_without_format(format_weights);


    c10::string_view layout_str_view = layout.value_or("TND");
    char *layout_ptr = const_cast<char *>(layout_str_view.data());
    const int64_t sparse_mode_const = sparse_mode.value_or(0);
    const bool deterministic_const = deterministic.value_or(false);

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnLightningIndexerGrad, format_query, format_key, format_dy, format_sparse_indices, format_weights,
        actual_seq_lengths_query_const, actual_seq_lengths_key_const,
        head_num, layout_ptr, sparse_mode_const, pre_tockens, next_tockens, deterministic_const,
        d_query, d_key, d_weights);

    return std::make_tuple(d_query, d_key, d_weights);
}
}