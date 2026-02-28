// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;


std::tuple<at::Tensor, at::Tensor> npu_block_sparse_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &block_sparse_mask,
    const c10::optional<at::IntArrayRef> &block_shape,
    const c10::optional<at::IntArrayRef> &actual_seq_lengths,
    const c10::optional<at::IntArrayRef> &actual_seq_lengths_kv,
    c10::string_view q_input_layout,
    c10::string_view kv_input_layout,
    int64_t num_key_value_heads,
    double scale_value,
    int64_t inner_precise,
    c10::optional<int64_t> softmax_lse_flag)
{
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "The shapes of the input value should be 3 or 4 dimensional, but got ",
        value.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));

    // 分配输出Tensor的数据类型
    at::Tensor attention_out = OpPreparation::apply_tensor_without_format(query);
    at::Tensor softmax_lse_out = OpPreparation::apply_tensor_without_format(query);

    // 获取参数默认值
    const at::IntArrayRef block_shape_value = block_shape.value_or(at::IntArrayRef());
    const at::IntArrayRef actual_seq_lengths_q_value = actual_seq_lengths.value_or(at::IntArrayRef());
    const at::IntArrayRef actual_seq_lengths_kv_value = actual_seq_lengths_kv.value_or(at::IntArrayRef());
    const int64_t softmax_lse_flag_value = softmax_lse_flag.value_or(0);

    // 初始化aclnn中的暂不支持参数
    const at::Tensor atten_mask = at::Tensor();
    const at::Tensor block_table = at::Tensor();
    const int64_t mask_type = 0;
    const int64_t block_size = 0;
    const int64_t pre_tokens = 2147483647;
    const int64_t next_tokens = 2147483647;

    // 获取到layout的指针，直接传给aclnn接口，供其获取字符串
    char *q_input_layout_ptr = const_cast<char *>(q_input_layout.data());
    char *kv_input_layout_ptr = const_cast<char *>(kv_input_layout.data());

    // 调用 aclnn 算子
    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnBlockSparseAttention, query, key, value, block_sparse_mask, atten_mask,
        block_shape_value, actual_seq_lengths_q_value, actual_seq_lengths_kv_value, block_table,
        q_input_layout_ptr, kv_input_layout_ptr, num_key_value_heads, mask_type, scale_value,
        inner_precise, block_size, pre_tokens, next_tokens, softmax_lse_flag_value,
        attention_out, softmax_lse_out);

    return std::make_tuple(attention_out, softmax_lse_out);
}
}