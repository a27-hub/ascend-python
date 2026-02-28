// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include <cstring>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;
const int64_t MAX_HEAD_DIM = 128;
using npu_preparation = at_npu::native::OpPreparation;


// 与 aclnnBlockSparseAttentionGrad 文档一致的最小约束检查，其余透传 aclnn 便于版本兼容。
static void check_params(const at::Tensor &query,
                         const at::Tensor &key,
                         const at::Tensor &value,
                         const c10::OptionalIntArrayRef block_shape,
                         const c10::OptionalIntArrayRef actual_seq_lengths,
                         const c10::OptionalIntArrayRef actual_seq_lengths_kv,
                         c10::string_view q_input_layout,
                         c10::string_view kv_input_layout,
                         int64_t num_key_value_heads)
{
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "query should be 3 or 4 dimensional, but got ", query.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "key should be 3 or 4 dimensional, but got ", key.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "value should be 3 or 4 dimensional, but got ", value.dim(), OPS_ERROR(ErrCode::PARAM));

    int64_t head_dim = query.size(-1);
    TORCH_CHECK(head_dim <= MAX_HEAD_DIM,
        "head_dim must be <= ", MAX_HEAD_DIM, ", but got ", head_dim, OPS_ERROR(ErrCode::PARAM));
}

// aclnn 要求输入为 ND 格式。先 contiguous 再 npu_format_cast，若仍非 ND 则分配 ND 张量并 copy_（与正向一致）
static at::Tensor format_to_nd(const at::Tensor &t)
{
    if (!t.defined() || !torch_npu::utils::is_npu(t)) {
        return t;
    }
    at::Tensor c = t.contiguous();
    at::Tensor nd = custom_ops::npu_format_cast(c, ACL_FORMAT_ND);
    if (npu_preparation::get_tensor_npu_format(nd) != ACL_FORMAT_ND) {
        at::Tensor nd_tensor = npu_preparation::apply_tensor_with_format(t.sizes(), t.options(), ACL_FORMAT_ND);
        nd_tensor.copy_(t);
        return nd_tensor;
    }
    return nd;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_block_sparse_attention_backward(
    const at::Tensor &d_out,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &attention_out,
    const at::Tensor &softmax_lse,
    const at::Tensor &block_sparse_mask,
    const c10::OptionalIntArrayRef block_shape,
    const c10::OptionalIntArrayRef actual_seq_lengths,
    const c10::OptionalIntArrayRef actual_seq_lengths_kv,
    c10::string_view q_input_layout,
    c10::string_view kv_input_layout,
    int64_t num_key_value_heads,
    double scale_value)
{
    check_params(query, key, value, block_shape, actual_seq_lengths, actual_seq_lengths_kv,
                 q_input_layout, kv_input_layout, num_key_value_heads);

    at::Tensor d_query = npu_preparation::apply_tensor_without_format(query);
    at::Tensor d_key = npu_preparation::apply_tensor_without_format(key);
    at::Tensor d_value = npu_preparation::apply_tensor_without_format(value);

    // 获取参数默认值：aclnnBlockSparseAttentionGrad 要求 blockShape 非空，未传时使用默认 [128, 128] 与 aclnn 文档一致
    static const int64_t kDefaultBlockShape[2] = {128, 128};
    const at::IntArrayRef block_shape_value = (block_shape.has_value() && block_shape->size() >= 2)
        ? *block_shape
        : at::IntArrayRef(kDefaultBlockShape, 2);
    const at::IntArrayRef actual_seq_lengths_value = actual_seq_lengths.value_or(at::IntArrayRef{});
    const at::IntArrayRef actual_seq_lengths_kv_value = actual_seq_lengths_kv.value_or(at::IntArrayRef{});

    // 初始化aclnn中的暂不支持参数
    const at::Tensor atten_mask = at::Tensor();
    const int64_t mask_type = 0;
    const int64_t pre_tokens = 2147483647;
    const int64_t next_tokens = 2147483647;

    // 获取到layout的指针，直接传给aclnn接口，供其获取字符串
    char *q_input_layout_ptr = const_cast<char *>(q_input_layout.data());
    char *kv_input_layout_ptr = const_cast<char *>(kv_input_layout.data());

    // aclnn 要求输入为 ND 格式；若为 NCL/NCHW 等则在此统一转为 ND
    at::Tensor d_out_nd = format_to_nd(d_out);
    at::Tensor query_nd = format_to_nd(query);
    at::Tensor key_nd = format_to_nd(key);
    at::Tensor value_nd = format_to_nd(value);
    at::Tensor attention_out_nd = format_to_nd(attention_out);
    at::Tensor softmax_lse_nd = format_to_nd(softmax_lse);
    at::Tensor block_sparse_mask_nd = format_to_nd(block_sparse_mask);

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnBlockSparseAttentionGrad, d_out_nd, query_nd, key_nd, value_nd, attention_out_nd,
        softmax_lse_nd, block_sparse_mask_nd, atten_mask, block_shape_value,
        actual_seq_lengths_value, actual_seq_lengths_kv_value,
        q_input_layout_ptr, kv_input_layout_ptr,
        num_key_value_heads, mask_type, scale_value, pre_tokens, next_tokens,
        d_query, d_key, d_value);

    return std::make_tuple(d_query, d_key, d_value);
}
}
