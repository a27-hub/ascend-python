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
#include <string>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;
using npu_preparation = at_npu::native::OpPreparation;


// 与 aclnnBlockSparseAttention 文档一致的最小约束检查，其余透传 aclnn 便于版本兼容。
static void check_params(const at::Tensor &query,
                         const at::Tensor &key,
                         const at::Tensor &value,
                         const at::Tensor &block_sparse_mask,
                         const c10::OptionalIntArrayRef block_shape,
                         const c10::OptionalIntArrayRef actual_seq_lengths,
                         const c10::OptionalIntArrayRef actual_seq_lengths_kv,
                         c10::string_view q_input_layout,
                         c10::string_view kv_input_layout,
                         int64_t num_key_value_heads,
                         int64_t inner_precise)
{
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "query should be 3 or 4 dimensional, but got ", query.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "key should be 3 or 4 dimensional, but got ", key.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "value should be 3 or 4 dimensional, but got ", value.dim(), OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(inner_precise == 0 || inner_precise == 1,
        "inner_precise must be 0 or 1, got ", inner_precise, OPS_ERROR(ErrCode::PARAM));
}

// aclnn 要求 query/key/value/block_sparse_mask 为 ND 格式。npu_format_cast 对 NCL 等有时不生效，故先 contiguous 再 cast，
// 若仍非 ND 则分配 ND 张量并 copy_ 以保证传入 aclnn 的为 ND。
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

std::tuple<at::Tensor, at::Tensor> npu_block_sparse_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &block_sparse_mask,
    c10::string_view q_input_layout,
    c10::string_view kv_input_layout,
    int64_t num_key_value_heads,
    double scale_value,
    int64_t inner_precise,
    const c10::OptionalIntArrayRef block_shape,
    const c10::OptionalIntArrayRef actual_seq_lengths,
    const c10::OptionalIntArrayRef actual_seq_lengths_kv,
    c10::optional<int64_t> softmax_lse_flag)
{
    check_params(query, key, value, block_sparse_mask, block_shape, actual_seq_lengths,
                 actual_seq_lengths_kv, q_input_layout, kv_input_layout, num_key_value_heads, inner_precise);

    // 分配输出 Tensor
    auto attention_out_shape = query.sizes().vec();
    attention_out_shape[attention_out_shape.size() - 1] = value.size(-1);
    at::Tensor attention_out = npu_preparation::apply_tensor_without_format(attention_out_shape, query.options());

    at::Tensor softmax_lse_out;
    if (query.dim() == DIMENSION_4D) {
        softmax_lse_out = npu_preparation::apply_tensor_without_format(
            {query.size(0), query.size(1), query.size(2), 1}, query.options().dtype(at::kFloat));
    } else {
        softmax_lse_out = npu_preparation::apply_tensor_without_format(
            {query.size(0), query.size(1), 1}, query.options().dtype(at::kFloat));
    }

    // 获取参数默认值：aclnnBlockSparseAttention 要求 blockShapeOptional 至少两元素，未传时使用默认 [128, 128] 与 aclnn 文档一致
    static const int64_t kDefaultBlockShape[2] = {128, 128};
    const at::IntArrayRef block_shape_value = (block_shape.has_value() && block_shape->size() >= 2)
        ? *block_shape
        : at::IntArrayRef(kDefaultBlockShape, 2);
    const at::IntArrayRef actual_seq_lengths_q_value = actual_seq_lengths.value_or(at::IntArrayRef{});
    const at::IntArrayRef actual_seq_lengths_kv_value = actual_seq_lengths_kv.value_or(at::IntArrayRef{});
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

    // aclnn 要求输入为 ND 格式；若为 NCL/NCHW 等则在此统一转为 ND
    at::Tensor query_nd = format_to_nd(query);
    at::Tensor key_nd = format_to_nd(key);
    at::Tensor value_nd = format_to_nd(value);
    at::Tensor block_sparse_mask_nd = format_to_nd(block_sparse_mask);

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnBlockSparseAttention, query_nd, key_nd, value_nd, block_sparse_mask_nd, atten_mask,
        block_shape_value, actual_seq_lengths_q_value, actual_seq_lengths_kv_value, block_table,
        q_input_layout_ptr, kv_input_layout_ptr, num_key_value_heads, mask_type, scale_value,
        inner_precise, block_size, pre_tokens, next_tokens, softmax_lse_flag_value,
        attention_out, softmax_lse_out);

    return std::make_tuple(attention_out, softmax_lse_out);
}
}
