// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rope_qk_rms_norm_k_store_kv_cache(
    const at::Tensor &qkv,
    const at::Tensor &sin,
    const at::Tensor &cos,
    const at::Tensor &gamma,
    const at::Tensor &k_scale,
    const at::Tensor &v_scale,
    const at::Tensor &index,
    const at::Tensor &k_cache,
    const at::Tensor &v_cache,
    double epsilon,
    int64_t q_head_num,
    int64_t kv_head_num)
{
    TORCH_CHECK((qkv.dim() == 3), "3D tensor expected for input kv" + OPS_ERROR(ErrCode::PARAM));
    
    int64_t hidden_dim = qkv.size(2);
    int64_t head_dim = hidden_dim / (q_head_num + kv_head_num + kv_head_num);
    c10::SmallVector<int64_t, SIZE> q_shape = {qkv.size(0), qkv.size(1), q_head_num * head_dim};
    at::Tensor q = npu_preparation::apply_tensor_without_format(q_shape, qkv.options());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnRopeQkRmsNormKStoreKvCache, qkv, sin, cos, gamma, index, k_scale, v_scale,
                                 k_cache, v_cache);
    return std::tie(q, k_cache, v_cache);
}
}
