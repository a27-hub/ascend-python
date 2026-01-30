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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// Tensor q, Tensor k, Tensor d_o, Tensor g, Tensor? upper_tri_matrix, Tensor? g_gamma, Tensor? A, Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size) -> (Tensor)
at::Tensor npu_chunk_bwd_dv_local(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &d_o,
    const at::Tensor &g,
    const c10::optional<at::Tensor> &upper_tri_matrix,
    const c10::optional<at::Tensor> &g_gamma,
    const c10::optional<at::Tensor> &A,
    const c10::optional<at::Tensor> &cu_seqlens,
    const c10::optional<at::Tensor> &chunk_indices,
    double scale, 
    int64_t chunk_size)
{
    at::Tensor dv = npu_preparation::apply_tensor_without_format(d_o.sizes(), d_o.options().dtype());
    EXEC_NPU_CMD(aclnnChunkBwdDvLocal,
        q, k, d_o, g, upper_tri_matrix, g_gamma, A, cu_seqlens, chunk_indices, scale, chunk_size, dv);
    return dv;
}

}  // namespace op_api
