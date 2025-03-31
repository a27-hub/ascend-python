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

namespace op_api
{
    using npu_preparation = at_npu::native::OpPreparation;

    std::tuple<at::Tensor, at::Tensor> npu_kv_rmsnorm_rope_cache(
        const at::Tensor &kv,
        const at::Tensor &gamma,
        const at::Tensor &cos,
        const at::Tensor &sin,
        const at::Tensor &index,
        const at::Tensor &k_cache_ref,
        const at::Tensor &v_cache_ref,
        double epsilon)
    {
        TORCH_CHECK((k_cache_ref.dim() == 4), "4D tensor expected for input k_cache" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((v_cache_ref.dim() == 4), "4D tensor expected for input v_cache" + OPS_ERROR(ErrCode::PARAM));

        EXEC_NPU_CMD(aclnnKvRmsNormRopeCache, kv, gamma, cos, sin, index, k_cache_ref,
                     v_cache_ref, epsilon);
        return std::tie(k_cache_ref, v_cache_ref);
    }
} // namespace op_api
