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
#include <typeinfo>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
static bool check_TimeOut_ElaticInfo_param(const c10::optional<at::Tensor> &time_out,
                                           const c10::optional<at::Tensor> &elastic_info)
{
    if (time_out.has_value()) {
        return true;
    }
    if (elastic_info.has_value()) {
        return true;
    }
    return false;
}
at::Tensor _npu_distribute_barrier(c10::string_view group, int64_t world_size, const c10::optional<at::Tensor> &x_ref,
    const c10::optional<at::Tensor> &time_out, const c10::optional<at::Tensor> &elastic_info)
{
    char *group_ptr = const_cast<char *>(group.data());
    const at::Tensor &ref = c10::value_or_else(x_ref, [] {
         return npu_preparation::apply_tensor_without_format({1},c10::Scalartype::Float); });
    if (check_aclnn_kernel_available("aclnnDistributeBarrierV2")) {
        std::cout<<"======================ref"<<typeid(ref).name()<<std::endl;
        std::cout<<"======================ref"<<time_out.has_value()<<std::endl;
        std::cout<<"======================ref"<<elastic_info.has_value()<<std::endl;
        std::cout<<"======================ref"<<x_ref.has_value()<<std::endl;
        EXEC_NPU_CMD(aclnnDistributeBarrierV2, ref, time_out, elastic_info, group_ptr, world_size);
    } else {
        TORCH_CHECK(!check_TimeOut_ElaticInfo_param(time_out, elastic_info),
                    "The aclnnDistributeBarrier do not support time_out and elastic_info",
                    OPS_ERROR(ErrCode::PARAM));
        EXEC_NPU_CMD(aclnnDistributeBarrier, ref, group_ptr, world_size);
    }
    return ref;
}
}