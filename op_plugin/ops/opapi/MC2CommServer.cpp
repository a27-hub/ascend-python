// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
typedef void HcclComm;
using CreateKfcCtxFunc = uint32_t ()(HcclComm, char *, uintptr_t *);

int64_t create_mc2_comm_context(const at::Tensor &self, int64_t comm, c10::string_view op_type) //当前方案 需要传tensor
{
    //check_params(comm, op_type);
    static const auto createKfcCtxFuncAddr = GetOpApiFuncAddr("CreateKfcCtx");
    TORCH_CHECK(createKfcCtxFuncAddr != nullptr, "CreateKfcCtx not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),
        " not found.", OPS_ERROR(ErrCode::PTR));

    CreateKfcCtxFunc ctxFunc = reinterpret_cast<CreateKfcCtxFunc>(createKfcCtxFuncAddr);
    uintptr_t ctx = 0UL;  
    char* op_type_ptr = const_cast<char*>(op_type.data());
    void* group = (void*) comm;                                
    auto api_ret = ctxFunc(group, op_type_ptr, &ctx); 
    NPU_CHECK_ERROR(api_ret, "call CreateKfcCtx fail");    
    return ctx;
}

void mc2_comm_server(const at::Tensor &comm_context)
{
    //check_params(comm_context);
    EXEC_NPU_CMD(aclnnMC2CommServer, comm_context);
}

at::Tensor all_gather_matmul_inner(const at::Tensor &self, const at::Tensor &comm_context, c10::string_view group, int64_t comm_turn, int64_t world_size)
{
    TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8 || world_size == 16 || world_size == 32,
        "world_size should be in [2, 4, 8, 16, 32], but the actual value is ", world_size, "." + OPS_ERROR(ErrCode::PARAM));
    auto result_dtype = self.scalar_type();
    c10::SmallVector<int64_t, op_infer::SIZE> out_gather_mm_size = {100, 100};
    auto out_gather_mm = at_npu::native::OpPreparation::apply_tensor_without_format(out_gather_mm_size, self.options().dtype(result_dtype));
    EXEC_NPU_CMD(aclnnAllGatherMatmulInner, self, comm_context, group, comm_turn, world_size, out_gather_mm);
    return out_gather_mm;
}
}
