// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/AtbCommon.h"
#include <acl/acl.h>

using namespace std;

namespace atb {

using GroupTopkParam = atb::infer::GroupTopkParam;
at::Tensor& npu_group_topk_out(const at::Tensor &self, int64_t k, int64_t group_num, int64_t n, at::Tensor &out)
{
    OpParamCache<GroupTopkParam>& GroupTopkParamCache = OpParamCache<GroupTopkParam>::getInstance();
    GroupTopkParam GroupTopkParam;
    GroupTopkParam.groupNum = static_cast<int32_t>(group_num);
    GroupTopkParam.k = static_cast<int32_t>(k);
    GroupTopkParam.n = static_cast<uint16_t>(n);
    GroupTopkParam.groupMultiFlag = static_cast<GroupTopkParam::GroupMultiFlag>(0);
    if (n > 1) {
        GroupTopkParam.groupMultiFlag = static_cast<GroupTopkParam::GroupMultiFlag>(1);
    }

    out = self;
    auto idx = at::arange(1024, self.options().device(at::Device(at::kPrivateUse1)).dtype(at::kInt));

    ParamSetter paramsetter;
    paramsetter.Input(self)
                .Input(idx)
                .Output(out);
    auto opGroupTopk = GroupTopkParamCache.getOperation(GroupTopkParam, "GroupTopkOperation");
    RunAtbCmd(opGroupTopk, paramsetter, "GroupTopkOperation");
    return out;
}

namespace {
TORCH_LIBRARY(atb, m)
{
    m.def("npu_group_topk.out(Tensor self, int k=0, int group_num=1, int n=1, *, Tensor(a!) out) -> Tensor(a!)");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("npu_group_topk.out", TORCH_FN(atb::npu_group_topk_out));
}
}

} // namespace atb