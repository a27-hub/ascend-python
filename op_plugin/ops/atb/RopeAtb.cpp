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
#include "op_plugin/utils/custom_functions/atb/atb_common.h"
#include "op_plugin/utils/custom_functions/atb/op_param_cache.h"
#include <acl/acl.h>

using namespace std;
namespace op_api {
using RopeParam = atb::infer::RopeParam;
void npu_rope(at::Tensor &query, at::Tensor &key, const at::Tensor &cos, const at::Tensor &sin, const at::Tensor &seqlen) 
{
    OpParamCache<RopeParam>& ropeParamCache = OpParamCache<RopeParam>::getInstance();
    RopeParam ropeparam;
    
    ropeparam.rotaryCoeff = 4;
    ropeparam.cosFormat = 0;
    ParamSetter parametter;
    parametter.Input(query)
                .Input(key)
                .Input(cos)
                .Input(sin)
                .Input(seqlen)
                .Output(query)
                .Output(key);
    auto opRope = ropeParamCache.getOperation(ropeparam, "RopeOperation");
    RunAtbCmd(opRope, parametter, "RopeOperation");
    return;
}
} // namespace