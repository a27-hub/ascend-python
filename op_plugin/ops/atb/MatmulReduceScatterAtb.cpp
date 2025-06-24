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
using LinearParallelParam = atb::infer::LinearParallelParam;
void _npu_matmul_reduce_scatter(const at::Tensor &input1, const at::Tensor &input2, at::Tensor &output, const c10::optional<at::Tensor> &biasOpt,
                                const c10::optional<at::Tensor> &deqScale, int64_t rank, int64_t rankSize, const std::string &commDomain, int64_t outdata_type)
{
    const c10::OptionalDeviceGuard device_guard(device_of(input1));
    OpParamCache<LinearParallelParam>& LinearParallelParamCache = OpParamCache<LinearParallelParam>::getInstance();
    LinearParallelParam param;
    const at::Tensor &bias = biasOpt.value_or(at::Tensor());
    const at::Tensor &deqscale = deqScale.value_or(at::Tensor());

    bool transWeight = input1.size(1) != input2.size(0);
    param.transWeight = transWeight;
    param.rank = rank;
    param.rankSize = rankSize;
    param.rankRoot = 0;
    // param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER;
    param.keepIntermediate = false;
    param.commDomain = commDomain;
    param.quantType = atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_CHANNEL;
    
    auto outdatatype = static_cast<aclDataType>(outdata_type);
    param.outDataType = outdatatype;

    ParamSetter paramsetter;
    paramsetter.Input(input1)
               .Input(input2);
    if (biasOpt.has_value()) {
        paramsetter.Input(bias);
    }
    if (deqScale.has_value()) {
        paramsetter.Input(deqscale);
    }
    paramsetter.Output(output);

    auto ops = LinearParallelParamCache.getOperation(param, "LcalMatmulReduceScatterOperation");
    RunAtbCmd(ops, paramsetter, "LcalMatmulReduceScatterOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_matmul_reduce_scatter(Tensor input1, Tensor input2, Tensor(a!) output, Tensor? biasOpt=None, Tensor? deqScale=None,int rank=0, int rankSize=1, str commDomain='0', int outdata_type=0) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_matmul_reduce_scatter", TORCH_FN(atb::_npu_matmul_reduce_scatter));
}
}

}