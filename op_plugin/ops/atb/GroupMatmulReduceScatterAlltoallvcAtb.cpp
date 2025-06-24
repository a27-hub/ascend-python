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
void _npu_gmm_reduce_scatter_alltoallv(const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &globalTokensPerExpertMatrix, const at::Tensor &maxOutputSize, at::Tensor &output,
    const c10::optional<at::Tensor> &deqScaleOpt, const c10::optional<at::Tensor> &dequantPerTokenScaleOpt,
    bool transWeight, int64_t rank, int64_t rankSize, const std::string &commDomain,
    int64_t quantType, int64_t outDataType, int64_t localExpertNums, int64_t epSize, int64_t tpSize)
{
    const c10::OptionalDeviceGuard device_guard(device_of(input));
    OpParamCache<LinearParallelParam>& LinearParallelParamCache = OpParamCache<LinearParallelParam>::getInstance();
    LinearParallelParam param;
    const at::Tensor &deqScale = deqScaleOpt.value_or(at::Tensor());
    const at::Tensor &dequantPerTokenScale = dequantPerTokenScaleOpt.value_or(at::Tensor());

    param.transWeight = transWeight;
    param.rank = rank;
    param.rankSize = rankSize;
    param.rankRoot = 0;
    param.hasResidual = false;
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC;
    param.commDomain = commDomain;
    param.quantType = static_cast<LinearParallelParam::QuantType>(quantType);
    param.outDataType = static_cast<aclDataType>(outDataType);
    param.moeInfo.localExpertNums = localExpertNums;
    param.moeInfo.epSize = epSize;
    param.moeInfo.tpSize = tpSize;

    ParamSetter paramsetter;
    paramsetter.Input(input)
            .Input(weight);
    if (deqScaleOpt.has_value()) {
        paramsetter.Input(deqScale);
    }
    if (dequantPerTokenScaleOpt.has_value()) {
        paramsetter.Input(dequantPerTokenScale);
    }
    paramsetter.Input(globalTokensPerExpertMatrix)
            .Input(maxOutputSize)
            .Output(output);
    auto ops = LinearParallelParamCache.getOperation(param, "LcalMatmulReduceScatterAllToAllVHiddenOperation");
    RunAtbCmd(ops, paramsetter, "LcalMatmulReduceScatterAllToAllVHiddenOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_gmm_reduce_scatter_alltoallv(Tensor input, Tensor weight, Tensor globalTokensPerExpertMatrix, Tensor maxOutputSize, Tensor(a!) output, Tensor? deqScaleOpt=None, Tensor? dequantPerTokenScaleOpt=None, bool transWeight=False, int rank=0, int rankSize=1, str commDomain='0', int quantType=-1, int outDataType=-1, int localExpertNums=1, int epSize=1, int tpSize=1) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_gmm_reduce_scatter_alltoallv", TORCH_FN(atb::_npu_gmm_reduce_scatter_alltoallv));
}
}

}