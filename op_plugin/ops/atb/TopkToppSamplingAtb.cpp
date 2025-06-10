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
#include <vector>

using namespace std;

namespace atb {

using TopkToppSamplingParam = atb::infer::TopkToppSamplingParam;
std::tuple<at::Tensor, at::Tensor> _npu_topk_topp_sampling(const at::Tensor &probs, const at::Tensor &k, const at::Tensor &p)
{
    const c10::OptionalDeviceGuard device_guard(device_of(probs));
    OpParamCache<TopkToppSamplingParam>& TopkToppSamplingParamCache = OpParamCache<TopkToppSamplingParam>::getInstance();
    TopkToppSamplingParam topktoppsamplingparam;
    topktoppsamplingparam.topkToppSamplingType = TopkToppSamplingParam::BATCH_TOPK_MULTINOMIAL_SAMPLING;
    std::vector<uint32_t> vec(probs.size(0), 0);
    topktoppsamplingparam.randSeeds = vec;
    at::Tensor sampled_indices = at::empty(k.sizes(), at::TensorOptions().dtype(k.scalar_type()).device(probs.device()));
    at::Tensor sampled_probs = at::empty(k.sizes(), at::TensorOptions().dtype(probs.scalar_type()).device(probs.device()));
    ParamSetter paramsetter;
    paramsetter.Input(probs)
                .Input(k)
                .Input(p)
                .Output(sampled_indices)
                .Output(sampled_probs);
    auto opTopkToppSampling = TopkToppSamplingParamCache.getOperation(topktoppsamplingparam, "TopkToppSamplingOperation");
    RunAtbCmd(opTopkToppSampling, paramsetter, "TopkToppSamplingOperation");
    std::tuple<at::Tensor, at::Tensor> out;
    out = std::make_tuple(sampled_indices, sampled_probs);
    return out;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_topk_topp_sampling(Tensor probs, Tensor k, Tensor p) -> (Tensor sampled_indices, Tensor sampled_probs)");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_topk_topp_sampling", TORCH_FN(atb::_npu_topk_topp_sampling));
}
}

} // namespace atb
