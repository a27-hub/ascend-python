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
#include <chrono>
#include <acl/acl.h>
using namespace std;
namespace op_api {
using SelfAttentionParam = atb::infer::SelfAttentionParam;
void npu_selfattention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &mask, const at::Tensor &seqLen, const double scale, const int64_t headNum, const int64_t kvHeadNum,
    at::Tensor & out) 
{
    OpParamCache<SelfAttentionParam>& selfAttentionParamCache = OpParamCache<SelfAttentionParam>::getInstance();
    SelfAttentionParam selfattentionparam;
    

	selfattentionparam.calcType = atb::infer::SelfAttentionParam::PA_ENCODER;
	selfattentionparam.kernelType = atb::infer::SelfAttentionParam::KERNELTYPE_DEFAULT;
	selfattentionparam.clampType = atb::infer::SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
	selfattentionparam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
	selfattentionparam.kvcacheCfg = atb::infer::SelfAttentionParam::K_CACHE_V_CACHE;
	selfattentionparam.scaleType = atb::infer::SelfAttentionParam::SCALE_TYPE_TOR;
	selfattentionparam.quantType = atb::infer::SelfAttentionParam::TYPE_QUANT_UNDEFINED;
	selfattentionparam.cacheType = atb::infer::SelfAttentionParam::CACHE_TYPE_NORM;
	selfattentionparam.outDataType = ACL_DT_UNDEFINED;
	selfattentionparam.headNum = headNum;
	selfattentionparam.kvHeadNum = kvHeadNum;
	selfattentionparam.qScale = 1;
	selfattentionparam.qkScale = scale;
	selfattentionparam.batchRunStatusEnable = false;
	selfattentionparam.isTriuMask = 0;
	selfattentionparam.clampMin = 0;
	selfattentionparam.clampMax = 0;
	selfattentionparam.inputLayout = atb::infer::TYPE_BSND;
	selfattentionparam.mlaVHeadSize = 0; 
	selfattentionparam.windowSize = 0;

    ParamSetter parametter;
    parametter.Input(query)
				.Input(key)
                .Input(value)
				.Input(mask)
				.Input(seqLen)		
                .Output(out);
    auto opSelfattention = selfAttentionParamCache.getOperation(selfattentionparam, "SelfAttentionOperation");
    RunAtbCmd(opSelfattention, parametter, "SelfAttentionOperation");

    return;
}
} // namespace