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
using SelfAttentionParam = atb::infer::SelfAttentionParam;

namespace {
static constexpr int64_t LONG_COMPRESS_LEN = 2048;
static constexpr int64_t LONG_COMPRESS_NZ_DIM1 = LONG_COMPRESS_LEN / 16;
static constexpr int64_t SHORT_COMPRESS_LEN = 128;
static constexpr int64_t SHORT_COMPRESS_NZ_DIM1 = SHORT_COMPRESS_LEN / 16;

bool IsNormCompressMask(const at::Tensor &mask)
{
    if (mask.dim() == 4 && mask.size(0) == 1 && mask.size(3) == 16) {
        return (mask.size(1) == LONG_COMPRESS_NZ_DIM1 && mask.size(2) == LONG_COMPRESS_LEN) ||
               (mask.size(1) == SHORT_COMPRESS_NZ_DIM1 && mask.size(2) == SHORT_COMPRESS_LEN);
    }
    if (mask.dim() == 2) {
        return (mask.size(0) == LONG_COMPRESS_LEN && mask.size(1) == LONG_COMPRESS_LEN) ||
               (mask.size(0) == SHORT_COMPRESS_LEN && mask.size(1) == SHORT_COMPRESS_LEN);
    }
    return false;
}
} // namespace

void _npu_flash_attention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &mask, const at::Tensor &seq_len, const double scale_value, const int64_t num_heads, const int64_t num_kv_heads, at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    OpParamCache<SelfAttentionParam>& selfAttentionParamCache = OpParamCache<SelfAttentionParam>::getInstance();
    SelfAttentionParam selfattentionparam;

    const bool norm_compress = IsNormCompressMask(mask);

    selfattentionparam.calcType = SelfAttentionParam::PA_ENCODER;
    selfattentionparam.kernelType = SelfAttentionParam::KERNELTYPE_DEFAULT;
    selfattentionparam.clampType = SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
    selfattentionparam.maskType = norm_compress ? SelfAttentionParam::MASK_TYPE_NORM_COMPRESS
                                                : SelfAttentionParam::MASK_TYPE_NORM;
    selfattentionparam.kvcacheCfg = SelfAttentionParam::K_CACHE_V_CACHE;
    selfattentionparam.scaleType = SelfAttentionParam::SCALE_TYPE_TOR;
    selfattentionparam.quantType = SelfAttentionParam::TYPE_QUANT_UNDEFINED;
    selfattentionparam.cacheType = SelfAttentionParam::CACHE_TYPE_NORM;
    selfattentionparam.outDataType = ACL_DT_UNDEFINED;
    selfattentionparam.headNum = num_heads;
    selfattentionparam.kvHeadNum = num_kv_heads;
    selfattentionparam.qScale = 1;
    selfattentionparam.qkScale = scale_value;
    selfattentionparam.batchRunStatusEnable = false;
    selfattentionparam.isTriuMask = norm_compress ? 1 : 0;
    selfattentionparam.clampMin = 0;
    selfattentionparam.clampMax = 0;
    selfattentionparam.inputLayout = atb::infer::TYPE_BSND;
    selfattentionparam.mlaVHeadSize = 0;
    selfattentionparam.windowSize = 0;

    ParamSetter parametter;
    if (norm_compress) {
        // 310P needs BOTH host-side seq_len (for tiling) AND device-side (for kernel exec).
        at::Tensor sl_npu = seq_len.device().type() == at::kCPU
                                ? seq_len.to(c10::Device(c10::DeviceType::PrivateUse1))
                                : seq_len;
        at::Tensor sl_cpu = seq_len.device().type() == at::kCPU
                                ? seq_len.contiguous()
                                : seq_len.cpu().contiguous();
        parametter.Input(query, true)
            .Input(key, true)
            .Input(value, true)
            .Input(mask)
            .Input(sl_npu, true)
            .Output(out);
        parametter.variant_pack_.inTensors[4].hostData = sl_cpu.data_ptr();
    } else {
        parametter.Input(query, true)
            .Input(key, true)
            .Input(value, true)
            .Input(mask)
            .Input(seq_len, true)
            .Output(out);
    }

    auto opSelfattention = selfAttentionParamCache.getOperation(selfattentionparam, "SelfAttentionOperation");
    RunAtbCmd(opSelfattention, parametter, "SelfAttentionOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_flash_attention(Tensor query, Tensor key, Tensor value, Tensor mask, Tensor seq_len, float scale_value, int num_heads, int num_kv_heads, Tensor(a!) out) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_flash_attention", TORCH_FN(atb::_npu_flash_attention));
}
}
} // namespace
