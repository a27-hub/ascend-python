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
using PagedAttentionParam = atb::infer::PagedAttentionParam;
void _npu_paged_attention_splitfuse(const at::Tensor &query, const at::Tensor &key_cache, const at::Tensor &value_cache, const at::Tensor &block_table, const at::Tensor &context_lens, const at::Tensor &mask, const at::Tensor &seq_len, int64_t num_kv_heads, int64_t num_heads, double scale_value, int64_t mask_type, at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    OpParamCache<PagedAttentionParam>& pagedAttentionParamCache = OpParamCache<PagedAttentionParam>::getInstance();
    PagedAttentionParam pagedparam;
    pagedparam.headNum = num_heads;
    pagedparam.qkScale = scale_value;
    pagedparam.kvHeadNum = num_kv_heads;
    pagedparam.maskType = static_cast<PagedAttentionParam::MaskType>(mask_type);
    pagedparam.batchRunStatusEnable = false;
    pagedparam.quantType = PagedAttentionParam::TYPE_QUANT_UNDEFINED;
    pagedparam.outDataType = ACL_DT_UNDEFINED;
    pagedparam.hasQuantOffset = false;
    pagedparam.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
    pagedparam.calcType = PagedAttentionParam::CALC_TYPE_SPEC;
    pagedparam.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
    pagedparam.inputLayout = atb::infer::TYPE_BSND;
    pagedparam.mlaVHeadSize = 0;

    // 310P needs BOTH host-side qLens/contextLens (for tiling) AND device-side
    // (for kernel exec). The standard ParamSetter::Input(tensor, true) only sets
    // ONE side. Build NPU + CPU copies and patch the variant pack ourselves.
    at::Tensor cl_npu = context_lens.device().type() == at::kCPU
                            ? context_lens.to(c10::Device(c10::DeviceType::PrivateUse1))
                            : context_lens;
    at::Tensor cl_cpu = context_lens.device().type() == at::kCPU
                            ? context_lens.contiguous()
                            : context_lens.cpu().contiguous();
    at::Tensor sl_npu = seq_len.device().type() == at::kCPU
                            ? seq_len.to(c10::Device(c10::DeviceType::PrivateUse1))
                            : seq_len;
    at::Tensor sl_cpu = seq_len.device().type() == at::kCPU
                            ? seq_len.contiguous()
                            : seq_len.cpu().contiguous();

    ParamSetter paramsetter;
    paramsetter.Input(query, true)
        .Input(key_cache)
        .Input(value_cache)
        .Input(block_table, true)
        .Input(cl_npu, true)
        .Input(mask)
        .Input(sl_npu, true)
        .Output(out);

    // index 4 = contextLens, index 6 = seq_len/qLens; both must have hostData.
    paramsetter.variant_pack_.inTensors[4].hostData = cl_cpu.data_ptr();
    paramsetter.variant_pack_.inTensors[6].hostData = sl_cpu.data_ptr();

    auto opPaged = pagedAttentionParamCache.getOperation(pagedparam, "PagedAttentionOperation");
    RunAtbCmd(opPaged, paramsetter, "PagedAttentionOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_paged_attention_splitfuse(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, Tensor context_lens, Tensor mask, Tensor seq_len, int num_kv_heads, int num_heads, float scale_value, *, int mask_type=3, Tensor(a!) out) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_paged_attention_splitfuse", TORCH_FN(atb::_npu_paged_attention_splitfuse));
}
}

}
