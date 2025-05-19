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

namespace atb {
std::tuple<at::Tensor&, at::Tensor&> _npu_mla_out(const at::Tensor &qNope, const at::Tensor &qRope, const at::Tensor & ctKV, const at::Tensor &kRope, const at::Tensor &block_tables,
    const at::Tensor & contextLens, const at::Tensor &mask, const at::Tensor &qseqlen, const at::Tensor & qkDescale, const at::Tensor & pvDescale, 
    int64_t headNum, double qkScale, int64_t kvHeadNum, int64_t maskType, int64_t CalcType, int64_t cacheMode,
    at::Tensor& attenOut,
    at::Tensor& lse)
{
    const c10::OptionalDeviceGuard device_guard(device_of(qNope));
    float qkScale_float = static_cast<float>(qkScale);
    EXEC_ATB_CMD(aclnnMLA, qNope, qRope, ctKV, kRope, block_tables, contextLens, mask, qseqlen, qkDescale, pvDescale, headNum, qkScale_float, kvHeadNum, maskType, CalcType, cacheMode, attenOut, lse);
    return std::forward_as_tuple(attenOut, lse);
}

std::tuple<at::Tensor, at::Tensor> _npu_mla(const at::Tensor &qNope, const at::Tensor &qRope, const at::Tensor & ctKV, const at::Tensor &kRope, const at::Tensor &block_tables,
    const at::Tensor & contextLens, const at::Tensor &mask, const at::Tensor &qseqlen, const at::Tensor & qkDescale, const at::Tensor & pvDescale, 
    int64_t headNum, double qkScale, int64_t kvHeadNum, int64_t maskType, int64_t CalcType, int64_t cacheMode)
{
    const c10::OptionalDeviceGuard device_guard(device_of(qNope));
    auto num_tokens = qRope.size(0);
    auto num_heads = qRope.size(1);
    at::Tensor attenOut = at::empty({num_tokens, num_heads, 512}, qRope.options());
    at::Tensor lse;
    if (CalcType == 2) {
        lse = at::empty({num_tokens, num_heads, 1}, qRope.options());
    } else {
        lse = at::Tensor();
    }
    float qkScale_float = static_cast<float>(qkScale);
    EXEC_ATB_CMD(aclnnMLA, qNope, qRope, ctKV, kRope, block_tables, contextLens, mask, qseqlen, qkDescale, pvDescale, headNum, qkScale_float, kvHeadNum, maskType, CalcType, cacheMode, attenOut, lse);
    return std::make_tuple(std::move(attenOut), std::move(lse));
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_mla(Tensor qNope, Tensor qRope, Tensor ctKV, Tensor kRope,Tensor block_tables, Tensor contextLens,Tensor mask, Tensor qseqlen,Tensor qkDescale, Tensor pvDescale,"
    "int headNum, float qkScale, int kvHeadNum, int maskType, int CalcType, int cacheMode) -> (Tensor, Tensor)");
    m.def("_npu_mla.out(Tensor qNope, Tensor qRope, Tensor ctKV, Tensor kRope,Tensor block_tables, Tensor contextLens,Tensor mask, Tensor qseqlen,Tensor qkDescale, Tensor pvDescale,"
    "int headNum, float qkScale, int kvHeadNum, int maskType, int CalcType, int cacheMode, *, Tensor(a!) attenOut, Tensor(b!) lse) -> (Tensor(a!),  Tensor(b!))");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_mla", TORCH_FN(atb::_npu_mla));
    m.impl("_npu_mla.out", TORCH_FN(atb::_npu_mla_out));
}
}

} // namespace atb