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
#include <acl/acl.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/AtbCommon.h"

namespace atb {
std::tuple<at::Tensor&, at::Tensor&> _npu_ring_mla_out(const at::Tensor &q_nope, const at::Tensor &q_rope, const at::Tensor &k_nope, const at::Tensor &k_rope, const at::Tensor &value,
    const at::Tensor &mask, const at::Tensor &seqlen, int64_t head_num, int64_t kv_head_num, const c10::optional<at::Tensor> &pre_out, const c10::optional<at::Tensor> &prev_lse,
    double qk_scale, int64_t kernel_type, int64_t mask_type, int64_t input_layout, int64_t calc_type,
    at::Tensor& output,
    at::Tensor& softmax_lse)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q_nope));
    float qkScale_float = static_cast<float>(qk_scale);
    EXEC_ATB_CMD(AtbRingMLA, q_nope, q_rope, k_nope, k_rope, value, mask, seqlen, pre_out, prev_lse, head_num, kv_head_num, qkScale_float, kernel_type, mask_type, input_layout, calc_type, output, softmax_lse);
    return std::forward_as_tuple(output, softmax_lse);
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_ring_mla.out(Tensor q_nope, Tensor q_rope, Tensor k_nope, Tensor k_rope, Tensor value, Tensor mask, Tensor seqlen, int head_num, int kv_head_num, *, Tensor? pre_out=None, Tensor? prev_lse=None, float qk_scale=1, int kernel_type=1, int mask_type=1, int input_layout=0, int calc_type=1, Tensor(a!) output, Tensor(b!) softmax_lse) -> (Tensor(a!), Tensor(b!))");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_ring_mla.out", TORCH_FN(atb::_npu_ring_mla_out));
}
}

} // namespace
