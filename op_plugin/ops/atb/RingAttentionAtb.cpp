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

using namespace std;
namespace atb {
using RingAttentionParam = atb::infer::RingAttentionParam;
void _npu_ring_attention(const at::Tensor &query, const at::Tensor &key,
                         const at::Tensor &value, const at::Tensor &mask,
                         const at::Tensor &seq_len, int64_t head_num,
                         int64_t kv_head_num, float qk_scale,
                         const at::Tensor &prev_out, const at::Tensor &prev_lse,
                         at::Tensor &output, at::Tensor &softmax_lse) {
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  OpParamCache<RingAttentionParam> &ringAttentionParamCache =
      OpParamCache<RingAttentionParam>::getInstance();

  RingAttentionParam ringAttentionParam;
  // TODO: judge calcType by input tensor
  ringAttentionParam.calcType = RingAttentionParam::MIDDLE;
  ringAttentionParam.headNum = head_num;
  ringAttentionParam.kvHeadNum = kv_head_num;
  ringAttentionParam.qkScale = qk_scale;

  ParamSetter paramsetter;
  paramsetter.Input(query)
      .Input(key)
      .Input(value)
      .Input(mask)
      .Input(seq_len)
      .Output(output)
      .Output(softmax_lse);

  auto opRingAttention = ringAttentionParamCache.getOperation(
      ringAttentionParam, "RingAttentionOperation");
  RunAtbCmd(opRingAttention, paramsetter, "RingAttentionOperation");

  return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m) {
  m.def(
      "_npu_ring_attention(Tensor query, Tensor key, Tensor value, Tensor "
      "mask, Tensor seq_len, int head_num, int kv_head_num, float qk_scale, "
      "Tensor prev_out, Tensor prev_lse, Tensor(a!) output, Tensor(b!) "
      "softmax_lse) -> ()");
}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m) {
  m.impl("_npu_ring_attention", TORCH_FN(atb::_npu_ring_attention));
}
}  // namespace

}  // namespace atb
