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
using RingMLAParam = atb::infer::RingMLAParam;
void _npu_ring_mla_ring(const at::Tensor &query_nope,
                        const at::Tensor &query_rope,
                        const at::Tensor &key_nope, const at::Tensor &key_rope,
                        const at::Tensor &value,
                        const c10::optional<at::Tensor> &mask,
                        const at::Tensor &seq_len, const at::Tensor &prev_out,
                        const at::Tensor &prev_lse, int64_t head_num,
                        int64_t kv_head_num, double qk_scale,
                        at::Tensor &output, at::Tensor &softmax_lse) {
  const c10::OptionalDeviceGuard device_guard(device_of(query_nope));
  OpParamCache<RingMLAParam> &ringMLAParamCache =
      OpParamCache<RingMLAParam>::getInstance();

  RingMLAParam ringMLAParam;
  ringMLAParam.calcType = RingMLAParam::CALC_TYPE_FIRST_RING;
  ringMLAParam.maskType =
      mask.has_value() ? RingMLAParam::MASK_TYPE_TRIU : RingMLAParam::NO_MASK;
  ringMLAParam.headNum = head_num;
  ringMLAParam.kvHeadNum = kv_head_num;
  ringMLAParam.qkScale = qk_scale;

  ParamSetter paramsetter;
  paramsetter.Input(query_nope)
      .Input(query_rope)
      .Input(key_nope)
      .Input(key_rope)
      .Input(value)
      .Input(mask)
      .Input(seq_len)
      .Input(prev_out)
      .Input(prev_lse)
      .Output(output)
      .Output(softmax_lse);

  auto opRingMLAOperation =
      ringMLAParamCache.getOperation(ringMLAParam, "RingMLAOperation");
  RunAtbCmd(opRingMLAOperation, paramsetter, "RingMLAOperation");

  return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m) {
  m.def(
      "_npu_ring_mla_ring(Tensor query_nope, Tensor query_rope, Tensor "
      "key_nope, Tensor key_rope, Tensor value, *, Tensor? mask=None, Tensor seq_len, "
      "Tensor prev_out, Tensor prev_lse, int head_num, int kv_head_num, float "
      "qk_scale, Tensor(a!) output, Tensor(b!) softmax_lse) -> ()");
}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m) {
  m.impl("_npu_ring_mla_ring", TORCH_FN(atb::_npu_ring_mla_ring));
}
}  // namespace

}  // namespace atb
