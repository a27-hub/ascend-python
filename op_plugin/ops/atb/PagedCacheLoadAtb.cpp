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

namespace atb {
at::Tensor &_npu_paged_cache_load(
    const at::Tensor &key_cache, const at::Tensor &value_cache,
    const at::Tensor &block_table, const at::Tensor &context_lens,
    at::Tensor &key, at::Tensor &value, at::Tensor &seq_starts,
    int64_t kv_cache_cfg, bool is_seq_lens_cumsum_mode, bool has_seq_starts) {
  const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
  EXEC_ATB_CMD(AtbCacheLoad, key_cache, value_cache, block_table, context_lens,
               key, value, seq_starts);
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m) {
  m.def(
      "_npu_paged_cache_load(Tensor key_cache, Tensor value_cache, Tensor "
      "block_table, Tensor context_lens, Tensor(a!) key, Tensor(b!)value, "
      "Tensor? seq_starts=None, int kv_cache_cfg=0, bool "
      "is_seq_lens_cumsum_mode=false, bool has_seq_starts=false ) -> ()");
}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m) {
  m.impl("_npu_paged_cache_load", TORCH_FN(atb::_npu_paged_cache_load));
}
}  // namespace
}  // namespace atb