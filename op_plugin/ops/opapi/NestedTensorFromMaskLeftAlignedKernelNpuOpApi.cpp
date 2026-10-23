// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {

// This ATen op returns a bool used to probe whether the mask is left-aligned (i.e. free of holes, so the
// nested-tensor fast path can be taken). NPU does not support the nested-tensor fast path, so it simply
// returns false to make the upper-level SDPA dispatch fall back to the dense path.
bool _nested_tensor_from_mask_left_aligned(const at::Tensor& /*t*/, const at::Tensor& /*mask*/) {
    TORCH_NPU_WARN_ONCE("now npu is not support torch._nested_tensor_from_mask");
    return false;
}

}  // namespace op_api
