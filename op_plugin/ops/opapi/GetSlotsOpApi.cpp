// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;

    at::Tensor npu_get_slots(
        const at::Tensor &seq_len,
        const at::Tensor &kv_cache_len,
        const at::Tensor &block_table,
        int64_t page_size,
        int64_t token_num)
    {
        TORCH_CHECK(seq_len.dim() == 1, "seq_len dim should equal 1", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(kv_cache_len.dim() == 1, "kv_cache_len dim should equal 1", OPS_ERROR(ErrCode::PARAM));

        at::SmallVector <int64_t, op_infer::SIZE> y_size = {token_num};

        at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(c10::ScalarType::Int));

        EXEC_NPU_CMD(aclnnGetSlots, seq_len, kv_cache_len, block_table, page_size, token_num, y);

        return y;
    }
}