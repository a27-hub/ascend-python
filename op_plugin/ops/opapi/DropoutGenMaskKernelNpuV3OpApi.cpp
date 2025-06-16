// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _npu_dropout_gen_mask_v3(at::Tensor & bitmask, at::IntArrayRef size, double p, int64_t seed, int64_t offset)
{
    TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p,
                OPS_ERROR(ErrCode::VALUE));

    c10::SmallVector<int64_t, op_infer::SIZE> self_size = {786432};
    at::Tensor self = npu_preparation::apply_tensor_without_format(self_size, bitmask.options().dtype(at::kBFloat16));
    double from = 0;
    double to = 1;
    EXEC_NPU_CMD(aclnnInplaceUniform, self, from, to, seed, offset);

    int64_t numels = c10::multiply_integers(size);
    c10::SmallVector<int64_t, SIZE> shapeSize = {numels};
    at::IntArrayRef shapeArray = at::IntArrayRef(shapeSize);

    float p_cast = static_cast<float>(p); 
    uint64_t length = (static_cast<uint64_t>(numels) + 128 - 1) / 128 * 128 / 8;
    at::Tensor mask = bitmask.narrow(0, 0, length);

    EXEC_NPU_CMD(aclnnDropoutV3, shapeArray, self, p_cast, mask, self);
    return ;
}

}  // namespace op_api
