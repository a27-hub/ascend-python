// Copyright (c) 2026 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

at::Tensor npu_isin_part_v1(const at::Tensor& value, const at::Tensor& index, int64_t elements_num)
{
    TORCH_CHECK(elements_num >= 0, "elements_num must be non-negative, but got ", elements_num,
                OPS_ERROR(ErrCode::PARAM));

    // 1. Prepare output shape
    c10::SmallVector<int64_t, 8> output_size = {elements_num};

    // 2. Prepare output Tensor (Bool + ND format)
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size,
        value.options().dtype(at::kBool),
        ACL_FORMAT_ND
    );

    // 3. Handle index type (Int64 -> Int32)
    at::Tensor index_opt = (index.scalar_type() == at::kLong) ?
        at_npu::native::custom_ops::npu_dtype_cast(index, at::kInt) : index;

    // 4. Execution call
    EXEC_NPU_CMD(aclnnIsinPartV1, value, index_opt, elements_num, result);

    return result;
}

} // namespace op_api
