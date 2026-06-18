// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

namespace {
at::Tensor aclnnMatmulChecksumNotFound(const at::Tensor &val)
{
    TORCH_CHECK(false, "aclnnMatmulChecksum not found. " + OPS_ERROR(ErrCode::INTERNAL));
    return val;
}
} // namespace

at::Tensor _npu_matmul_checksum(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c)
{
    // TODO(catlass): replace aclnnMatmulChecksum with the actual aclnn function
    // name and align the argument list with the real aclnn header once the
    // optimized matmul checksum kernel (catlass) lands. Until then
    // DO_COMPATIBILITY short-circuits to aclnnMatmulChecksumNotFound, so the op
    // stays unreachable from PTA, which falls back to the pure-Python
    // implementation in torch_npu/asd/checksum.py.
    DO_COMPATIBILITY(aclnnMatmulChecksum, aclnnMatmulChecksumNotFound(c));
    at::Tensor result = npu_preparation::apply_tensor_without_format({}, c.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnMatmulChecksum, a, b, c, result);
    return result;
}
} // namespace op_api
