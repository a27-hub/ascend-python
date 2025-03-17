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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const static int64_t ROTATE_HALF = 0;
const static int64_t ROTATE_INTERLEAVED = 1;
const static int64_t BROADCAST_SHAPE_THRESHOLD = 1024;

static bool isRotaryMulBackwardMixDtypeSupport(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& cos,
    const at::Tensor& sin)
{
    return self.dtype() == cos.dtype() && self.dtype() == sin.dtype() && self.dtype() == grad.dtype() ? false : true;
}

static bool isRotaryMulBackwardHalfOpSupport(
    const at::Tensor& grad,
    const at::Tensor& cos)
{
    int64_t broadcastDimNum = 1;
    for (int64_t i = 0; i < grad.dim(); i++) {
        if (grad.sizes()[i] != cos.sizes()[i]) {
            broadcastDimNum = broadcastDimNum * grad.sizes()[i];
        }
        if (broadcastDimNum > BROADCAST_SHAPE_THRESHOLD) {
            return false;
        }
    }
    return true;
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> rotaryMulBackwardHalfSmallOp(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& cos,
    const at::Tensor& sin)
{
    at::Tensor xGradMul = at::mul(self, grad);
    at::Tensor cosGradMul = at::mul(cos, grad);
    at::Tensor sinGradMul = at::mul(sin, grad);
    std::vector<at::Tensor> sinChunk = sinGradMul.chunk(2, -1);
    at::Tensor sinChunkCat = at::cat({sinChunk[1], sinChunk[0] * (-1)}, 3);
    dx = at::add(sinChunkCat, cosGradMul);
    c10::SmallVector<int64_t, SIZE> dims;
    for (int i = 0; i < 4; i++) {
        if (self.sizes()[i] != cos.sizes()[i]) {
            dims.emplace_back(i);
        }
    }
    std::vector<at::Tensor> xChunk = self.chunk(2, -1);
    at::Tensor xChunkCat = at::cat({xChunk[1] * (-1), xChunk[0]}, 3);
    at::Tensor xChunkCatGrad = at::mul(xChunkCat, grad);
    dsin = at::sum(xChunkCatGrad, dims, true);
    dcos = at::sum(xGradMul, dims, true);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(dx, dcos, dsin);
}

static at::Tensor npu_dtype_cast_impl_op_api(const at::Tensor& self, at::ScalarType dtype)
{
    return self.dtype() == dtype ? self : self.to(dtype);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rotary_mul_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t mode)
{
    TORCH_CHECK(mode == ROTATE_HALF || mode == ROTATE_INTERLEAVED,
        "The mode of npu_rotary_mul_backward should be 0(rotate_half) or 1(rotate_interleaved), but got ", mode,
        OPS_ERROR(ErrCode::PARAM));
    DO_COMPATIBILITY(aclnnRotaryPositionEmbeddingGrad, acl_op::npu_rotary_mul_backward(grad, self, cos, sin, mode));
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1) {
        return acl_op::npu_rotary_mul_backward(grad, self, cos, sin, mode);
    }
    at::Tensor dx = npu_preparation::apply_tensor_without_format(grad.sizes(), self.options());
    at::Tensor dcos = npu_preparation::apply_tensor_without_format(cos.sizes(), self.options());
    at::Tensor dsin = npu_preparation::apply_tensor_without_format(sin.sizes(), self.options());
    bool isMixDataType = isRotaryMulBackwardMixDtypeSupport(grad, self, cos, sin);
    bool checkSupport = isRotaryMulBackwardHalfOpSupport(grad, cos);
    if (isMixDataType) {
        at::Tensor gradCast = npu_dtype_cast_impl_op_api(grad, self.scalar_type());
        at::Tensor cosCast = npu_dtype_cast_impl_op_api(cos, self.scalar_type());
        at::Tensor sinCast = npu_dtype_cast_impl_op_api(sin, self.scalar_type());
        if (mode == ROTATE_HALF && !checkSupport) {
            return rotaryMulBackwardHalfSmallOp(gradCast, self, cosCast, sinCast);
        } else {
            EXEC_NPU_CMD(aclnnRotaryPositionEmbeddingGrad, gradCast, cosCast, sinCast, self, mode, dx, dcos, dsin);
            return std::tuple<at::Tensor, at::Tensor, at::Tensor>(dx, dcos, dsin);
        }
    } else {
        if (mode == ROTATE_HALF && !checkSupport) {
            return rotaryMulBackwardHalfSmallOp(grad, self, cos, sin);
        } else {
            EXEC_NPU_CMD(aclnnRotaryPositionEmbeddingGrad, grad, cos, sin, self, mode, dx, dcos, dsin);
            return std::tuple<at::Tensor, at::Tensor, at::Tensor>(dx, dcos, dsin);
        }
    }
}
}
