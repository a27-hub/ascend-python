// Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
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

at::Tensor _cdist_backward(
    const at::Tensor& grad,
    const at::Tensor& x1,
    const at::Tensor& x2,
    double p,
    const at::Tensor& cdist)
{
    DO_COMPATIBILITY(aclnnCdistBackward, acl_op::_cdist_backward(grad, x1, x2, p, cdist));

    float p_cast;

    if (std::isinf(p)) {
        p_cast = -1;
    } else {
        TORCH_CHECK(
            p <= std::numeric_limits<float>::max(),
            "The value of p (" + std::to_string(p) + ") exceeds the maximum value of float ("
                + std::to_string(std::numeric_limits<float>::max()) + ")" + OPS_ERROR(ErrCode::PARAM));
        p_cast = static_cast<float>(p);
    }
    // The current operator has precision issues when handling integers and infinity.
    bool p_in_range = (p_cast >= 0.0 && p_cast <= 2.0) || (p_cast == -1);
    if (p_in_range && (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950)) {
        return acl_op::_cdist_backward(grad, x1, x2, p, cdist);
    }
    
    // Broadcast x1 to match x2's dimensions (excluding last two dims)
    auto x1_sizes = x1.sizes();
    auto x2_sizes = x2.sizes();
    
    // Compute broadcast shape for x1: broadcast all dims except last two
    std::vector<int64_t> broadcast_dims;
    size_t x1_ndim = x1_sizes.size();
    size_t x2_ndim = x2_sizes.size();
    size_t max_ndim = std::max(x1_ndim, x2_ndim);
    
    for (size_t i = 0; i < max_ndim - 2; ++i) {
        int64_t dim1 = (i < x1_ndim - 2) ? x1_sizes[i] : 1;
        int64_t dim2 = (i < x2_ndim - 2) ? x2_sizes[i] : 1;
        TORCH_CHECK(
            dim1 == dim2 || dim1 == 1 || dim2 == 1,
            "The size of tensor a ", dim1, " must match the size of tensor b ", dim2,
            " at non-singleton dimension ", i, OPS_ERROR(ErrCode::PARAM));
        broadcast_dims.push_back(std::max(dim1, dim2));
    }
    // Add last two dimensions from x1
    broadcast_dims.push_back(x1_sizes[x1_ndim - 2]);
    broadcast_dims.push_back(x1_sizes[x1_ndim - 1]);
    
    at::Tensor x1_broadcast = x1;
    if (x1_sizes.vec() != broadcast_dims) {
        x1_broadcast = x1.expand(broadcast_dims);
    }
    
    auto output_size = x1_broadcast.sizes();
    auto output_dtype = grad.scalar_type();

    at::Tensor out = at_npu::native::OpPreparation::apply_tensor_without_format(
        output_size,
        grad.options().dtype(output_dtype));


    EXEC_NPU_CMD(aclnnCdistBackward, grad, x1_broadcast, x2, cdist, p_cast, out);

    // Reduce gradient back to original x1 shape (sum over broadcast dims)
    if (x1_sizes.vec() != broadcast_dims) {
        std::vector<int64_t> reduce_dims;
        size_t out_ndim = out.dim();
        size_t orig_ndim = x1_sizes.size();
        
        for (size_t i = 0; i < out_ndim - 2; ++i) {
            if (i >= orig_ndim - 2 || (x1_sizes[i] == 1 && out.sizes()[i] > 1)) {
                reduce_dims.push_back(i);
            }
        }
        
        if (!reduce_dims.empty()) {
            out = op_api::sum(out, reduce_dims, true, c10::nullopt);
        }
        out = out.reshape(x1_sizes);
    }

    return out;
}
}