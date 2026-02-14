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
using npu_utils = at_npu::native::NpuUtils;
const size_t DIMS_ONE = 1;
const size_t DIMS_TWO = 2;
const size_t DIMS_THREE = 3;
const size_t DIMS_FOUR = 4;

at::Tensor npu_mhc_post(const at::Tensor &x, const at::Tensor &h_res, const at::Tensor &h_out, const at::Tensor &h_post)
{
    size_t xDims = x.dim();
    size_t hResDims = h_res.dim();
    size_t hOutDims = h_out.dim();
    size_t hPostDims = h_post.dim();

    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "float16 or bfloat16 tensor expected but got a tensor with dtype: ",
        x.scalar_type(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        h_res.scalar_type() == at::kFloat,
        "float tensor expected but got a tensor with dtype: ",
        h_res.scalar_type(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        x.scalar_type() == h_out.scalar_type(), "The dtype of x and h_out should be same, but got ",
        x.scalar_type(), "and ", h_out.scalar_type(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        h_res.scalar_type() == h_post.scalar_type(), "The dtype of h_res and h_post should be same, but got ",
        h_res.scalar_type(), "and ", h_post.scalar_type(), OPS_ERROR(ErrCode::PARAM));
    
    TORCH_CHECK(
        xDims == DIMS_THREE || xDims == DIMS_FOUR, "x should be 3D or 4D.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        xDims == hResDims, "The dims of x and hRes should be equal, but xDims is ", xDims,
        "and hResDims is ", hResDims, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        hOutDims == DIMS_THREE || hOutDims == DIMS_TWO, "h_out should be 2D or 3D.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        hOutDims == hPostDims, "The dims of hOut and hPost should be equal, but hOutDims is ",
        hOutDims, "and hPostDims is ", hPostDims, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        h_res.size(hResDims - DIMS_ONE) == h_res.size(hResDims - DIMS_TWO),
        "The last two dims of hRes should be same.", OPS_ERROR(ErrCode::PARAM));

    for(size_t i = 0; i < xDims - DIMS_ONE; i++) {
        int32_t xDimI = x.size(i);
        int32_t hResDimI = h_res.size(i);
        TORCH_CHECK(
            xDimI == hResDimI,
            "The shapes of x and hRes do not satisfy the required constraint relationship.",
            OPS_ERROR(ErrCode::PARAM));
    }
    for (size_t i = 0; i < hOutDims - DIMS_ONE; i++) {
        int32_t hOutDimI = h_out.size(i);
        int32_t hPostDimI = h_post.size(i);
        TORCH_CHECK(
            hOutDimI == hPostDimI,
            "The shapes of hOut and hPost do not satisfy the required constraint relationship.",
            OPS_ERROR(ErrCode::PARAM));
    }
    for (size_t i = 0; i < hResDims - DIMS_TWO; i++) {
        int32_t hResDimI = h_res.size(i);
        int32_t hOutDimI = h_out.size(i);
        TORCH_CHECK(
            hResDimI == hOutDimI,
            "The shapes of hRes and hOut do not satisfy the required constraint relationship.",
            OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(
        x.size(xDims - DIMS_ONE) == h_out.size(hOutDims - DIMS_ONE),
        "The last dims of x and hOut should be same.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        h_post.size(hPostDims - DIMS_ONE) == h_res.size(hResDims - DIMS_ONE),
        "The last dims of hPost and hRes should be same.",
        OPS_ERROR(ErrCode::PARAM));

    std::vector<int64_t> yShape;
    for (size_t i = 0; i < xDims; i++) {
        yShape.push_back(x.size(i));
    }
    at::Tensor y = npu_preparation::apply_tensor_without_format(yShape, x.options());
    EXEC_NPU_CMD(aclnnMhcPost, x, h_res, h_out, h_post, y);
    return y;
}

}