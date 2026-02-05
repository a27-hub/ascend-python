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

#include <unordered_map>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// reduce value mapping: add/sum -> 1, multiply/prod -> 2, amax -> 3, amin -> 4
static const std::unordered_map<c10::string_view, int64_t> REDUCE_MAP = {
    {"add", 1}, {"sum", 1},
    {"multiply", 2}, {"prod", 2},
    {"amax", 3},
    {"amin", 4}
};

static inline bool reduce_valid(c10::string_view reduce)
{
    return REDUCE_MAP.find(reduce) != REDUCE_MAP.end();
}

static int64_t get_reduce(c10::string_view reduce)
{
    auto it = REDUCE_MAP.find(reduce);
    return it != REDUCE_MAP.end() ? it->second : 0;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Tensor& src, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnScatter, acl_op::scatter_out(self, dim, index, src, out));
    npu_preparation::check_tensor({self, src, index}, out, self);
    int64_t reduction = 0;
    EXEC_NPU_CMD(aclnnScatter, self, dim, index, src, reduction, out);
    return out;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Tensor& src, c10::string_view reduce, at::Tensor& out)
{
    npu_preparation::check_tensor({self, src, index}, out, self);
    TORCH_CHECK(reduce_valid(reduce),
        "Reduce should be one of: add, sum, multiply, prod, amax, amin. Got: ", reduce,
        OPS_ERROR(ErrCode::PARAM));
    int64_t reduction = get_reduce(reduce);
    EXEC_NPU_CMD(aclnnScatter, self, dim, index, src, reduction, out);
    return out;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Scalar& value, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnScatterValue, acl_op::scatter_out(self, dim, index, value, out));
    npu_preparation::check_tensor({self, index}, out, self);
    int64_t reduction = 0;
    EXEC_NPU_CMD(aclnnScatterValue, self, dim, index, value, reduction, out);
    return out;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Scalar& value, c10::string_view reduce, at::Tensor& out)
{
    npu_preparation::check_tensor({self, index}, out, self);
    TORCH_CHECK(reduce_valid(reduce),
        "Reduce should be one of: add, sum, multiply, prod, amax, amin. Got: ", reduce,
        OPS_ERROR(ErrCode::PARAM));
    int64_t reduction = get_reduce(reduce);
    EXEC_NPU_CMD(aclnnScatterValue, self, dim, index, value, reduction, out);
    return out;
}

at::Tensor &scatter_(at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &src)
{
    DO_COMPATIBILITY(aclnnInplaceScatter, acl_op::scatter_(self, dim, index, src));
    npu_preparation::check_tensor({self, src, index}, self, self);
    int64_t reduction = 0;
    EXEC_NPU_CMD(aclnnInplaceScatter, self, dim, index, src, reduction);
    return self;
}

at::Tensor &scatter_(at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Scalar& value)
{
    DO_COMPATIBILITY(aclnnInplaceScatterValue, acl_op::scatter_(self, dim, index, value));
    npu_preparation::check_tensor({self, index}, self, self);
    int64_t reduction = 0;
    EXEC_NPU_CMD(aclnnInplaceScatterValue, self, dim, index, value, reduction);
    return self;
}
}
