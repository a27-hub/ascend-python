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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"


namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
void exec_scatter_reduce(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    bool include_self,
    at::Tensor& out,
    const char* op_name)
{
    if (reduce == "sum" || reduce == "add") {
        EXEC_NPU_CMD(aclnnScatterAdd, self, dim, index, src, include_self, out);
    } else if (reduce == "amin" || reduce == "min") {
        EXEC_NPU_CMD(aclnnScatterMin, self, dim, index, src, include_self, out);
    } else if (reduce == "amax" || reduce == "max") {
        EXEC_NPU_CMD(aclnnScatterMax, self, dim, index, src, include_self, out);
    } else if (reduce == "prod" || reduce == "mul") {
        EXEC_NPU_CMD(aclnnScatterMul, self, dim, index, src, include_self, out);
    } else if (reduce == "mean") {
        EXEC_NPU_CMD(aclnnScatterMean, self, dim, index, src, include_self, out);
    } else {
        TORCH_CHECK(false, op_name, ": expected reduce to be one of sum, add, amin, min, amax, max, mul, prod or mean, but got ", reduce, OPS_ERROR(ErrCode::PARAM));
    }
}
} // namespace

at::Tensor scatter_reduce(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    bool include_self)
{
    if (include_self && (reduce == "sum" || reduce == "add")) {
        DO_COMPATIBILITY(aclnnScatterAdd, acl_op::scatter_add(self, dim, index, src));
    }
    auto result = self.clone(at::MemoryFormat::Contiguous);
    npu_preparation::CheckMemory({result, index, src}, {result});
    exec_scatter_reduce(result, dim, index, src, reduce, include_self, result, "scatter_reduce()");
    return result;
}

at::Tensor& scatter_reduce_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    bool include_self,
    at::Tensor& out)
{
    npu_preparation::CheckMemory({self, index, src}, {out});
    exec_scatter_reduce(self, dim, index, src, reduce, include_self, out, "scatter_reduce()");
    return out;
}

at::Tensor& scatter_reduce_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    bool include_self)
{
    npu_preparation::CheckMemory({self, index, src}, {self});
    if (include_self && (reduce == "sum" || reduce == "add")) {
        DO_COMPATIBILITY(aclnnScatterAdd, acl_op::scatter_add_(self, dim, index, src));
    }
    exec_scatter_reduce(self, dim, index, src, reduce, include_self, self, "scatter_reduce_()");
    return self;
}


}
