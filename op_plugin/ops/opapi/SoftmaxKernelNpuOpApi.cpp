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
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor softmax(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnSoftmax, acl_op::softmax(self, dim, dtype));
    auto result = [&]() {
        at::NoNamesGuard guard;
        at::Tensor converted = self;
        if (dtype.has_value() && self.scalar_type() != dtype.value()) {
            converted = at_npu::native::custom_ops::_npu_dtype_cast(self, dtype.value());
        }
        return at::_softmax(converted, dim, false);
    }();
    at::namedinference::propagate_names(result, self);
    return result;
}

at::Tensor softmax(const at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype)
{
    return op_api::softmax(self, dimname_to_position(self, dim), dtype);
}

at::Tensor & _softmax_out(const at::Tensor& self, int64_t dim, bool half_to_float, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnSoftmax, acl_op::_softmax_out(self, dim, half_to_float, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = half_to_float ? at::ScalarType::Float : self.scalar_type();
    if (!out.sizes().equals(output_size_0)) {
        out.resize_(output_size_0);
    }
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSoftmax, self, dim, out);
    return out;
}

}
