// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
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


bool is_fused_op_optim(const at::Tensor& self, at::IntArrayRef split_sizes) {
    if (!op_plugin::utils::is_gte_cann_version_830rc1()) {
        return false;
    }

    if (self.dim() != 2) {
        return false;
    }

    int64_t len = split_sizes.size();
    if ((len <= 32) || (len > 64)) {
        return false;
    }

    at::ScalarType dtype = self.scalar_type();

    int64_t dim0 = self.size(0);
    int64_t dim1 = self.size(1);

    if (dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16) {
        return (dim0 <= 16) && (dim1 > 65536);
    } else if (dtype == at::ScalarType::Float) {
        return (dim0 <= 8) && (dim1 > 65536);
    } else {
        return false;
    }
}


void split_with_sizes_copy_out(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim, at::TensorList out)
{
    if(is_fused_op_optim) {
        EXEC_NPU_CMD(aclnnSplitWithSize, self, split_sizes, dim, out);
    } else {
        at::native::split_with_sizes_copy_out(self, split_sizes, dim, out);
        // auto array = self.split_with_sizes(split_sizes, dim);
        // copy_tensor_array_to_out("split_with_sizes_copy_out", array, out);
    }
}

}  // namespace op_api
