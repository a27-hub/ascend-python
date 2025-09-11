// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
inline c10::SmallVector<int64_t, SIZE> swiglu_infershape(const at::Tensor &x, int64_t dim)
{
    if (dim < 0) {
        dim += x.sizes().size();
    }
    TORCH_CHECK(dim < x.sizes().size(), "dim out of range", dim);
    auto output_sizes = op_infer::array_to_small_vector(x.sizes());
    output_sizes[dim] /= 2;
    return output_sizes;
}

at::Tensor npu_swiglu(const at::Tensor &x, int64_t dim)
{
    auto output_sizes = swiglu_infershape(x, dim);
    at::Tensor result = at::empty(output_sizes, x.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("SwiGlu")
            .Input(x)
            .Output(result)
            .Attr("dim", dim)
            .Run();
    return result;
}
} // namespace acl_op
