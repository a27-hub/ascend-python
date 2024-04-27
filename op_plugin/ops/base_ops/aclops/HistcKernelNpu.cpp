// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor &histc_out_nocheck(at::Tensor &result, const at::Tensor &self, int64_t bins, const at::Scalar &min,
                              const at::Scalar &max)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Histogram")
        .Input(self)
        .Output(result)
        .Attr("bins", bins)
        .Attr("min", min)
        .Attr("max", max)
        .Run();
    return result;
}

at::Tensor &histogram_fixed_width(at::Tensor &result, const at::Tensor &self, at::Tensor &range, at::Tensor &nbins)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("HistogramFixedWidth")
        .Input(self)
        .Input(range)
        .Input(nbins)
        .Output(result)
        .Attr("dtype", 3)
        .Run();
    return result;
}
} // namespace

at::Tensor &histc_out(const at::Tensor &self, int64_t bins, const at::Scalar &min, const at::Scalar &max,
                      at::Tensor &result)
{
    npu_preparation::CheckOut({self}, result, self, {bins});
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        histc_out_nocheck(contiguous_result, self, bins, min, max);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        histc_out_nocheck(result, self, bins, min, max);
    }
    return result;
}

at::Tensor histc(const at::Tensor &self, int64_t bins, const at::Scalar &min, const at::Scalar &max)
{
    bool is_fp = (self.dtype() == at::kInt) ? false : true;

    at::Tensor result =
        npu_preparation::apply_tensor({bins}, self.options().dtype(at::kInt), self);

    at::Tensor nbins = npu_preparation::apply_tensor({1}, self.options().dtype(at::kInt), self);

    at::Tensor range = npu_preparation::apply_tensor({2}, self.options().dtype(is_fp ? at::kFloat : at::kInt), self);
    nbins[0] = bins;
    if (is_fp) {
        range[0] = min.toFloat();
        range[1] = max.toFloat();
    } else {
        range[0] = min.toInt();
        range[1] = max.toInt();
    }

    histogram_fixed_width(result, self, range, nbins)
    return result;
}
} // namespace acl_op
