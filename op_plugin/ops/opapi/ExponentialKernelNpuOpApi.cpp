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


#include <limits>
#include "op_plugin/OpApiInterface.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace op_api {

at::Tensor& exponential_(at::Tensor& self, double lambd, c10::optional<at::Generator> generator)
{
    TORCH_CHECK(lambd > 0.0, "exponential_ expects lambd > 0.0, but found lambd=",
        lambd, OPS_ERROR(ErrCode::PARAM));
    if (std::isinf(lambd)) {
        self.zero_();
        return self;
    }
    
    // Low-precision floating types (bf16/fp16) must not be sampled and
    // transformed in place: the uniform source, the (1 - u) subtraction and
    // the final clamp all collapse onto the coarse bf16/fp16 grid, so the
    // exponential tail is quantized to a handful of levels (bf16: min ~= eps/2
    // = 3.9e-3, only ~128 distinct values). Do the whole draw in fp32 and cast
    // back, matching the CPU/fp32 reference distribution.
    if (self.scalar_type() == at::kBFloat16 || self.scalar_type() == at::kHalf) {
        auto tmp = at::empty(self.sizes(), self.options().dtype(at::kFloat));
        tmp.uniform_(0.0, 1.0, generator);
        tmp.neg_();
        tmp.add_(1.0);
        auto eps = std::numeric_limits<float>::epsilon() / 2;
        auto mask = tmp >= (1.0 - eps);
        tmp.masked_fill_(mask, 1.0 - eps);
        tmp.log_();
        tmp.mul_(-1.0 / lambd);
        self.copy_(tmp);
        return self;
    }

    self.uniform_(0.0, 1.0, generator);

    if (self.dtype() == at::kDouble) {
        self = op_api::sub_(self, at::Scalar(1.0), at::Scalar(1.0));
        self = op_api::mul_(self, at::Scalar(-1.0));
        self = op_api::log_(self);
        self = op_api::div_(self, at::Scalar(-lambd));

        auto eps = std::numeric_limits<double>::min();
        self = self.add(eps);
        return self;
    }

    self.neg_();
    self.add_(1.0);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                    self.scalar_type(), "exponential_", [&]() {
        auto eps = std::numeric_limits<scalar_t>::epsilon() / 2;
        auto mask = self >= (1.0 - eps);
        self.masked_fill_(mask, 1.0 - eps);
    });

    self.log_();
    self.mul_(-1.0 / lambd);
    return self;
}
}  // namespace op_api
