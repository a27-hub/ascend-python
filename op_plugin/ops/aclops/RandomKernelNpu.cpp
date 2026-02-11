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

#include <limits>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <climits>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/AclOpsInterface.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using npu_compile_type = at_npu::native::CompileType;

namespace {
// RANDOM_DOUBLE_MAX = 1 << 53
const int64_t RANDOM_DOUBLE_MAX = 9007199254740992;
const int64_t RANDOM_HALF_MAX = 1 << 11;
const int64_t RANDOM_FLOAT_MAX = 1 << 24;

at::Tensor& random_out_npu(
    at::Tensor& result,
    at::Tensor& self,
    int64_t from,
    int64_t to,
    c10::optional<at::Generator> gen) {
  auto gen_val = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_val->philox_engine_inputs(10);
  const int64_t seed = static_cast<int64_t>(pair.first);
  const int64_t offset = static_cast<int64_t>(pair.second);
  at::SmallVector<int64_t, N> key = {seed};
  at::SmallVector<int64_t, N> counter = {0, offset};
  const int32_t alg = 1;
  at_npu::native::OpCommand cmd;
  cmd.Name("StatelessRandomUniformV2")
      .Input(self.sizes(), at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(key, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(counter, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(at::Scalar(alg), at::ScalarType::Int);
  // StatelessRandomUniformV2 doesn't support int output
  if (isIntegralType(self.scalar_type(), true)) {
    at::Tensor result_cp = npu_preparation::apply_tensor(self, self.options().dtype(at::kFloat));
    cmd.Attr("dtype", at::kFloat)
        .Output(result_cp)
        .Run();
    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    result_cp = result_cp.mul(to).sub(result_cp.mul(from).sub(static_cast<float>(from)));
    result_cp = at_npu::native::custom_ops::_npu_dtype_cast(result_cp, self.scalar_type());
    result.copy_(result_cp);
  } else {
    cmd.Attr("dtype", self.scalar_type())
        .Output(result)
        .Run();
    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    auto result_cp = result.mul(to).sub(result.mul(from).sub(static_cast<float>(from)));
    // round off numbers
    result_cp = at_npu::native::custom_ops::_npu_dtype_cast(result_cp, at::kLong);
    result_cp = at_npu::native::custom_ops::_npu_dtype_cast(result_cp, self.scalar_type());
    result.copy_(result_cp);
  }
  return result;
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
    TORCH_CHECK((var) >= (min) && (var) <= (max), (name), " is out of bounds for ", (dtype));

#define WARN_OUT_OF_BOUNDS(var, name, digits, dtype) \
    if ((var) < -(1LL << (digits)) || (var) > (1LL << (digits))) { \
        TORCH_WARN((name), " is out of bounds [-(2^", (digits), "), 2^", (digits), "]. ", \
            "Due to precision limitations ", (dtype), " can support discrete uniform distribution only within this range. ", \
            "This warning will become an error in next release, please fix the code in advance"); \
    }

void check_random_bounds(caffe2::TypeMeta dtype, int64_t from, int64_t to)
{
    TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);

    const auto scalar_type = c10::typeMetaToScalarType(dtype);
    if (at::isFloatingType(scalar_type)) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "check_random_fp_bounds", [&] {
            const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
            const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
            CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
            CHECK_OUT_OF_BOUNDS((to - 1), "to - 1", min, max, dtype);

            constexpr auto digits = std::numeric_limits<scalar_t>::digits;
            WARN_OUT_OF_BOUNDS(from, "from", digits, dtype);
            WARN_OUT_OF_BOUNDS((to - 1), "to - 1", digits, dtype);
        });
    } else if (scalar_type == at::kUInt64) {
        const auto min = 0;
        const auto max = std::numeric_limits<int64_t>::max();
        CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
        CHECK_OUT_OF_BOUNDS((to - 1), "to - 1", min, max, dtype);
    } else if (at::isIntegralType(scalar_type, true)) {
        if (scalar_type == at::kBool) {
            const auto min = 0;
            const auto max = 1;
            CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
            CHECK_OUT_OF_BOUNDS((to - 1), "to - 1", min, max, dtype);
        } else {
            AT_DISPATCH_ALL_TYPES_AND2(at::kUInt16, at::kUInt32, scalar_type, "check_random_integral_bounds", [&] {
                if constexpr (std::is_integral_v<scalar_t>) {
                    const auto min = static_cast<int64_t>(std::numeric_limits<scalar_t>::lowest());
                    const auto max = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
                    CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
                    CHECK_OUT_OF_BOUNDS((to - 1), "to - 1", min, max, dtype);
                }
            });
        }
    } else {
        TORCH_CHECK(false, "check_random_bounds handles only integral, floating-point and boolean types");
    }
}

int64_t get_max(const caffe2::TypeMeta dtype)
{
  if (dtype == at::kHalf) { return RANDOM_HALF_MAX + 1; }
  if (dtype == at::kFloat) { return RANDOM_FLOAT_MAX + 1; }
  if (dtype == at::kDouble) { return RANDOM_DOUBLE_MAX + 1; }
  if (dtype == at::kInt) { return INT_MAX; }
  if (dtype == at::kShort) { return SHRT_MAX + 1; }
  if (dtype == at::kChar) { return SCHAR_MAX + 1; }
  if (dtype == at::kByte) { return UCHAR_MAX + 1; }
  if (dtype == at::kLong) { return LONG_MAX; }
  return 1;
}
} // namespace

at::Tensor& random_npu_(at::Tensor& self, int64_t from, int64_t to, c10::optional<at::Generator> gen) {
    check_random_bounds(self.dtype(), from, to);
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    random_out_npu(contiguous_self, contiguous_self, from, to, gen);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    random_out_npu(self, self, from, to, gen);
  }
  return self;
}

at::Tensor& random_(
    at::Tensor& self, int64_t from,
    c10::optional<int64_t> to,
    c10::optional<at::Generator> gen) {
  int64_t to_val = to.has_value() ? to.value() : get_max(self.dtype());
  random_npu_(self, from, to_val, gen);
  return self;
}

at::Tensor& random_(at::Tensor& self, int64_t to, c10::optional<at::Generator> gen) {
  random_npu_(self, 0, to, gen);
  return self;
}

at::Tensor& random_(at::Tensor& self, c10::optional<at::Generator> gen) {
  random_npu_(self, 0, get_max(self.dtype()), gen);

  return self;
}
} // namespace acl_op
