// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/ops/dvm/lazy_fusion_replay.h"

#include <ATen/ATen.h>

#include "op_plugin/OpApiInterface.h"

namespace lazy_fusion {
namespace {

using TensorMap = std::unordered_map<void *, at::Tensor>;

// Helper: resolve the i-th tensor input of an op.
inline at::Tensor TIn(const ReplayOp &op, size_t i, const TensorMap &in_map, const TensorMap &out_map) {
  return ResolveTensor(op.tensor_inputs.at(i), in_map, out_map);
}

// ===================== Cast =====================
std::vector<at::Tensor> ReplayCast(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  auto self = TIn(op, 0, in_map, out_map);
  auto dtype = static_cast<at::ScalarType>(op.int_inputs.at(0));
  return {op_api::_npu_dtype_cast(self, dtype)};
}
REGISTER_LAZY_FUSION_REPLAY(cast, ReplayCast);

// ===================== Unary =====================
std::vector<at::Tensor> ReplayAbs(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::abs(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(abs, ReplayAbs);

std::vector<at::Tensor> ReplayNeg(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::neg(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(neg, ReplayNeg);

std::vector<at::Tensor> ReplaySqrt(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::sqrt(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(sqrt, ReplaySqrt);

std::vector<at::Tensor> ReplayExp(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::exp(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(exp, ReplayExp);

std::vector<at::Tensor> ReplayExpInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  // exp_ is inplace; reference impl produces a fresh tensor.
  return {op_api::exp(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(exp_, ReplayExpInplace);

std::vector<at::Tensor> ReplayReciprocal(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::reciprocal(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(reciprocal, ReplayReciprocal);

// ===================== Binary: add/sub/mul/div/pow =====================
// add(self, other: Tensor, alpha)
std::vector<at::Tensor> ReplayAddTensor(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::add(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(add, ReplayAddTensor);

// add(self, other: Scalar, alpha) — recorded with the same op name "add".
// The replay function inspects tensor_inputs size: if only 1 tensor, it's the
// scalar overload. We register a single dispatcher under "add".
std::vector<at::Tensor> ReplayAddDispatch(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::add(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map), op.scalar_inputs.at(0))};
  }
  // scalar overload: add(self, scalar, alpha)
  return {op_api::add(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0), op.scalar_inputs.at(1))};
}
// Override the earlier registration.
struct AddDispatchRegister {
  AddDispatchRegister() { ReplayRegistry::Instance().Register("add", ReplayAddDispatch); }
} g_add_dispatch_register;

std::vector<at::Tensor> ReplayAddInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  // add_ is inplace; reference produces a fresh tensor.
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::add(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map), op.scalar_inputs.at(0))};
  }
  return {op_api::add(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0), op.scalar_inputs.at(1))};
}
REGISTER_LAZY_FUSION_REPLAY(add_, ReplayAddInplace);

std::vector<at::Tensor> ReplaySub(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::sub(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map), op.scalar_inputs.at(0))};
  }
  return {op_api::sub(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0), op.scalar_inputs.at(1))};
}
REGISTER_LAZY_FUSION_REPLAY(sub, ReplaySub);

std::vector<at::Tensor> ReplaySubInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::sub(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map), op.scalar_inputs.at(0))};
  }
  return {op_api::sub(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0), op.scalar_inputs.at(1))};
}
REGISTER_LAZY_FUSION_REPLAY(sub_, ReplaySubInplace);

std::vector<at::Tensor> ReplayMul(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::mul(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::mul(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(mul, ReplayMul);

std::vector<at::Tensor> ReplayMulInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::mul(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::mul(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(mul_, ReplayMulInplace);

std::vector<at::Tensor> ReplayDiv(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::div(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::div(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(div, ReplayDiv);

std::vector<at::Tensor> ReplayDivInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::div(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::div(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(div_, ReplayDivInplace);

std::vector<at::Tensor> ReplayPow(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::pow(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::pow(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(pow, ReplayPow);

std::vector<at::Tensor> ReplayPowInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::pow(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::pow(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(pow_, ReplayPowInplace);

// ===================== FloorDivide =====================
std::vector<at::Tensor> ReplayFloorDivide(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::floor_divide(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::floor_divide(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(floor_divide, ReplayFloorDivide);

std::vector<at::Tensor> ReplayFloorDivideInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::floor_divide(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::floor_divide(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(floor_divide_, ReplayFloorDivideInplace);

std::vector<at::Tensor> ReplayFloorDivideOut(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::floor_divide(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::floor_divide(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(floor_divide_out, ReplayFloorDivideOut);

// ===================== Comparison =====================
std::vector<at::Tensor> ReplayEq(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::eq(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::eq(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(eq, ReplayEq);

std::vector<at::Tensor> ReplayNe(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::ne(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::ne(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(ne, ReplayNe);

std::vector<at::Tensor> ReplayGt(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::gt(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::gt(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(gt, ReplayGt);

std::vector<at::Tensor> ReplayGe(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::ge(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::ge(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(ge, ReplayGe);

std::vector<at::Tensor> ReplayLt(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::lt(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::lt(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(lt, ReplayLt);

std::vector<at::Tensor> ReplayLe(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  if (op.tensor_inputs.size() >= 2) {
    return {op_api::le(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
  }
  return {op_api::le(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(le, ReplayLe);

std::vector<at::Tensor> ReplayMaximum(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::maximum(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(maximum, ReplayMaximum);

std::vector<at::Tensor> ReplayMinimum(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::minimum(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(minimum, ReplayMinimum);

// ===================== Where =====================
std::vector<at::Tensor> ReplayWhere(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::where(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map), TIn(op, 2, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(where, ReplayWhere);

std::vector<at::Tensor> ReplayWhereOut(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::where(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map), TIn(op, 2, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(where_out, ReplayWhereOut);

// ===================== Activation =====================
std::vector<at::Tensor> ReplaySigmoid(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::sigmoid(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(sigmoid, ReplaySigmoid);

std::vector<at::Tensor> ReplaySigmoidBackward(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::sigmoid_backward(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(sigmoid_backward, ReplaySigmoidBackward);

std::vector<at::Tensor> ReplayTanh(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::tanh(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(tanh, ReplayTanh);

std::vector<at::Tensor> ReplayTanhInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::tanh(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(tanh_, ReplayTanhInplace);

std::vector<at::Tensor> ReplayTanhBackward(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::tanh_backward(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(tanh_backward, ReplayTanhBackward);

std::vector<at::Tensor> ReplayGelu(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  auto self = TIn(op, 0, in_map, out_map);
  auto approximate = op.str_inputs.empty() ? std::string("none") : op.str_inputs.at(0);
  return {op_api::gelu(self, approximate)};
}
REGISTER_LAZY_FUSION_REPLAY(gelu, ReplayGelu);

std::vector<at::Tensor> ReplayGeluBackward(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  auto grad = TIn(op, 0, in_map, out_map);
  auto self = TIn(op, 1, in_map, out_map);
  auto approximate = op.str_inputs.empty() ? std::string("none") : op.str_inputs.at(0);
  return {op_api::gelu_backward(grad, self, approximate)};
}
REGISTER_LAZY_FUSION_REPLAY(gelu_backward, ReplayGeluBackward);

std::vector<at::Tensor> ReplayRelu(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::relu(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(relu, ReplayRelu);

std::vector<at::Tensor> ReplayReluInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::relu(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(relu_, ReplayReluInplace);

std::vector<at::Tensor> ReplayLeakyRelu(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::leaky_relu(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(leaky_relu, ReplayLeakyRelu);

std::vector<at::Tensor> ReplayLeakyReluInplace(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::leaky_relu(TIn(op, 0, in_map, out_map), op.scalar_inputs.at(0))};
}
REGISTER_LAZY_FUSION_REPLAY(leaky_relu_, ReplayLeakyReluInplace);

std::vector<at::Tensor> ReplaySilu(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::silu(TIn(op, 0, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(silu, ReplaySilu);

std::vector<at::Tensor> ReplaySiluBackward(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::silu_backward(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(silu_backward, ReplaySiluBackward);

// ===================== Reduce: sum =====================
std::vector<at::Tensor> ReplaySum(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  auto self = TIn(op, 0, in_map, out_map);
  // Two overloads share the name "sum":
  //   1) sum(self, dtype)                    — DumpOp("sum", self, dtype)
  //   2) sum(self, dim, keepdim, dtype)      — DumpOp("sum", self, dim, keepdim, dtype)
  // Recording maps:
  //   overload 1: bool_inputs=[] ; int_inputs=[dtype] (or [] if dtype nullopt)
  //   overload 2: bool_inputs=[keepdim] ; int_inputs=[dim0, dim1, ..., dtype]
  //               (dim may be empty if OptionalIntArrayRef is nullopt; dtype is
  //                always appended as the last int_input, Undefined if nullopt)
  if (op.bool_inputs.empty()) {
    // overload 1: sum(self, dtype)
    c10::optional<at::ScalarType> dtype = c10::nullopt;
    if (!op.int_inputs.empty()) {
      auto v = op.int_inputs.at(0);
      if (v != static_cast<int64_t>(at::ScalarType::Undefined)) {
        dtype = static_cast<at::ScalarType>(v);
      }
    }
    return {op_api::sum(self, dtype)};
  }
  // overload 2: sum(self, dim, keepdim, dtype)
  // The last int_input is dtype; the rest are dim values.
  bool keepdim = static_cast<bool>(op.bool_inputs.at(0));
  c10::optional<at::ScalarType> dtype = c10::nullopt;
  std::vector<int64_t> dim_vec;
  if (!op.int_inputs.empty()) {
    // Last element is dtype.
    auto dtype_val = op.int_inputs.back();
    if (dtype_val != static_cast<int64_t>(at::ScalarType::Undefined)) {
      dtype = static_cast<at::ScalarType>(dtype_val);
    }
    // Remaining elements are dim values.
    if (op.int_inputs.size() > 1) {
      dim_vec.assign(op.int_inputs.begin(), op.int_inputs.end() - 1);
    }
  }
  at::OptionalIntArrayRef dim = c10::nullopt;
  if (!dim_vec.empty()) {
    dim = at::IntArrayRef(dim_vec);
  }
  return {op_api::sum(self, dim, keepdim, dtype)};
}
REGISTER_LAZY_FUSION_REPLAY(sum, ReplaySum);

std::vector<at::Tensor> ReplaySumOut(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  auto self = TIn(op, 0, in_map, out_map);
  // sum_out(self, dim, keepdim, dtype, out) — DumpOp("sum_out", self, dim, keepdim, dtype)
  // Recording: bool_inputs=[keepdim] ; int_inputs=[dim0, ..., dtype]
  bool keepdim = op.bool_inputs.empty() ? false : static_cast<bool>(op.bool_inputs.at(0));
  c10::optional<at::ScalarType> dtype = c10::nullopt;
  std::vector<int64_t> dim_vec;
  if (!op.int_inputs.empty()) {
    auto dtype_val = op.int_inputs.back();
    if (dtype_val != static_cast<int64_t>(at::ScalarType::Undefined)) {
      dtype = static_cast<at::ScalarType>(dtype_val);
    }
    if (op.int_inputs.size() > 1) {
      dim_vec.assign(op.int_inputs.begin(), op.int_inputs.end() - 1);
    }
  }
  at::OptionalIntArrayRef dim = c10::nullopt;
  if (!dim_vec.empty()) {
    dim = at::IntArrayRef(dim_vec);
  }
  return {op_api::sum(self, dim, keepdim, dtype)};
}
REGISTER_LAZY_FUSION_REPLAY(sum_out, ReplaySumOut);

// ===================== MatMul =====================
std::vector<at::Tensor> ReplayMatMul(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::matmul(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(matmul, ReplayMatMul);

std::vector<at::Tensor> ReplayMm(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::mm(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(mm, ReplayMm);

std::vector<at::Tensor> ReplayBmm(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  return {op_api::bmm(TIn(op, 0, in_map, out_map), TIn(op, 1, in_map, out_map))};
}
REGISTER_LAZY_FUSION_REPLAY(bmm, ReplayBmm);

std::vector<at::Tensor> ReplayAddmm(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  auto self = TIn(op, 0, in_map, out_map);
  auto mat1 = TIn(op, 1, in_map, out_map);
  auto mat2 = TIn(op, 2, in_map, out_map);
  auto beta = op.scalar_inputs.at(0);
  auto alpha = op.scalar_inputs.at(1);
  return {op_api::addmm(self, mat1, mat2, beta, alpha)};
}
REGISTER_LAZY_FUSION_REPLAY(addmm, ReplayAddmm);

// ===================== Foreach =====================
std::vector<at::Tensor> ReplayForeachSqrt(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  // _foreach_sqrt returns new tensors (not inplace). tensorlist_inputs[0] is the input list.
  std::vector<at::Tensor> resolved;
  if (op.tensorlist_inputs.empty()) {
    return {};
  }
  for (const auto &t : op.tensorlist_inputs.at(0)) {
    resolved.push_back(ResolveTensor(t, in_map, out_map));
  }
  return op_api::_foreach_sqrt(at::TensorList(resolved));
}
REGISTER_LAZY_FUSION_REPLAY(_foreach_sqrt, ReplayForeachSqrt);

std::vector<at::Tensor> ReplayForeachSqrtInplace(const ReplayOp &op, const TensorMap &in_map,
                                                  const TensorMap &out_map) {
  if (op.tensorlist_inputs.empty()) {
    return {};
  }
  std::vector<at::Tensor> resolved;
  for (const auto &t : op.tensorlist_inputs.at(0)) {
    resolved.push_back(ResolveTensor(t, in_map, out_map));
  }
  auto out = op_api::_foreach_sqrt(at::TensorList(resolved));
  // Copy back to inplace targets so the fused output (in original tensors) can be compared.
  for (size_t i = 0; i < resolved.size() && i < out.size(); ++i) {
    resolved[i].copy_(out[i]);
  }
  return out;
}
REGISTER_LAZY_FUSION_REPLAY(_foreach_sqrt_, ReplayForeachSqrtInplace);

std::vector<at::Tensor> ReplayForeachMulInplace(const ReplayOp &op, const TensorMap &in_map,
                                                 const TensorMap &out_map) {
  if (op.tensorlist_inputs.empty()) {
    return {};
  }
  std::vector<at::Tensor> resolved;
  for (const auto &t : op.tensorlist_inputs.at(0)) {
    resolved.push_back(ResolveTensor(t, in_map, out_map));
  }
  op_api::_foreach_mul_(at::TensorList(resolved), op.scalar_inputs.at(0));
  // Return clones so comparison uses the reference results (which are now in resolved).
  std::vector<at::Tensor> out;
  out.reserve(resolved.size());
  for (auto &t : resolved) {
    out.push_back(t.clone());
  }
  return out;
}
REGISTER_LAZY_FUSION_REPLAY(_foreach_mul_, ReplayForeachMulInplace);

std::vector<at::Tensor> ReplayForeachAddInplace(const ReplayOp &op, const TensorMap &in_map,
                                                 const TensorMap &out_map) {
  if (op.tensorlist_inputs.empty()) {
    return {};
  }
  std::vector<at::Tensor> resolved;
  for (const auto &t : op.tensorlist_inputs.at(0)) {
    resolved.push_back(ResolveTensor(t, in_map, out_map));
  }
  op_api::_foreach_add_(at::TensorList(resolved), op.scalar_inputs.at(0));
  std::vector<at::Tensor> out;
  out.reserve(resolved.size());
  for (auto &t : resolved) {
    out.push_back(t.clone());
  }
  return out;
}
REGISTER_LAZY_FUSION_REPLAY(_foreach_add_, ReplayForeachAddInplace);

std::vector<at::Tensor> ReplayForeachDivInplace(const ReplayOp &op, const TensorMap &in_map,
                                                 const TensorMap &out_map) {
  if (op.tensorlist_inputs.empty()) {
    return {};
  }
  std::vector<at::Tensor> resolved;
  for (const auto &t : op.tensorlist_inputs.at(0)) {
    resolved.push_back(ResolveTensor(t, in_map, out_map));
  }
  if (!op.scalar_array_inputs.empty()) {
    op_api::_foreach_div_(at::TensorList(resolved), at::ArrayRef<at::Scalar>(op.scalar_array_inputs.at(0)));
  } else {
    op_api::_foreach_div_(at::TensorList(resolved), op.scalar_inputs.at(0));
  }
  std::vector<at::Tensor> out;
  out.reserve(resolved.size());
  for (auto &t : resolved) {
    out.push_back(t.clone());
  }
  return out;
}
REGISTER_LAZY_FUSION_REPLAY(_foreach_div_, ReplayForeachDivInplace);

std::vector<at::Tensor> ReplayForeachAddcmulInplace(const ReplayOp &op, const TensorMap &in_map,
                                                     const TensorMap &out_map) {
  if (op.tensorlist_inputs.size() < 3) {
    return {};
  }
  std::vector<at::Tensor> a, b, c;
  for (const auto &t : op.tensorlist_inputs.at(0)) a.push_back(ResolveTensor(t, in_map, out_map));
  for (const auto &t : op.tensorlist_inputs.at(1)) b.push_back(ResolveTensor(t, in_map, out_map));
  for (const auto &t : op.tensorlist_inputs.at(2)) c.push_back(ResolveTensor(t, in_map, out_map));
  op_api::_foreach_addcmul_(at::TensorList(a), at::TensorList(b), at::TensorList(c), op.scalar_inputs.at(0));
  std::vector<at::Tensor> out;
  out.reserve(a.size());
  for (auto &t : a) out.push_back(t.clone());
  return out;
}
REGISTER_LAZY_FUSION_REPLAY(_foreach_addcmul_, ReplayForeachAddcmulInplace);

std::vector<at::Tensor> ReplayForeachAddcdivInplace(const ReplayOp &op, const TensorMap &in_map,
                                                     const TensorMap &out_map) {
  if (op.tensorlist_inputs.size() < 3) {
    return {};
  }
  std::vector<at::Tensor> a, b, c;
  for (const auto &t : op.tensorlist_inputs.at(0)) a.push_back(ResolveTensor(t, in_map, out_map));
  for (const auto &t : op.tensorlist_inputs.at(1)) b.push_back(ResolveTensor(t, in_map, out_map));
  for (const auto &t : op.tensorlist_inputs.at(2)) c.push_back(ResolveTensor(t, in_map, out_map));
  if (!op.scalar_array_inputs.empty()) {
    op_api::_foreach_addcdiv_(at::TensorList(a), at::TensorList(b), at::TensorList(c),
                              at::ArrayRef<at::Scalar>(op.scalar_array_inputs.at(0)));
  } else {
    op_api::_foreach_addcdiv_(at::TensorList(a), at::TensorList(b), at::TensorList(c), op.scalar_inputs.at(0));
  }
  std::vector<at::Tensor> out;
  out.reserve(a.size());
  for (auto &t : a) out.push_back(t.clone());
  return out;
}
REGISTER_LAZY_FUSION_REPLAY(_foreach_addcdiv_, ReplayForeachAddcdivInplace);

// ===================== BatchNorm =====================
std::vector<at::Tensor> ReplayNativeBatchNorm(const ReplayOp &op, const TensorMap &in_map,
                                              const TensorMap &out_map) {
  auto input = TIn(op, 0, in_map, out_map);
  c10::optional<at::Tensor> weight = op.opt_tensor_inputs.size() > 0 ? op.opt_tensor_inputs.at(0) : c10::nullopt;
  c10::optional<at::Tensor> bias = op.opt_tensor_inputs.size() > 1 ? op.opt_tensor_inputs.at(1) : c10::nullopt;
  c10::optional<at::Tensor> running_mean = op.opt_tensor_inputs.size() > 2 ? op.opt_tensor_inputs.at(2) : c10::nullopt;
  c10::optional<at::Tensor> running_var = op.opt_tensor_inputs.size() > 3 ? op.opt_tensor_inputs.at(3) : c10::nullopt;
  bool training = static_cast<bool>(op.bool_inputs.at(0));
  double momentum = op.double_inputs.at(0);
  double eps = op.double_inputs.at(1);
  auto result = op_api::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
}
REGISTER_LAZY_FUSION_REPLAY(native_batch_norm, ReplayNativeBatchNorm);

std::vector<at::Tensor> ReplayNativeBatchNormBackward(const ReplayOp &op, const TensorMap &in_map,
                                                       const TensorMap &out_map) {
  auto grad_out = TIn(op, 0, in_map, out_map);
  auto input = TIn(op, 1, in_map, out_map);
  c10::optional<at::Tensor> weight = op.opt_tensor_inputs.size() > 0 ? op.opt_tensor_inputs.at(0) : c10::nullopt;
  c10::optional<at::Tensor> running_mean = op.opt_tensor_inputs.size() > 1 ? op.opt_tensor_inputs.at(1) : c10::nullopt;
  c10::optional<at::Tensor> running_var = op.opt_tensor_inputs.size() > 2 ? op.opt_tensor_inputs.at(2) : c10::nullopt;
  c10::optional<at::Tensor> save_mean = op.opt_tensor_inputs.size() > 3 ? op.opt_tensor_inputs.at(3) : c10::nullopt;
  c10::optional<at::Tensor> save_invstd = op.opt_tensor_inputs.size() > 4 ? op.opt_tensor_inputs.at(4) : c10::nullopt;
  bool train = static_cast<bool>(op.bool_inputs.at(0));
  double eps = op.double_inputs.at(0);
  ::std::array<bool, 3> output_mask = {static_cast<bool>(op.bool_inputs.at(1)),
                                        static_cast<bool>(op.bool_inputs.at(2)),
                                        static_cast<bool>(op.bool_inputs.at(3))};
  auto result = op_api::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean,
                                                    save_invstd, train, eps, output_mask);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
}
REGISTER_LAZY_FUSION_REPLAY(native_batch_norm_backward, ReplayNativeBatchNormBackward);

// ===================== SwiGLU =====================
std::vector<at::Tensor> ReplayNpuSwiglu(const ReplayOp &op, const TensorMap &in_map, const TensorMap &out_map) {
  auto self = TIn(op, 0, in_map, out_map);
  int64_t dim = op.int_inputs.at(0);
  return {op_api::npu_swiglu(self, dim)};
}
REGISTER_LAZY_FUSION_REPLAY(npu_swiglu, ReplayNpuSwiglu);

}  // namespace
}  // namespace lazy_fusion
