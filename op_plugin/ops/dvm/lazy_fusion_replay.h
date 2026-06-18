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

#ifndef DVM_LAZY_FUSION_REPLAY_H
#define DVM_LAZY_FUSION_REPLAY_H

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>

namespace lazy_fusion {

// A recorded op for accuracy-check replay. The op's tensor inputs are stored as
// references (kept alive by the fusion graph) and resolved at Flush time via
// data_ptr matching against the cloned input snapshot or prior op outputs.
struct ReplayOp {
  std::string name;
  // Tensor inputs in call order. Each entry is the original tensor reference
  // (not yet cloned). At replay time, the resolver maps each to either a cloned
  // input or a prior op's reference output.
  std::vector<at::Tensor> tensor_inputs;
  // Optional tensor inputs (preserving position via tensor_inputs with
  // undefined entries is ambiguous, so store separately keyed by call index).
  std::vector<c10::optional<at::Tensor>> opt_tensor_inputs;
  // Scalar inputs in call order.
  std::vector<at::Scalar> scalar_inputs;
  // Integer inputs (dim, keepdim, etc.) in call order.
  std::vector<int64_t> int_inputs;
  // Boolean inputs in call order (stored as int 0/1 for simplicity).
  std::vector<int64_t> bool_inputs;
  // Double inputs (momentum, eps) in call order.
  std::vector<double> double_inputs;
  // String inputs (approximate, etc.) in call order.
  std::vector<std::string> str_inputs;
  // TensorList inputs: each inner vector is one TensorList argument. The outer
  // vector preserves the order of TensorList arguments in the call.
  std::vector<std::vector<at::Tensor>> tensorlist_inputs;
  // Scalar array inputs (e.g. foreach_div_(tensors, scalars)).
  std::vector<std::vector<at::Scalar>> scalar_array_inputs;
  // Number of outputs this op produces (matches outputs_ slice length).
  size_t output_num{0};
  // data_ptr of each output tensor (from outputs_ tensor_meta), used to build
  // the output_map so subsequent ops can resolve these outputs as their inputs.
  std::vector<void *> output_data_ptrs;
};

// Resolve a tensor input to the corresponding reference-implementation tensor.
// `input_map` maps original input data_ptr -> cloned tensor.
// `output_map` maps prior op output data_ptr -> reference output tensor.
// Returns the resolved tensor (or undefined if not found).
inline at::Tensor ResolveTensor(const at::Tensor &t,
                                const std::unordered_map<void *, at::Tensor> &input_map,
                                const std::unordered_map<void *, at::Tensor> &output_map) {
  if (!t.defined()) {
    return t;
  }
  void *ptr = t.data_ptr();
  auto it_in = input_map.find(ptr);
  if (it_in != input_map.end()) {
    return it_in->second;
  }
  auto it_out = output_map.find(ptr);
  if (it_out != output_map.end()) {
    return it_out->second;
  }
  // External constant / CPU scalar: clone it for isolation.
  return t.clone();
}

// Registry mapping op name -> replay function.
// A replay function takes the ReplayOp plus the input/output maps and returns
// the reference outputs (from op_api).
using ReplayFunc = std::function<std::vector<at::Tensor>(
    const ReplayOp &, const std::unordered_map<void *, at::Tensor> &,
    const std::unordered_map<void *, at::Tensor> &)>;

class ReplayRegistry {
 public:
  static ReplayRegistry &Instance() {
    static ReplayRegistry inst;
    return inst;
  }
  void Register(const std::string &name, ReplayFunc func) { funcs_[name] = std::move(func); }
  ReplayFunc Get(const std::string &name) const {
    auto it = funcs_.find(name);
    return it != funcs_.end() ? it->second : nullptr;
  }

 private:
  std::unordered_map<std::string, ReplayFunc> funcs_;
};

// Convenience macro to register a replay function at static init time.
#define REGISTER_LAZY_FUSION_REPLAY(name, func) \
  static bool reg_lazy_fusion_replay_##name = [] { \
    ::lazy_fusion::ReplayRegistry::Instance().Register(#name, func); \
    return true; \
  }()

}  // namespace lazy_fusion

#endif  // DVM_LAZY_FUSION_REPLAY_H
