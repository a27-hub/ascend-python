// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#ifndef _DVM_PYBIND_API_H_
#define _DVM_PYBIND_API_H_
#include <memory>
#include <vector>
#include "op_plugin/ops/dvm/dvm.h"
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace dvm {
class NDObjectPy {
public:
  explicit NDObjectPy(NDObject *obj) : obj_(obj) {}
  py::object GetShape() { return py::none(); }
  py::object GetDType() { return py::none(); }
  NDObject *Get() const { return obj_; }

private:
  NDObject *obj_;
};
using NDOpPyPtr = std::shared_ptr<NDObjectPy>;
using ShapeRefPtr = std::shared_ptr<ShapeRef>;

class KernelPy {
public:
  KernelPy(const KernelType &kernel_type);
  ~KernelPy();

  py::object Load(const at::SymIntArrayRef &shape, at::ScalarType dtype);
  py::object ViewLoad(const at::SymIntArrayRef &shape,
                      const at::SymIntArrayRef &stride, at::ScalarType dtype);
  py::object SliceLoad(const at::SymIntArrayRef &shape,
                       const at::SymIntArrayRef &start,
                       const at::SymIntArrayRef &size, at::ScalarType dtype);
  py::object StridedSliceLoad(const at::SymIntArrayRef &shape,
                              const at::SymIntArrayRef &start,
                              const at::SymIntArrayRef &end,
                              const at::SymIntArrayRef &step,
                              at::ScalarType dtype);
  py::object Store(py::object obj);
  void SetStoreInplace(py::object store);
  template<UnaryOpType op_type>
  py::object Unary(py::object input);
  template<BinaryOpType op_type>
  py::object Binary(py::object lhs, py::object rhs);
  py::object Broadcast(py::object input, const at::SymIntArrayRef &shape);
  py::object BroadcastScalar(c10::Scalar scalar,
                             const at::SymIntArrayRef &shape,
                             at::ScalarType dtype);
  py::object Reshape(py::object input, const at::SymIntArrayRef &shape);
  py::object Cast(py::object input, at::ScalarType dtype);
  py::object ReduceMax(py::object input, const at::SymIntArrayRef &dims,
                       bool keepdims);
  py::object ReduceMin(py::object input, const at::SymIntArrayRef &dims,
                       bool keepdims);
  py::object ReduceSum(py::object input, const at::SymIntArrayRef &dims,
                       bool keepdims);
  py::object Select(py::object cond, py::object lhs, py::object rhs);
  py::object ElementAny(py::object input);
  py::object Copy(py::object input);
  py::object OneHot(py::object indices, int depth, int axis,
                    c10::Scalar on_value, c10::Scalar off_value,
                    at::ScalarType dtype);
  py::object MatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b,
                    py::object bias);
  py::object GroupedMatMul(py::object lhs, py::object rhs, bool trans_a,
                           bool trans_b, py::object bias, py::object group_list,
                           int64_t group_type, int64_t group_list_type);
  void ParallelNext();
  int GetInputSize() { return loads_.size(); }
  int GetOutputSize() { return stores_.size(); }
  void SetName(const std::string &op_name, const std::string &op_fullname) {
    op_name_ = op_name;
    op_fullname_ = op_fullname;
    kernel_.SetOpName(op_name_.c_str(), op_fullname_.c_str());
  }

  void Reset();
  void CodeGen();
  void Run(py::sequence inputs_tensor, py::sequence outputs_tensor);

  std::string DisAssemble();
  std::string DumpGraph();

  static void SetDeterm(bool enable) { SetDeterministic(enable); }
  static void SetTuning(bool enable) { SetOnlineTuning(enable); }
  ShapeRef *SymIntArraytoShapeRef(const at::SymIntArrayRef &shape);

protected:
  Kernel kernel_;
  std::vector<std::vector<int64_t>> shape_vec_;
  std::vector<ShapeRefPtr> shape_;
  std::vector<NDObject *> loads_;
  std::vector<NDObject *> stores_;
  size_t ws_size_;
  std::string op_name_;
  std::string op_fullname_;
  bool empty_input_{true};
};

class GraphSplitKernelPy : public KernelPy {
 public:
  GraphSplitKernelPy(bool dyn_shape) : KernelPy(dyn_shape ? KernelType::kDynSplit : KernelType::kStaticSplit) {} 

  void Setup();
  py::object Call(py::args inputs);

  void *AllocWorkspace(uint64_t size) {
    int64_t size_data = size;
    c10::IntArrayRef shape = c10::IntArrayRef(&size_data, 1);
    at::Tensor tensor = at_npu::native::OpPreparation::apply_tensor_without_format(shape, options_);
    wss_.push_back(tensor);
    return tensor.data_ptr();
  }

 protected:
  std::vector<RelocEntry> relocs_;
  std::vector<at::Tensor> wss_;
  c10::TensorOptions options_;
};
} // namespace dvm
#endif // _DVM_PYBIND_API_H_
