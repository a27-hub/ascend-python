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
#include "op_plugin/ops/dvm/dvm.h"
#include <memory>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/python_headers.h>
#include <vector>

namespace dvm {
using KernelPtr = std::share_ptr<Kernel>;
using NDOpPyPtr = std::shared_ptr<NDObjectPy>;

class NDObjectPy {
public:
  explicit NDObjectPy(NDObject *obj) : obj_(obj) {}
  py::object GetShape() { return py::none; }
  std::string GetDType() { return py::none; }
  NDObject *Get() const { return obj_; }

private:
  NDObject *obj_;
};



class KernelPy {
public:
  KernelPy(const std::string &ker_type);
  ~KernelPy();

  py::object Load(const py::object &shape, const std::string &type);
  py::object ViewLoad(const py::object &shape, const py::object &stride,
                      int64_t offset, const std::string &type);
  py::object SliceLoad(const py::object &shape, const py::object &start,
                       const py::object &size, const std::string &type);
  py::object StridedSliceLoad(const py::object &shape, const py::object &start,
                              const py::object &end, const py::object &step,
                              const std::string &type);
  py::object MultiLoad(const py::object &shape, const std::string &type);
  py::object Store(const py::object &obj);
  py::object PadStore(const py::object &obj, int64_t pad_shape);
  void SetStoreInplace(const py::object &store);
  py::object Unary(const std::string &op_name, const py::object &input);
  py::object Binary(const std::string &op_name, const py::object &lhs,
                    const py::object &rhs);
  py::object Broadcast(const py::object &input, const py::object &shape,
                       const std::string &dtype, bool dummy_load);
  py::object Reshape(const py::object &input, const py::object &shape);
  py::object Cast(const py::object &input, const std::string &type);
  py::object Reduce(const std::string &type, const py::object &input,
                    const py::object &dims, bool keepdims);
  py::object Select(const py::object &cond, const py::object &lhs,
                    const py::object &rhs);
  py::object ElementAny(const py::object &input);
  py::object Copy(const py::object &input);
  py::object OneHot(const py::object &indices, int depth, int axis,
                    const py::object &on_value, const py::object &off_value,
                    const std::string &dtype);
  py::object AllReduce(const std::string &type, const py::object &input);
  py::object AllGather(const py::object &input);
  py::object AllGatherV2(const py::object &input);
  py::object ReduceScatter(const py::object &input);
  py::object MatMul(const py::object &lhs, const py::object &rhs, bool trans_a,
                    bool trans_b, const py::object &bias);
  py::object GroupedMatMul(const py::object &lhs, const py::object &rhs,
                           bool trans_a, bool trans_b, const py::object &bias,
                           const py::object &group_list, int64_t group_type,
                           int64_t group_list_type);
  void ParallelNext();

  void Reset();

  void Input(const py::object &load, const py::object &array);
  py::object Output(const py::object &store);
  void CodeGen(const py::object &pass_names);
  void Run();

  py::object DisAssemble();
  py::object DumpGraph();

  static void SetDeterm(bool enable) { SetDeterministic(enable); }
  static void SetTuning(bool enable) { SetOnlineTuning(enable); }

  struct LoadInfo {
    NDObject *op;
    std::vector<int64_t> shape;
  };

  struct StoreInfo {
    NDObject *op;
  };

protected:
  ShapeRef *GetShapeRef(const py::object &shape);
  void PrepareIO();

  KernelPtr kernel_;
  std::vector<std::vector<int64_t>> shape_vec_;
  std::vector<ShapeRef *> shape_;
  std::vector<LoadInfo> loads_;
  std::vector<StoreInfo> stores_;

  void *workspace_{nullptr};
};
} // namespace dvm
#endif // _DVM_PYBIND_API_H_
