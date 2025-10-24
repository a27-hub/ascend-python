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

#include "op_plugin/ops/dvm/pybind_api.h"
#include <fstream>

namespace dvm {
namespace {
std::vector<int64_t> GetVector(const py::object &shape) {
  py::list shape_list = py::cast<py::list>(shape);
  size_t size = shape_list.size();
  std::vector<int64_t> shape_vec(size);
  for (size_t i = 0; i < size; ++i) {
    shape_vec[i] = py::cast<int64_t>(shape_list[i]);
  }
  return shape_vec;
}

template <typename T> std::pair<bool, T> GetScalar(const py::object &obj) {
  if (py::isinstance<py::int_>(obj)) {
    return {true, static_cast<T>(py::cast<int64_t>(obj))};
  } else if (py::isinstance<py::float_>(obj)) {
    return {true, py::cast<T>(obj)};
  }
  return {false, (T)0};
}

template <typename T> T &FindVectorInfo(std::vector<T> &infos, NDObject *op) {
  for (auto &info : infos) {
    if (info.op == op) {
      return info;
    }
  }
  return infos.front();
}

} // namespace

KernelPy::KernelPy(const std::string &ker_type, ) {}

KernelPy::~KernelPy() {
  for (auto ref : shape_) {
    delete ref;
  }
}

py::object KernelPy::Unary(const std::string &op_name,
                           const py::object &input) {
  int op_type = UnaryOp::QueryId(op_name);
  if (op_type < 0) {
    std::string err_msg = "Could not find op: " + op_name;
    throw std::invalid_argument(err_msg);
  }
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Unary(op_type, in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Cast(const py::object &input, const std::string &type) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Cast(in_obj, StringToTypeID(type));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Select(const py::object &cond, const py::object &lhs,
                            const py::object &rhs) {
  auto input1 = lhs.cast<NDOpPyPtr>()->Get();
  auto input2 = rhs.cast<NDOpPyPtr>()->Get();
  auto input0 = cond.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Select(input0, input1, input2);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceMax(const py::object &input, const py::object &dims,
                               bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = GetShapeRef(dims);
  auto op = kernel_.Reduce(reduce_map[type], in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceSum(const py::object &input, const py::object &dims,
                               bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = GetShapeRef(dims);
  auto op = kernel_.Reduce(reduce_map[type], in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceMin(const py::object &input, const py::object &dims,
                               bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = GetShapeRef(dims);
  auto op = kernel_.Reduce(reduce_map[type], in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Binary(const std::string &op_name, const py::object &lhs,
                            const py::object &rhs) {
  int op_type = BinaryOp::QueryId(op_name);
  if (op_type < 0) {
    std::string err_msg = "Could not find op: " + op_name;
    throw std::invalid_argument(err_msg);
  }
  NDObject *op;
  auto [lhs_is_scalar, lhs_scalar] = GetScalar<float>(lhs);
  auto [rhs_is_scalar, rhs_scalar] = GetScalar<float>(rhs);
  if (lhs_is_scalar) {
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    if (kernel_.GetDType(input2) == dvm::kInt32) {
      op = kernel_.Binary(op_type, GetScalar<int>(lhs).second, input2);
    } else {
      op = kernel_.Binary(op_type, lhs_scalar, input2);
    }
  } else if (rhs_is_scalar) {
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    if (kernel_.GetDType(input1) == dvm::kInt32) {
      op = kernel_.Binary(op_type, input1, GetScalar<int>(rhs).second);
    } else {
      op = kernel_.Binary(op_type, input1, rhs_scalar);
    }
  } else {
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    op = kernel_.Binary(op_type, input1, input2);
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Broadcast(const py::object &input, const py::object &shape,
                               const std::string &dtype, bool dummy_load) {
  auto shape_ref = GetShapeRef(shape);
  NDObject *op;
  auto [is_scalar, scalar] = GetScalar<float>(input);
  if (is_scalar) {
    auto type_id = StringToTypeID(dtype);
    op = kernel_.Broadcast(scalar, shape_ref, type_id, dummy_load);
  } else {
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    op = kernel_.Broadcast(in_obj, shape_ref);
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Reshape(const py::object &input, const py::object &shape) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto shape_ref = GetShapeRef(shape);
  auto op = kernel_.Reshape(in_obj, shape_ref);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Copy(const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Copy(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::OneHot(const py::object &indices, int depth, int axis,
                            const py::object &on_value,
                            const py::object &off_value,
                            const std::string &dtype) {
  auto indices_obj = indices.cast<NDOpPyPtr>()->Get();
  auto depth_ref =
      shape_.emplace_back(new ShapeRef(shape_vec_.emplace_back(1, depth)));
  auto type_id = StringToTypeID(dtype);
  NDObject *op;
  if (type_id == kInt32) {
    op = kernel_.OneHot(indices_obj, depth_ref, axis,
                        py::cast<int32_t>(on_value),
                        py::cast<int32_t>(off_value));
  } else {
    auto on_float = py::cast<float>(on_value);
    auto off_float = py::cast<float>(off_value);
    if (type_id == kFloat16) {
      op = kernel_.OneHot(indices_obj, depth_ref, axis, Float16(on_float),
                          Float16(off_float));
    } else if (type_id == kBFloat16) {
      op = kernel_.OneHot(indices_obj, depth_ref, axis, BFloat16(on_float),
                          BFloat16(off_float));
    } else {
      op = kernel_.OneHot(indices_obj, depth_ref, axis, on_float, off_float);
    }
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Load(const py::object &shape, const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto op = kernel_.Load(nullptr, shape_ref, StringToTypeID(type));
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ViewLoad(const py::object &shape, const py::object &stride,
                              int64_t offset, const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto stride_ref = GetShapeRef(stride);
  const int64_t *offset_ptr = nullptr;
  if (offset != 0) {
    auto &offset_vec = shape_vec_.emplace_back(1, offset);
    offset_ptr = &offset_vec[0];
  }
  auto op = kernel_.Load(nullptr, shape_ref, stride_ref, offset_ptr,
                         StringToTypeID(type));
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::SliceLoad(const py::object &shape, const py::object &start,
                               const py::object &size,
                               const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto start_ref = GetShapeRef(start);
  auto size_ref = GetShapeRef(size);
  auto op = kernel_.SliceLoad(nullptr, shape_ref, start_ref, size_ref,
                              StringToTypeID(type));
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::StridedSliceLoad(const py::object &shape,
                                      const py::object &start,
                                      const py::object &end,
                                      const py::object &step,
                                      const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto start_ref = GetShapeRef(start);
  auto end_ref = GetShapeRef(end);
  auto step_ref = GetShapeRef(step);
  auto op = kernel_.StridedSliceLoad(nullptr, shape_ref, start_ref, end_ref,
                                     step_ref, StringToTypeID(type));
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::MultiLoad(const py::object &shape,
                               const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto op =
      kernel_.MultiLoad(nullptr, shape_ref, StringToTypeID(type), &g_mpc.comm);
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Store(const py::object &obj) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Store(nullptr, in_obj);
  auto &store = stores_.emplace_back();
  store.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::PadStore(const py::object &obj, int64_t pad_size) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.PadStore(nullptr, in_obj, pad_size);
  auto &store = stores_.emplace_back();
  store.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

void KernelPy::SetStoreInplace(const py::object &obj) {
  auto store = obj.cast<NDOpPyPtr>()->Get();
  kernel_.SetStoreInplace(store);
}

py::object KernelPy::ElementAny(const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.ElemAny(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::AllReduce(const std::string &type,
                               const py::object &input) {
  auto input_obj = input.cast<NDOpPyPtr>()->Get();
  ReduceOpType reduce_type{ReduceOpType::kReduceOpEnd};
  if (type == "sum") {
    reduce_type = ReduceOpType::kSum;
  } else if (type == "max") {
    reduce_type = ReduceOpType::kMax;
  } else {
    throw py::value_error("Unsupported AllReduce type: " + type);
  }
  auto op = kernel_.AllReduce(reduce_type, input_obj, &g_mpc.comm);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceScatter(const py::object &input) {
  auto input_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.ReduceScatter(input_obj, &g_mpc.comm);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::AllGather(const py::object &input) {
  auto input_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.AllGather(input_obj, &g_mpc.comm);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::AllGatherV2(const py::object &input) {
  auto input_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.AllGatherV2(input_obj, &g_mpc.comm);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::MatMul(const py::object &lhs, const py::object &rhs,
                            bool trans_a, bool trans_b,
                            const py::object &bias) {
  auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
  auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
  auto op =
      kernel_.MatMul(lhs_obj, rhs_obj, trans_a, trans_b,
                     bias.is_none() ? nullptr : bias.cast<NDOpPyPtr>()->Get());
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::GroupedMatMul(const py::object &lhs, const py::object &rhs,
                                   bool trans_a, bool trans_b,
                                   const py::object &bias,
                                   const py::object &group_list,
                                   int64_t group_type,
                                   int64_t group_list_type) {
  auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
  auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
  NDObject *bias_obj = bias.is_none() ? nullptr : bias.cast<NDOpPyPtr>()->Get();
  NDObject *group_list_obj =
      group_list.is_none() ? nullptr : group_list.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.GroupedMatMul(lhs_obj, rhs_obj, trans_a, trans_b, bias_obj,
                                  group_list_obj, dvm::GroupType(group_type),
                                  (GroupListType)group_list_type);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

void KernelPy::ParallelNext() { kernel_.ParallelNext(); }

void KernelPy::CodeGen(const py::object &pass_names) {}

py::object KernelPy::DisAssemble() {
  std::string data = kernel_->DisAssemble();
  return py::cast(data);
}

py::object KernelPy::DumpGraph() {
  std::string data = kernel_->DumpGraph();
  return py::cast(data);
}

void KernelPy::Run() {}

void KernelPy::Input(const py::object &load, const py::object &array) {
  // auto op = static_cast<NDAccess *>(load.cast<NDOpPyPtr>()->Get());
  // auto &info = FindVectorInfo(loads_, op);
  // auto input = py::array(array);
  // py::buffer_info buf = input.request();
  // runner_->AllocLoad(buf, info);
  // if (kernel_.GetImpl()->KType() == kDynShape ||
  //     kernel_.GetImpl()->KType() == kDynMix) {
  //   info.shape.resize(buf.ndim);
  //   for (size_t i = 0; i < static_cast<size_t>(buf.ndim); ++i) {
  //     info.shape[i] = buf.shape[i];
  //   }
  //   *(op->shape_ref_) = info.shape;
  // }
}

PYBIND11_MODULE(_dvm_py, m) {
  (void)py::class_<NDObjectPy, std::shared_ptr<NDObjectPy>>(m, "NDObject")
      .def("shape", &NDObjectPy::GetShape, "get shape")
      .def("dtype", &NDObjectPy::GetDType, "get dtype");

  (void)py::class_<HandlePy, std::shared_ptr<HandlePy>>(m, "NDHandle");

  (void)py::class_<KernelPy, std::shared_ptr<KernelPy>>(m, "Kernel")
      .def(py::init<const std::string &, const std::string &, int>())
      .def("load", &KernelPy::Load, "load array")
      .def("view_load", &KernelPy::ViewLoad, "load array")
      .def("slice_load", &KernelPy::SliceLoad, "load array")
      .def("stridedslice_load", &KernelPy::StridedSliceLoad, "load array")
      .def("multi_load", &KernelPy::MultiLoad, "load array(for reducescatter)")
      .def("store", &KernelPy::Store, "store array")
      .def("pad_store", &KernelPy::PadStore, "pad store array")
      .def("set_store_inplace", &KernelPy::SetStoreInplace, "store inplace")
      .def("unary", &KernelPy::Unary, "emit unary op")
      .def("cast", &KernelPy::Cast, "emit cast op")
      .def("element_any", &KernelPy::ElementAny, "emit element_any op")
      .def("binary", &KernelPy::Binary, "emit binary op")
      .def("select", &KernelPy::Select, "emit select op")
      .def("broadcast", &KernelPy::Broadcast, "emit broadcast op",
           py::arg("input"), py::arg("shape"), py::arg("dtype") = "float32",
           py::arg("dummy_load") = true)
      .def("reshape", &KernelPy::Reshape, "emit reshape op")
      .def("reduce", &KernelPy::Reduce, "emit reduce op")
      .def("copy", &KernelPy::Copy, "emit copy op")
      .def("one_hot", &KernelPy::OneHot, "emit onehot op")
      .def("allreduce", &KernelPy::AllReduce, "emit allreduce op")
      .def("allgather", &KernelPy::AllGather, "emit allgather op")
      .def("allgatherv2", &KernelPy::AllGatherV2, "emit allgatherv2 op")
      .def("reducescatter", &KernelPy::ReduceScatter, "emit reducescatter op")
      .def("matmul", &KernelPy::MatMul, "emit matmul op", py::arg("lhs"),
           py::arg("rhs"), py::arg("trans_a"), py::arg("trans_b"),
           py::arg("bias") = py::none())
      .def("gmm", &KernelPy::GroupedMatMul, "emit grouped_matmul op",
           py::arg("lhs"), py::arg("rhs"), py::arg("trans_a"),
           py::arg("trans_b"), py::arg("bias"), py::arg("group_list"),
           py::arg("group_type"), py::arg("group_list_type") = 0)
      .def("p_next", &KernelPy::ParallelNext, "parallel next")
      .def("stage_switch", &KernelPy::StageSwitch, "stage switch")
      .def("stage_load", &KernelPy::StageLoad, "stage load")
      .def("stage_store", &KernelPy::StageStore, "stage store")
      .def("new_stage", &KernelPy::NewStage, "add stage")
      .def("schedule", &KernelPy::GetSchedule, "get schedule")
      .def("reset", &KernelPy::Reset, "reset eager")
      .def("input", &KernelPy::Input, "get ouput array")
      .def("output", &KernelPy::Output, "get ouput array")
      .def("codegen", &KernelPy::CodeGen, "generate code")
      .def("das", &KernelPy::DisAssemble, "disassemble code")
      .def("dump", &KernelPy::DumpGraph, "dump graph")
      .def("perf", &KernelPy::Perf, "perf test")
      .def("msprof", &KernelPy::Msprof, "perf test")
      .def("run", &KernelPy::Run, "run kernel")
      .def_static("set_determ", &KernelPy::SetDeterm, "set deterministic")
      .def_static("set_online_tuning", &KernelPy::SetTuning,
                  "set online tuning")
}
} // namespace dvm
