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
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include <algorithm>
#include <fstream>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void TORCH_NPU_API THDVM_init(PyObject *module) {
  using namespace dvm;
  auto torch_C_m = py::handle(module).cast<py::module>();
  pybind11::enum_<KernelType>(torch_C_m, "KernelType")
      .value("kStaticShape", KernelType::kStaticShape)
      .value("kDynShape", KernelType::kDynShape)
      .export_values();

  pybind11::enum_<BinaryOpType>(torch_C_m, "BinaryOpType")
      .value("kEqual", BinaryOpType::kEqual)
      .value("kNotEqual", BinaryOpType::kNotEqual)
      .value("kGreater", BinaryOpType::kGreater)
      .value("kGreaterEqual", BinaryOpType::kGreaterEqual)
      .value("kLess", BinaryOpType::kLess)
      .value("kLessEqual", BinaryOpType::kLessEqual)
      .value("kAdd", BinaryOpType::kAdd)
      .value("kSub", BinaryOpType::kSub)
      .value("kMul", BinaryOpType::kMul)
      .value("kDiv", BinaryOpType::kDiv)
      .value("kPow", BinaryOpType::kPow)
      .value("kMaximum", BinaryOpType::kMaximum)
      .value("kMinimum", BinaryOpType::kMinimum)
      .value("kLogicalAnd", BinaryOpType::kLogicalAnd)
      .value("kLogicalOr", BinaryOpType::kLogicalOr)
      .value("kBinaryOpEnd", BinaryOpType::kBinaryOpEnd)
      .export_values();

  pybind11::enum_<UnaryOpType>(torch_C_m, "UnaryOpType")
      .value("kSqrt", UnaryOpType::kSqrt)
      .value("kAbs", UnaryOpType::kAbs)
      .value("kLog", UnaryOpType::kLog)
      .value("kExp", UnaryOpType::kExp)
      .value("kReciprocal", UnaryOpType::kReciprocal)
      .value("kIsFinite", UnaryOpType::kIsFinite)
      .value("kLogicalNot", UnaryOpType::kLogicalNot)
      .value("kRound", UnaryOpType::kRound)
      .value("kFloor", UnaryOpType::kFloor)
      .value("kCeil", UnaryOpType::kCeil)
      .value("kTrunc", UnaryOpType::kTrunc)
      .value("kUnaryOpEnd", UnaryOpType::kUnaryOpEnd)
      .export_values();

  shared_ptr_class_<NDObjectPy>(torch_C_m, "NDObject");

  shared_ptr_class_<KernelPy>(torch_C_m, "Kernel")
      .def(py::init<const KernelType &>())
      .def("input_size", &KernelPy::GetInputSize, "load array")
      .def("output_size", &KernelPy::GetOutputSize, "load array")
      .def("set_name", &KernelPy::SetName, "load array")
      .def("load", &KernelPy::Load, "load array")
      .def("view_load", &KernelPy::ViewLoad, "load array")
      .def("slice_load", &KernelPy::SliceLoad, "load array")
      .def("stridedslice_load", &KernelPy::StridedSliceLoad, "load array")
      .def("store", &KernelPy::Store, "store array")
      .def("set_store_inplace", &KernelPy::SetStoreInplace, "store inplace")
      .def("unary", &KernelPy::Unary, "emit unary op")
      .def("cast", &KernelPy::Cast, "emit cast op")
      .def("element_any", &KernelPy::ElementAny, "emit element_any op")
      .def("binary", &KernelPy::Binary, "emit binary op")
      .def("select", &KernelPy::Select, "emit select op")
      .def("broadcast", &KernelPy::Broadcast, "emit broadcast op",
           py::arg("input"), py::arg("shape"))
      .def("broadcast_scalar", &KernelPy::BroadcastScalar, "emit broadcast op",
           py::arg("input"), py::arg("shape"), py::arg("dtype"))
      .def("reshape", &KernelPy::Reshape, "emit reshape op")
      .def("copy", &KernelPy::Copy, "emit copy op")
      .def("one_hot", &KernelPy::OneHot, "emit onehot op")
      .def("matmul", &KernelPy::MatMul, "emit matmul op", py::arg("lhs"),
           py::arg("rhs"), py::arg("trans_a"), py::arg("trans_b"),
           py::arg("bias") = py::none())
      .def("gmm", &KernelPy::GroupedMatMul, "emit grouped_matmul op",
           py::arg("lhs"), py::arg("rhs"), py::arg("trans_a"),
           py::arg("trans_b"), py::arg("bias"), py::arg("group_list"),
           py::arg("group_type"), py::arg("group_list_type") = 0)
      .def("p_next", &KernelPy::ParallelNext, "parallel next")
      .def("codegen", &KernelPy::CodeGen, "generate code")
      .def("das", &KernelPy::DisAssemble, "disassemble code")
      .def("dump", &KernelPy::DumpGraph, "dump graph")
      .def("run", &KernelPy::Run, "run kernel")
      .def_static("set_determ", &KernelPy::SetDeterm, "set deterministic")
      .def_static("set_online_tuning", &KernelPy::SetTuning,
                  "set online tuning");
}

namespace dvm {
namespace {
DType TorchDtype2DvmDType(at::ScalarType dtype) {
  switch (dtype) {
  case at::ScalarType::Float:
    return DType::kFloat32;
  case at::ScalarType::Half:
    return DType::kFloat16;
  case at::ScalarType::BFloat16:
    return DType::kBFloat16;
  case at::ScalarType::Int:
    return DType::kInt32;
  case at::ScalarType::Long:
    return DType::kInt64;
  case at::ScalarType::Bool:
    return DType::kBool;
  default:
    return DType::kFloat32;
  }
}

at::ScalarType DvmDType2TorchDtype(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return at::ScalarType::Float;
  case DType::kFloat16:
    return at::ScalarType::Half;
  case DType::kBFloat16:
    return at::ScalarType::BFloat16;
  case DType::kInt32:
    return at::ScalarType::Int;
  case DType::kInt64:
    return at::ScalarType::Long;
  case DType::kBool:
    return at::ScalarType::Bool;
  default:
    throw std::runtime_error("Unsupported Dtype conversion");
  }
}
} // namespace

KernelPy::KernelPy(const KernelType &kernel_type) {
  kernel_ = std::make_shared<Kernel>();
  kernel_->Reset(kernel_type);
}

KernelPy::~KernelPy() {}

ShapeRef *KernelPy::SymIntArraytoShapeRef(const at::SymIntArrayRef &shape) {
  auto symints = shape.vec();
  std::vector<int64_t> result;
  result.reserve(symints.size());
  for (const auto &symint : symints) {
    if (symint.is_symbolic()) {
      result.push_back(-1);
    } else {
      result.push_back(symint.expect_int());
    }
  }
  auto &shape_vec = shape_vec_.emplace_back(result);
  auto shape_ref = shape_.emplace_back(std::make_shared<ShapeRef>(shape_vec));

  return shape_ref.get();
}

py::object KernelPy::Unary(const UnaryOpType &op_type, py::object input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->Unary(op_type, in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Cast(py::object input, at::ScalarType dtype) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->Cast(in_obj, TorchDtype2DvmDType(dtype));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Select(py::object cond, py::object lhs, py::object rhs) {
  auto input1 = lhs.cast<NDOpPyPtr>()->Get();
  auto input2 = rhs.cast<NDOpPyPtr>()->Get();
  auto input0 = cond.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->Select(input0, input1, input2);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceMax(py::object input, const at::SymIntArrayRef &dims,
                               bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = SymIntArraytoShapeRef(dims);
  auto op = kernel_->Reduce(ReduceOpType::kMax, in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceSum(py::object input, const at::SymIntArrayRef &dims,
                               bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = SymIntArraytoShapeRef(dims);
  auto op = kernel_->Reduce(ReduceOpType::kSum, in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceMin(py::object input, const at::SymIntArrayRef &dims,
                               bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = SymIntArraytoShapeRef(dims);
  auto op = kernel_->Reduce(ReduceOpType::kMin, in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Binary(const BinaryOpType &op_type, py::object lhs,
                            py::object rhs) {
  NDObject *op;
  if (py::isinstance<py::int_>(lhs)) {
    op =
        kernel_->Binary(op_type, lhs.cast<int>(), rhs.cast<NDOpPyPtr>()->Get());
  } else if (py::isinstance<py::float_>(lhs)) {
    op = kernel_->Binary(op_type, lhs.cast<float>(),
                         rhs.cast<NDOpPyPtr>()->Get());
  } else if (py::isinstance<py::int_>(rhs)) {
    op =
        kernel_->Binary(op_type, lhs.cast<NDOpPyPtr>()->Get(), rhs.cast<int>());
  } else if (py::isinstance<py::float_>(rhs)) {
    op = kernel_->Binary(op_type, lhs.cast<NDOpPyPtr>()->Get(),
                         rhs.cast<float>());
  } else {
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    op = kernel_->Binary(op_type, input1, input2);
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Broadcast(py::object input,
                               const at::SymIntArrayRef &shape) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->Broadcast(in_obj, shape_ref);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::BroadcastScalar(c10::Scalar scalar,
                                     const at::SymIntArrayRef &shape,
                                     at::ScalarType dtype) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto type_id = TorchDtype2DvmDType(dtype);
  NDObject *op;
  if (scalar.isIntegral()) {
    op = kernel_->Broadcast(scalar.toInt(), shape_ref, type_id, empty_input_);
  } else {
    op = kernel_->Broadcast(scalar.toFloat(), shape_ref, type_id, empty_input_);
  }
  empty_input_ = empty_input_ && false;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Reshape(py::object input,
                             const at::SymIntArrayRef &shape) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto op = kernel_->Reshape(in_obj, shape_ref);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Copy(py::object input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->Copy(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::OneHot(py::object indices, int depth, int axis,
                            py::object on_value, py::object off_value,
                            at::ScalarType dtype) {
  auto indices_obj = indices.cast<NDOpPyPtr>()->Get();
  auto &shape_vec = shape_vec_.emplace_back(1, depth);
  auto depth_ref =
      shape_.emplace_back(std::make_shared<ShapeRef>(shape_vec)).get();
  auto type_id = TorchDtype2DvmDType(dtype);
  NDObject *op;
  if (type_id == kInt32) {
    op = kernel_->OneHot(indices_obj, depth_ref, axis,
                         py::cast<int32_t>(on_value),
                         py::cast<int32_t>(off_value));
  } else {
    auto on_float = py::cast<float>(on_value);
    auto off_float = py::cast<float>(off_value);
    if (type_id == kFloat16) {
      op = kernel_->OneHot(indices_obj, depth_ref, axis, Float16(on_float),
                           Float16(off_float));
    } else if (type_id == kBFloat16) {
      op = kernel_->OneHot(indices_obj, depth_ref, axis, BFloat16(on_float),
                           BFloat16(off_float));
    } else {
      op = kernel_->OneHot(indices_obj, depth_ref, axis, on_float, off_float);
    }
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Load(const at::SymIntArrayRef &shape,
                          at::ScalarType dtype) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto op = kernel_->Load(nullptr, shape_ref, TorchDtype2DvmDType(dtype));
  loads_.emplace_back(op);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ViewLoad(const at::SymIntArrayRef &shape,
                              const at::SymIntArrayRef &stride,
                              at::ScalarType dtype) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto stride_ref = SymIntArraytoShapeRef(stride);
  const int64_t *offset_ptr = nullptr;
  auto op = kernel_->Load(nullptr, shape_ref, stride_ref, offset_ptr,
                          TorchDtype2DvmDType(dtype));
  loads_.emplace_back(op);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::SliceLoad(const at::SymIntArrayRef &shape,
                               const at::SymIntArrayRef &start,
                               const at::SymIntArrayRef &size,
                               at::ScalarType dtype) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto start_ref = SymIntArraytoShapeRef(start);
  auto size_ref = SymIntArraytoShapeRef(size);
  auto op = kernel_->SliceLoad(nullptr, shape_ref, start_ref, size_ref,
                               TorchDtype2DvmDType(dtype));
  loads_.emplace_back(op);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::StridedSliceLoad(const at::SymIntArrayRef &shape,
                                      const at::SymIntArrayRef &start,
                                      const at::SymIntArrayRef &end,
                                      const at::SymIntArrayRef &step,
                                      at::ScalarType dtype) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto start_ref = SymIntArraytoShapeRef(start);
  auto end_ref = SymIntArraytoShapeRef(end);
  auto step_ref = SymIntArraytoShapeRef(step);
  auto op = kernel_->StridedSliceLoad(nullptr, shape_ref, start_ref, end_ref,
                                      step_ref, TorchDtype2DvmDType(dtype));
  loads_.emplace_back(op);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Store(py::object obj) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->Store(nullptr, in_obj);
  return py::cast(std::make_shared<NDObjectPy>(stores_.emplace_back(op)));
}

void KernelPy::SetStoreInplace(py::object obj) {
  auto store = obj.cast<NDOpPyPtr>()->Get();
  kernel_->SetStoreInplace(store);
}

py::object KernelPy::ElementAny(py::object input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->ElemAny(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::MatMul(py::object lhs, py::object rhs, bool trans_a,
                            bool trans_b, py::object bias) {
  auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
  auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
  auto op =
      kernel_->MatMul(lhs_obj, rhs_obj, trans_a, trans_b,
                      bias.is_none() ? nullptr : bias.cast<NDOpPyPtr>()->Get());
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::GroupedMatMul(py::object lhs, py::object rhs, bool trans_a,
                                   bool trans_b, py::object bias,
                                   py::object group_list, int64_t group_type,
                                   int64_t group_list_type) {
  auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
  auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
  NDObject *bias_obj = bias.is_none() ? nullptr : bias.cast<NDOpPyPtr>()->Get();
  NDObject *group_list_obj =
      group_list.is_none() ? nullptr : group_list.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->GroupedMatMul(lhs_obj, rhs_obj, trans_a, trans_b, bias_obj,
                                   group_list_obj, GroupType(group_type),
                                   (GroupListType)group_list_type);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

void KernelPy::ParallelNext() { kernel_->ParallelNext(); }

void KernelPy::CodeGen() { ws_size_ = kernel_->CodeGen(); }

std::string KernelPy::DisAssemble() { return kernel_->Das(); }

std::string KernelPy::DumpGraph() { return kernel_->Dump(); }

void KernelPy::Run(py::sequence inputs_tensor, py::sequence outputs_tensor) {
  // Todo: use share_form_this() and public dvm::Kernel
  std::vector<void *> inputs_addr(loads_.size());
  std::vector<void *> outputs_addr(stores_.size());
  std::vector<dvm::NDObject *> inputs = loads_;
  std::vector<dvm::NDObject *> outputs = stores_;
  for (size_t i = 0; i < inputs_addr.size(); ++i) {
    inputs_addr[i] = inputs_tensor[i].cast<at::Tensor>().data_ptr();
  }
  for (size_t i = 0; i < outputs_addr.size(); ++i) {
    outputs_addr[i] = outputs_tensor[i].cast<at::Tensor>().data_ptr();
  }
  auto kernel = kernel_;
  auto ws_size = ws_size_;
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
  auto dvm_call = [kernel, inputs_addr, outputs_addr, stream, inputs, outputs,
                   ws_size]() mutable -> int {
    dvm::RelocTable reloc_table;
    reloc_table.inputs_size = inputs.size();
    reloc_table.outputs_size = outputs.size();
    reloc_table.inputs = inputs.data();
    reloc_table.outputs = outputs.data();
    void *workspace_ptr = nullptr;
    at::Tensor workspace_tensor;
    if (ws_size != 0) {
      workspace_tensor = at_npu::native::allocate_workspace(ws_size, stream);
      workspace_ptr = const_cast<void *>(workspace_tensor.storage().data());
    }
    auto ret = kernel->Launch(reloc_table, inputs_addr.data(),
                              outputs_addr.data(), workspace_ptr, stream);

    return ret;
  };
  at_npu::native::OpCommand::RunOpApiV2(op_name_, dvm_call);
}

void KernelPy::Input(py::object load, const at::Tensor &array) {}

} // namespace dvm