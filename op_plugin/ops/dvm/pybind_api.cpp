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
      .def("sqrt", &KernelPy::Unary<UnaryOpType::kSqrt>, "emit sqrt")
      .def("abs", &KernelPy::Unary<UnaryOpType::kAbs>, "emit abs")
      .def("log", &KernelPy::Unary<UnaryOpType::kLog>, "emit log")
      .def("exp", &KernelPy::Unary<UnaryOpType::kExp>, "emit exp")
      .def("reciprocal", &KernelPy::Unary<UnaryOpType::kReciprocal>,
           "emit reciprocal")
      .def("isfinite", &KernelPy::Unary<UnaryOpType::kIsFinite>,
           "emit isfinite")
      .def("logical_not", &KernelPy::Unary<UnaryOpType::kLogicalNot>,
           "emit logical_not")
      .def("round", &KernelPy::Unary<UnaryOpType::kRound>, "emit round")
      .def("floor", &KernelPy::Unary<UnaryOpType::kFloor>, "emit floor")
      .def("ceil", &KernelPy::Unary<UnaryOpType::kCeil>, "emit ceil")
      .def("trunc", &KernelPy::Unary<UnaryOpType::kTrunc>, "emit trunc")
      .def("cast", &KernelPy::Cast, "emit cast op")
      .def("element_any", &KernelPy::ElementAny, "emit element_any op")
      .def("equal", &KernelPy::Binary<BinaryOpType::kEqual>, "emit equal")
      .def("not_equal", &KernelPy::Binary<BinaryOpType::kNotEqual>,
           "emit not_equal")
      .def("greater", &KernelPy::Binary<BinaryOpType::kGreater>, "emit greater")
      .def("greater_equal", &KernelPy::Binary<BinaryOpType::kGreaterEqual>,
           "emit greater_equal")
      .def("less", &KernelPy::Binary<BinaryOpType::kLess>, "emit less")
      .def("less_equal", &KernelPy::Binary<BinaryOpType::kLessEqual>,
           "emit less_equal")
      .def("add", &KernelPy::Binary<BinaryOpType::kAdd>, "emit add")
      .def("sub", &KernelPy::Binary<BinaryOpType::kSub>, "emit sub")
      .def("mul", &KernelPy::Binary<BinaryOpType::kMul>, "emit mul")
      .def("div", &KernelPy::Binary<BinaryOpType::kDiv>, "emit div")
      .def("pow", &KernelPy::Binary<BinaryOpType::kPow>, "emit pow")
      .def("maximum", &KernelPy::Binary<BinaryOpType::kMaximum>, "emit maximum")
      .def("minimum", &KernelPy::Binary<BinaryOpType::kMinimum>, "emit minimum")
      .def("logical_and", &KernelPy::Binary<BinaryOpType::kLogicalAnd>,
           "emit logical_and")
      .def("logical_or", &KernelPy::Binary<BinaryOpType::kLogicalOr>,
           "emit logical_or")
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

  pybind11::class_<GraphSplitKernelPy, KernelPy,
                   std::shared_ptr<GraphSplitKernelPy>>(torch_C_m,
                                                        "GraphSplitKernel")
      .def(py::init<bool>())
      .def("setup", &GraphSplitKernelPy::Setup, "setup")
      .def("__call__", &GraphSplitKernelPy::Call, "call kernel");
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
  kernel_.Reset(kernel_type);
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

template <UnaryOpType op_type> py::object KernelPy::Unary(py::object input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Unary(op_type, in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Cast(py::object input, at::ScalarType dtype) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Cast(in_obj, TorchDtype2DvmDType(dtype));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Select(py::object cond, py::object lhs, py::object rhs) {
  auto input1 = lhs.cast<NDOpPyPtr>()->Get();
  auto input2 = rhs.cast<NDOpPyPtr>()->Get();
  auto input0 = cond.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Select(input0, input1, input2);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceMax(py::object input, const at::SymIntArrayRef &dims,
                               bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = SymIntArraytoShapeRef(dims);
  auto op = kernel_.Reduce(ReduceOpType::kMax, in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceSum(py::object input, const at::SymIntArrayRef &dims,
                               bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = SymIntArraytoShapeRef(dims);
  auto op = kernel_.Reduce(ReduceOpType::kSum, in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ReduceMin(py::object input, const at::SymIntArrayRef &dims,
                               bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = SymIntArraytoShapeRef(dims);
  auto op = kernel_.Reduce(ReduceOpType::kMin, in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

template <BinaryOpType op_type>
py::object KernelPy::Binary(py::object lhs, py::object rhs) {
  NDObject *op;
  if (py::isinstance<py::int_>(lhs)) {
    op = kernel_.Binary(op_type, lhs.cast<int>(), rhs.cast<NDOpPyPtr>()->Get());
  } else if (py::isinstance<py::float_>(lhs)) {
    op = kernel_.Binary(op_type, lhs.cast<float>(),
                        rhs.cast<NDOpPyPtr>()->Get());
  } else if (py::isinstance<py::int_>(rhs)) {
    op = kernel_.Binary(op_type, lhs.cast<NDOpPyPtr>()->Get(), rhs.cast<int>());
  } else if (py::isinstance<py::float_>(rhs)) {
    op = kernel_.Binary(op_type, lhs.cast<NDOpPyPtr>()->Get(),
                        rhs.cast<float>());
  } else {
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    op = kernel_.Binary(op_type, input1, input2);
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Broadcast(py::object input,
                               const at::SymIntArrayRef &shape) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Broadcast(in_obj, shape_ref);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::BroadcastScalar(c10::Scalar scalar,
                                     const at::SymIntArrayRef &shape,
                                     at::ScalarType dtype) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto type_id = TorchDtype2DvmDType(dtype);
  NDObject *op;
  if (scalar.isIntegral()) {
    op = kernel_.Broadcast(scalar.toInt(), shape_ref, type_id, empty_input_);
  } else {
    op = kernel_.Broadcast(scalar.toFloat(), shape_ref, type_id, empty_input_);
  }
  empty_input_ = empty_input_ && false;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Reshape(py::object input,
                             const at::SymIntArrayRef &shape) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto op = kernel_.Reshape(in_obj, shape_ref);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Copy(py::object input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Copy(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::OneHot(py::object indices, int depth, int axis,
                            c10::Scalar on_value, c10::Scalar off_value,
                            at::ScalarType dtype) {
  auto indices_obj = indices.cast<NDOpPyPtr>()->Get();
  auto &shape_vec = shape_vec_.emplace_back(1, depth);
  auto depth_ref =
      shape_.emplace_back(std::make_shared<ShapeRef>(shape_vec)).get();
  auto type_id = TorchDtype2DvmDType(dtype);
  NDObject *op;
  if (type_id == kInt32) {
    op = kernel_.OneHot(indices_obj, depth_ref, axis, on_value.toInt(),
                        off_value.toInt());
  } else if (type_id == kFloat16) {
    op = kernel_.OneHot(indices_obj, depth_ref, axis, on_value.toHalf(),
                        off_value.toHalf());
  } else if (type_id == kBFloat16) {
    op = kernel_.OneHot(indices_obj, depth_ref, axis, on_value.toBFloat16(),
                        off_value.toBFloat16());
  } else {
    op = kernel_.OneHot(indices_obj, depth_ref, axis, on_value.toFloat(),
                        off_value.toFloat());
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Load(const at::SymIntArrayRef &shape,
                          at::ScalarType dtype) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto op = kernel_.Load(nullptr, shape_ref, TorchDtype2DvmDType(dtype));
  loads_.emplace_back(op);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ViewLoad(const at::SymIntArrayRef &shape,
                              const at::SymIntArrayRef &stride,
                              at::ScalarType dtype) {
  auto shape_ref = SymIntArraytoShapeRef(shape);
  auto stride_ref = SymIntArraytoShapeRef(stride);
  const int64_t *offset_ptr = nullptr;
  auto op = kernel_.Load(nullptr, shape_ref, stride_ref, offset_ptr,
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
  auto op = kernel_.SliceLoad(nullptr, shape_ref, start_ref, size_ref,
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
  auto op = kernel_.StridedSliceLoad(nullptr, shape_ref, start_ref, end_ref,
                                     step_ref, TorchDtype2DvmDType(dtype));
  loads_.emplace_back(op);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Store(py::object obj) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Store(nullptr, in_obj);
  return py::cast(std::make_shared<NDObjectPy>(stores_.emplace_back(op)));
}

void KernelPy::SetStoreInplace(py::object obj) {
  auto store = obj.cast<NDOpPyPtr>()->Get();
  kernel_.SetStoreInplace(store);
}

py::object KernelPy::ElementAny(py::object input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.ElemAny(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::MatMul(py::object lhs, py::object rhs, bool trans_a,
                            bool trans_b, py::object bias) {
  auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
  auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
  auto op =
      kernel_.MatMul(lhs_obj, rhs_obj, trans_a, trans_b,
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
  auto op = kernel_.GroupedMatMul(lhs_obj, rhs_obj, trans_a, trans_b, bias_obj,
                                  group_list_obj, GroupType(group_type),
                                  (GroupListType)group_list_type);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

void KernelPy::ParallelNext() { kernel_.ParallelNext(); }

void KernelPy::CodeGen() { ws_size_ = kernel_.CodeGen(); }

std::string KernelPy::DisAssemble() { return kernel_.Das(); }

std::string KernelPy::DumpGraph() { return kernel_.Dump(); }

void KernelPy::Run(py::sequence inputs_tensor, py::sequence outputs_tensor) {
  std::vector<void *> inputs_addr(loads_.size());
  std::vector<void *> outputs_addr(stores_.size());
  auto get_data_ptr = [](const at::Tensor &tensor) -> void * {
    return static_cast<char *>(tensor.data_ptr()) +
           tensor.itemsize() * tensor.storage_offset();
  };
  for (size_t i = 0; i < inputs_addr.size(); ++i) {
    inputs_addr[i] = get_data_ptr(inputs_tensor[i].cast<at::Tensor>());
  }
  for (size_t i = 0; i < outputs_addr.size(); ++i) {
    outputs_addr[i] = get_data_ptr(outputs_tensor[i].cast<at::Tensor>());
  }
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
  auto dvm_call = [this, inputs_addr, outputs_addr, stream]() mutable -> int {
    dvm::RelocTable reloc_table;
    reloc_table.inputs_size = loads_.size();
    reloc_table.outputs_size = stores_.size();
    reloc_table.inputs = loads_.data();
    reloc_table.outputs = stores_.data();
    void *workspace_ptr = nullptr;
    at::Tensor workspace_tensor;
    if (ws_size_ != 0) {
      workspace_tensor = at_npu::native::allocate_workspace(ws_size_, stream);
      workspace_ptr = const_cast<void *>(workspace_tensor.storage().data());
    }
    auto ret = kernel_.Launch(reloc_table, inputs_addr.data(),
                              outputs_addr.data(), workspace_ptr, stream);

    return ret;
  };
  at_npu::native::OpCommand::RunOpApiV2(op_name_, dvm_call);
}

static void *GraphSplitWsAlloc(uint64_t size, void *user_data) {
  return static_cast<GraphSplitKernelPy *>(user_data)->AllocWorkspace(size);
}

void GraphSplitKernelPy::Setup() {
  relocs_.resize(loads_.size() + stores_.size());
  RelocEntry *reloc = relocs_.data();
  for (auto op : loads_) {
    reloc->io = op;
    reloc->addr = nullptr;
    reloc++;
  }
  for (auto op : stores_) {
    reloc->io = op;
    reloc->addr = nullptr;
    reloc++;
  }
  kernel_.SetWsAlloc(GraphSplitWsAlloc, this);
}

py::object GraphSplitKernelPy::Call(py::args inputs) {
  kernel_.Infer();
  AT_ASSERT(inputs.size() == loads_.size());
  static auto byte_dtype =
      c10::scalarTypeToTypeMeta(dtype_or_default(at::kByte));
  options_ = inputs[0].cast<at::Tensor>().options().dtype(byte_dtype);
  RelocEntry *reloc = relocs_.data();
  for (auto arg : inputs) {
    reloc->addr = arg.cast<at::Tensor>().data_ptr();
    reloc++;
  }
  py::object ret;
  auto create_output = [this](NDObject *store) -> at::Tensor {
    auto shape_ref = kernel_.GetShape(store);
    c10::IntArrayRef shape = c10::IntArrayRef(shape_ref->data, shape_ref->size);
    at::ScalarType dtype = DvmDType2TorchDtype(kernel_.GetDType(store));
    return at_npu::native::OpPreparation::apply_tensor_without_format(
        shape, options_.dtype(dtype));
  };
  if (stores_.size() == 1) {
    at::Tensor tensor = create_output(stores_.front());
    reloc->addr = tensor.data_ptr();
    ret = py::cast(tensor);
  } else {
    py::tuple tuple_ret(stores_.size());
    for (size_t i = 0; i < stores_.size(); ++i) {
      at::Tensor tensor = create_output(stores_[i]);
      reloc->addr = tensor.data_ptr();
      reloc++;
      tuple_ret[i] = py::cast(tensor);
    }
    ret = tuple_ret;
  }
  kernel_.CodeGen(relocs_.data(), relocs_.size());
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto launch_call = [this, stream]() -> int { return kernel_.Launch(stream); };
  at_npu::native::OpCommand::RunOpApiV2(op_name_, launch_call);
  wss_.clear();
  return ret;
}
} // namespace dvm