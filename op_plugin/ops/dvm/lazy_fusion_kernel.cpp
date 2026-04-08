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

#include "op_plugin/ops/dvm/lazy_fusion_kernel.h"
#include <linux/limits.h>
#include <sys/stat.h>
#include <fstream>
#include <ATen/record_function.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"

namespace lazy_fusion {
// dump_as_text is fixed at process startup; cache as static const so the
// per-op check folds into a single global load + branch (predicted not-taken).
static inline bool DumpEnabled() {
  static const bool v = g_lazy_fusion_manager.flags_.dump_as_text;
  return v;
}
namespace {
class LazyFusionDump {
 public:
  static LazyFusionDump &Instance() {
    static LazyFusionDump instance;
    return instance;
  }

  void DumpGraphInfo(std::stringstream &buf) {
    std::string file_name = "lazy_fusion_" + std::to_string(getpid()) + "_graph.txt";
    std::string file_path = dump_dir_ + "/" + file_name;
    DumpToFile(file_path, buf);
  }

  void DumpKernelInfo(std::stringstream &buf) {
    std::string file_name = "lazy_fusion_" + std::to_string(getpid()) + "_kernel.txt";
    std::string file_path = dump_dir_ + "/" + file_name;
    DumpToFile(file_path, buf);
  }

 private:
  void DumpToFile(const std::string &file_path, std::stringstream &buf) {
    if (!enable_dump_) {
      return;
    }
    ChangeFileMode(file_path, S_IWUSR);
    std::ofstream of(file_path, std::ios::app);
    if (!of.is_open()) {
      ASCEND_LOGW("Open dump file '%s' failed!", file_path.c_str());
      ChangeFileMode(file_path, S_IRUSR);
      return;
    }
    of << buf.str() << "\n";
    of.close();
    ChangeFileMode(file_path, S_IRUSR);
    buf.str("");
  }
  LazyFusionDump() { CreateDumpDir(); }
  ~LazyFusionDump() = default;

  void ChangeFileMode(const std::string &file_name, mode_t mode) {
    if (access(file_name.c_str(), F_OK) == -1) {
      return;
    }
    try {
      if (chmod(file_name.c_str(), mode) != 0) {
        ASCEND_LOGW("Change file '%s' to mode %d fail", file_name.c_str(), mode);
      }
    } catch (std::exception &e) {
      ASCEND_LOGW("File '%s' change mode failed! May be not exist.", file_name.c_str())
    }
  }

  bool IsFileExist(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    return access(path.c_str(), F_OK) == 0;
  }

  bool IsDir(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    struct stat st = {0};
    int ret = lstat(path.c_str(), &st);
    if (ret != 0) {
      return false;
    }
    return S_ISDIR(st.st_mode);
  }

  bool CreateDir(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    if (IsFileExist(path)) {
      return IsDir(path);
    }
    size_t pos = 0;
    while ((pos = path.find_first_of('/', pos)) != std::string::npos) {
      std::string base_dir = path.substr(0, ++pos);
      if (IsFileExist(base_dir)) {
        if (IsDir(base_dir)) {
          continue;
        } else {
          return false;
        }
      }
      if (mkdir(base_dir.c_str(), 0750) != 0) {
        return false;
      }
    }
    return mkdir(path.c_str(), 0750) == 0;
  }

  void CreateDumpDir() {
    if (!g_lazy_fusion_manager.flags_.dump_as_text) {
      return;
    }
    enable_dump_ = CreateDir(g_lazy_fusion_manager.flags_.dump_dir);
    if (!enable_dump_) {
      ASCEND_LOGW("Failed to create dump directory: %s", g_lazy_fusion_manager.flags_.dump_dir.c_str());
      return;
    }
    auto path = g_lazy_fusion_manager.flags_.dump_dir.c_str();
    char real_path[PATH_MAX] = {0};
    if (strlen(path) >= PATH_MAX || realpath(path, real_path) == nullptr) {
      ASCEND_LOGW("Get realpath failed, path: %s", path);
      enable_dump_ = false;
      return;
    }
    dump_dir_ = real_path;
    ASCEND_LOGW("dvm dump directory: %s", dump_dir_.c_str());
  }

  bool enable_dump_{false};
  std::string dump_dir_;
};

bool EqualShape(dvm::ShapeRef *shape1, c10::IntArrayRef shape2) {
  if (shape1->size != shape2.size()) {
    return false;
  }
  for (size_t i = 0; i < shape1->size; ++i) {
    if (shape1->data[i] != shape2[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace

Manager g_lazy_fusion_manager;

Manager::~Manager() {
  while (!pool_.empty()) {
    auto top = pool_.front();
    delete top;
    pool_.pop();
  }
}

LazyFusionKernel *Manager::NewKernel() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!pool_.empty()) {
      auto k = pool_.front();
      pool_.pop();
      return k;
    }
  }
  return new LazyFusionKernel();
}

LazyFusionKernel::LazyFusionKernel() {
  dvm::Kernel::Reset(dvm::KernelType::kEager, dvm::KernelFlag::kUnifyWS);
  inputs_.reserve(32);
  outputs_.reserve(16);
}

LazyFusionKernel::~LazyFusionKernel() {
  for (auto load : inputs_) {
    delete load;
  }
  for (auto s : cached_shape_) {
    delete s;
  }
  for (auto op : ops_map_) {
    delete op;
  }
}

void LazyFusionKernel::CacheDvmOp(torch_npu::NPUStorageImpl *storage, dvm::NDObject *obj, int64_t storage_offset) {
  if (ops_map_used_ == ops_map_.size()) {
    ops_map_.push_back(new DvmOp());
  }
  auto p = ops_map_[ops_map_used_++];
  p->op = obj;
  p->storage_offset = storage_offset;
  storage->dvm_op_ = p;
}

dvm::NDObject *LazyFusionKernel::Input(const at::Tensor &x, bool enable_cast, dvm::ShapeRef *shape) {
  auto input_type = TransType(x.scalar_type());
  bool cast_to_fp32 = (enable_cast && input_type == dvm::DType::kBFloat16);
  auto storage = torch_npu::NPUBridge::GetNpuStorageImpl(x);
  if (storage->dvm_op_ != nullptr) {
    auto dvm_op = static_cast<DvmOp *>(storage->dvm_op_);
    auto op = dvm_op->op;
    if (dvm_op->storage_offset == x.storage_offset() && EqualShape(dvm::Kernel::GetShape(op), x.sizes())) {
      if (cast_to_fp32) {
        op = Cast(op, dvm::DType::kFloat32);
      }
      return op;
    }
  }
  if (input_used_ == inputs_.size()) {
    inputs_.push_back(new Load());
  }
  auto load = inputs_[input_used_++];
  if (shape != nullptr) {
    load->shape = *shape;
  } else {
    auto ptr = GetShapeRef(x.sizes());
    load->shape = *ptr;
  }
  load->data_ptr = x.data_ptr();
  load->tensor = x;
  auto load_op = dvm::Kernel::Load(load->data_ptr, &(load->shape), input_type);
  if (DumpEnabled()) {
    dump_buf_ << "p" << (input_used_ - 1) << ": " << ToString(x) << "\n";
  }
  CacheDvmOp(storage, load_op, x.storage_offset());
  return cast_to_fp32 ? Cast(load_op, dvm::DType::kFloat32) : load_op;
}

void LazyFusionKernel::Output(const at::Tensor &tensor, dvm::NDObject *obj, bool inplace) {
  auto tensor_type = TransType(tensor.scalar_type());
  if (dvm::Kernel::GetDType(obj) != tensor_type) {
    obj = Cast(obj, tensor_type);
  }
  auto storage = torch_npu::NPUBridge::GetNpuStorageImpl(tensor);
  auto data_ptr = tensor.data_ptr();
  if (inplace && storage->dvm_op_ != nullptr) {
    auto nbytes = tensor.nbytes();
    for (int64_t i = static_cast<int64_t>(outputs_.size()) - 1; i >= 0; --i) {
      auto idx = static_cast<size_t>(i);
      if (outputs_[idx].data_ptr == data_ptr && nbytes >= outputs_[idx].nbytes) {
        outputs_[idx].skip = true;
        break;
      }
    }
  }
  auto &store = outputs_.emplace_back(obj, data_ptr, inplace, tensor);
  if (DumpEnabled()) {
    dump_buf_ << "%" << (outputs_.size() - 1) << ": " << ToString(tensor) << "\n";
  }
  CacheDvmOp(storage, obj, tensor.storage_offset());
}

void *LazyFusionKernel::Alloc(size_t size) {
  auto ws_tensor = at_npu::native::allocate_workspace(size, stream_);
  void *addr = const_cast<void *>(ws_tensor.storage().data());
  workspace_.emplace_back(std::move(ws_tensor));
  if (DumpEnabled()) {
    dump_buf_ << "workspace: " << addr << " " << size << "\n";
  }
  return addr;
}

void LazyFusionKernel::Flush() {
  if (flushed_) {
    return;
  }
  if (outputs_.empty()) {
    Clear();
    return;
  }
  flushed_ = true;
  // Emit Store
  for (auto &out : outputs_) {
    if (out.skip) {
      continue;
    }
    auto p = out.storage.lock();
    if (p) {
      auto storage = static_cast<torch_npu::NPUStorageImpl *>(p.get());
      storage->dvm_op_ = nullptr;
    } else {
      out.skip = true;
      continue;
    }
    auto store = dvm::Kernel::Store(out.data_ptr, out.op);
    if (out.inplace) {
      dvm::Kernel::SetStoreInplace(store);
    }
  }
  if (DumpEnabled()) {
    DumpGraph();
  }
  ClearGraphRefs();
  at_npu::native::OpCommand::RunOpApiV2("Dvm", [this]() -> int { return CodeGenAndLaunch(); });
}

void LazyFusionKernel::ClearRuntimeState() {
  workspace_.clear();
  dvm::Kernel::Clear();
  flushed_ = false;
  g_lazy_fusion_manager.FreeKernel(this);
}

int LazyFusionKernel::CodeGenAndLaunch() {
  if (c10_npu::check_dequeue_need_use(stream_)) {
    NPU_CHECK_ERROR(c10_npu::UseStreamResInCurrentThread(stream_));
  }
  if (DumpEnabled()) {
    LazyFusionDump::Instance().DumpGraphInfo(dump_buf_);
    dump_buf_ << "[lazy_fusion before split]\n";
    dump_buf_ << Dump() << "\n";
    LazyFusionDump::Instance().DumpKernelInfo(dump_buf_);
  }
  {
    RECORD_FUNCTION("CodeGen", {});
    dvm::Kernel::CodeGen(nullptr, 0, this);
  }
  if (DumpEnabled()) {
    dump_buf_ << "[lazy_fusion after split]\n";
    dump_buf_ << Dump() << "\n";
    dump_buf_ << Das() << "\n";
    LazyFusionDump::Instance().DumpKernelInfo(dump_buf_);
  }
  auto ret = dvm::Kernel::Launch(stream_);
  if (g_lazy_fusion_manager.flags_.synchronize) {
    auto err = c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream_);
    if (err != ACL_ERROR_NONE) {
      ASCEND_LOGE("SyncStream failed for dvm kernel, id = %zu, error = %d", id_, err);
    }
  }
  ClearRuntimeState();
  return ret;
}

std::string LazyFusionKernel::ToString(const at::OptionalIntArrayRef &t) {
  if (t.has_value()) {
    std::stringstream ss;
    ss << t.value();
    return ss.str();
  }
  return "None";
}

std::string LazyFusionKernel::ToString(const at::Scalar &t) {
  auto type = t.type();
  std::string type_str = c10::toString(type);
  if (type == at::ScalarType::Long) {
    return type_str + "(" + std::to_string(t.toLong()) + ")";
  }
  if (type == at::ScalarType::Double) {
    return type_str + "(" + std::to_string(t.toDouble()) + ")";
  }
  return type_str + "(?)";
}

std::string LazyFusionKernel::ToString(const at::Tensor &t) {
  std::stringstream ss;
  ss << "Tensor(shape=" << t.sizes() << " dtype=" << c10::toString(t.scalar_type()) << " ";
  if (t.dim() == 0 && !torch_npu::utils::is_npu(t)) {
    ss << "value=" << ToString(t.item()) << " ";
  }
  ss << "strides=" << t.strides() << " is_contiguous=" << t.is_contiguous();
  auto impl = t.storage().unsafeGetStorageImpl();
  if (impl != nullptr) {
    auto ptr = static_cast<torch_npu::NPUStorageImpl *>(impl);
    ss << " storage=" << impl << " ptr=" << ptr->data() << " offset=" << t.storage_offset() << " addr=" << t.data_ptr();
  }
  ss << ")";
  return ss.str();
}

std::string LazyFusionKernel::ToString(const at::Tensor &t, bool verbose) {
  if (verbose || (t.dim() == 0 && !torch_npu::utils::is_npu(t))) {
    return ToString(t);
  }
  auto data_addr = t.data_ptr();
  for (int64_t i = static_cast<int64_t>(input_used_) - 1; i >= 0; --i) {
    auto idx = static_cast<size_t>(i);
    if (inputs_[idx]->data_ptr == data_addr) {
      return "p" + std::to_string(i);
    }
  }
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (outputs_[i].data_ptr == data_addr) {
      return "%" + std::to_string(i);
    }
  }
  return ToString(t);
}

void LazyFusionKernel::DumpGraph() {
  dump_buf_ << "lazy_fusion_graph";
  for (const auto &op : dump_ops_) {
    dump_buf_ << "_" << op.name;
  }
  dump_buf_ << "(" << id() << ", " << this << ") {\n";
  size_t output_idx = 0;
  for (const auto &op : dump_ops_) {
    dump_buf_ << "  ";
    for (size_t i = 0; i < op.output_num; ++i) {
      if (i != 0) {
        dump_buf_ << ", ";
      }
      dump_buf_ << "%" << output_idx;
      output_idx += 1;
    }
    dump_buf_ << " = " << op.name << "(";
    for (size_t i = 0; i < op.inputs.size(); ++i) {
      if (i != 0) {
        dump_buf_ << ", ";
      }
      dump_buf_ << op.inputs[i];
    }
    dump_buf_ << ")\n";
  }
  dump_buf_ << "  return(";
  output_idx = 0;
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (!outputs_[i].skip) {
      if (output_idx != 0) {
        dump_buf_ << ", ";
      }
      dump_buf_ << "%" << i;
      output_idx += 1;
    }
  }
  dump_buf_ << ")\n}\n";
}

bool IsEnabled(const std::string &op, unsigned int op_level) {
  const auto &disable_ops = g_lazy_fusion_manager.flags_.disable_ops;
  const auto &enable_ops = g_lazy_fusion_manager.flags_.enable_ops;
  const auto &enable_ops_only = g_lazy_fusion_manager.flags_.enable_ops_only;
  bool enable = false;
  if (!enable_ops_only.empty()) {
    enable = std::find(enable_ops_only.begin(), enable_ops_only.end(), op) != enable_ops_only.end();
  } else if (op_level > OpLevel_0) {
    enable = std::find(enable_ops.begin(), enable_ops.end(), op) != enable_ops.end();
  } else {
    enable = std::find(disable_ops.begin(), disable_ops.end(), op) == disable_ops.end();
  }
  ASCEND_LOGI("op [%s], op_level is %d, enable = %d", op.c_str(), op_level, enable);
  return enable;
}
}  // namespace lazy_fusion
