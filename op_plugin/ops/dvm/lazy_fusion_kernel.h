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

#ifndef DVM_LAZY_FUSION_KERNEL_H
#define DVM_LAZY_FUSION_KERNEL_H

#include <vector>
#include <queue>
#include <mutex>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include "op_plugin/utils/OpAdapter.h"
#include "dvm.h"
#include "op_plugin/ops/dvm/lazy_fusion_flags.h"

namespace lazy_fusion {
constexpr unsigned int OpLevel_0 = 0;
constexpr unsigned int OpLevel_1 = 1;

class LazyFusionKernel : public dvm::Kernel, public dvm::WsAllocator {
 public:
  LazyFusionKernel();
  ~LazyFusionKernel();
  void Flush();

  void Reset(aclrtStream stream, size_t id) {
    stream_ = stream;
    id_ = id;
  }
  size_t id() const { return id_; }

  dvm::NDObject *Input(const at::Tensor &x, bool enable_cast = true, dvm::ShapeRef *shape = nullptr);
  void Output(const at::Tensor &tensor, dvm::NDObject *obj, bool inplace = false);
  at::Tensor Output(dvm::NDObject *obj, c10::IntArrayRef shape, const c10::TensorOptions &options) {
    at::Tensor tensor = at_npu::native::OpPreparation::apply_tensor_without_format(shape, options);
    Output(tensor, obj);
    return tensor;
  }

  c10::IntArrayRef GetShape(dvm::NDObject *obj) {
    auto shape_ref = dvm::Kernel::GetShape(obj);
    return c10::IntArrayRef(shape_ref->data, shape_ref->size);
  }

  template <typename T>
  dvm::ShapeRef *GetShapeRef(const T &shape) {
    if (cache_shape_used_ == cached_shape_.size()) {
      cached_shape_.push_back(new ShapeWithRef());
    }
    auto ws = cached_shape_[cache_shape_used_++];
    ws->Update(shape);
    return ws;
  }

  dvm::DType TransType(at::ScalarType type) {
    switch (type) {
      case at::ScalarType::Bool:
        return dvm::DType::kBool;
      case at::ScalarType::Int:
        return dvm::DType::kInt32;
      case at::ScalarType::Half:
        return dvm::DType::kFloat16;
      case at::ScalarType::Float:
        return dvm::DType::kFloat32;
      case at::ScalarType::BFloat16:
        return dvm::DType::kBFloat16;
      default:
        return dvm::DType::kDataTypeEnd;
    }
  }

  // dvm::WsAllocator interface
  void *Alloc(size_t size) override;

  struct Op {
    std::string name;
    std::vector<std::string> inputs;
    size_t output_num;
  };

  template <typename... Args>
  void DumpOp(const std::string &op_name, const Args &... inputs) {
    auto &op = dump_ops_.emplace_back();
    op.name = op_name;
    op.output_num = outputs_.size() - dump_idx_;
    (DumpOpInput(&op, inputs), ...);
    dump_idx_ = outputs_.size();
  }

 private:
  void ClearGraph() {
    for (size_t i = 0; i < input_used_; i++) {
      auto p = inputs_[i]->tensor.storage().getWeakStorageImpl().lock();
      if (p) {
        auto storage = static_cast<torch_npu::NPUStorageImpl *>(p.get());
        storage->dvm_op_ = nullptr;
      }
      inputs_[i]->tensor = at::Tensor();
    }
    outputs_.clear();
    workspace_.clear();
    dump_ops_.clear();
    input_used_ = 0;
    ops_map_used_ = 0;
    cache_shape_used_ = 0;
    dump_idx_ = 0;
    flushed_ = false;
  }
  void ClearKernel();
  void Clear() {
    ClearGraph();
    ClearKernel();
  }

  template <typename T>
  std::string ToString(T t) { return std::to_string(t); }
  std::string ToString(bool t) { return t ? "True" : "False"; }
  std::string ToString(c10::string_view t) { return std::string(t); }
  std::string ToString(const c10::ScalarType &t) { return c10::toString(t); }
  std::string ToString(const at::OptionalIntArrayRef &t);
  std::string ToString(const at::Scalar &t);
  std::string ToString(const at::Tensor &t);
  std::string ToString(const at::Tensor &t, bool verbose);

  template <typename T>
  void DumpOpInput(Op *op, const T &t) {
    op->inputs.push_back(ToString(t));
  }

  template <typename T>
  void DumpOpInput(Op *op, const c10::optional<T> &t) {
    if (!t.has_value()) {
      op->inputs.push_back("None");
    } else {
      DumpOpInput(op, t.value());
    }
  }

  void DumpOpInput(Op *op, const at::Tensor &t) {
    op->inputs.push_back(ToString(t, false));
  }

  void DumpOpInput(Op *op, const at::TensorList tensors) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (i != 0) {
        ss << ", ";
      }
      ss << ToString(tensors[i], false);
    }
    ss << "]";
    op->inputs.push_back(ss.str());
  }

  void DumpOpInput(Op *op, at::ArrayRef<at::Scalar> scalars) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < scalars.size(); ++i) {
      if (i != 0) {
        ss << ", ";
      }
      ss << ToString(scalars[i]);
    }
    ss << "]";
    op->inputs.push_back(ss.str());
  }

  void DumpGraph();

  struct Load {
    dvm::ShapeRef shape;
    void *data_ptr{nullptr};
    at::Tensor tensor;
  };

  struct Store {
    Store() = default;
    Store(dvm::NDObject *p, void *ptr, bool is_inplace, const at::Tensor &t)
        : op(p), data_ptr(ptr), nbytes(t.nbytes()), inplace(is_inplace),
          storage(t.storage().getWeakStorageImpl()) {}
    dvm::NDObject *op;
    void *data_ptr{nullptr};
    size_t nbytes{0};
    bool inplace{false};
    bool skip{false};
    c10::weak_intrusive_ptr<c10::StorageImpl> storage;
  };

  struct ShapeWithRef : public dvm::ShapeRef {
    ShapeWithRef() {
      data = shape_data;
      size = 0;
    }

    template <typename T>
    void Update(const T &shape) {
      size_t idx = 0;
      for (auto i = shape.begin(); i != shape.end(); ++i) {
        shape_data[idx++] = *i;
      }
      size = idx;
    }

    int64_t shape_data[op_infer::SIZE];
  };

  struct DvmOp {
    dvm::NDObject *op{nullptr};
    int64_t storage_offset{0};
  };

  void CacheDvmOp(torch_npu::NPUStorageImpl *storage, dvm::NDObject *obj, int64_t storage_offset);

  std::vector<Load *> inputs_;
  std::vector<Store> outputs_;
  std::vector<at::Tensor> workspace_;
  std::vector<DvmOp *> ops_map_;
  std::vector<ShapeWithRef *> cached_shape_;
  std::vector<Op> dump_ops_;
  size_t input_used_{0};
  size_t ops_map_used_{0};
  size_t cache_shape_used_{0};
  size_t dump_idx_{0};
  size_t id_{0};
  bool flushed_{false};
  aclrtStream stream_;
  std::stringstream dump_buf_;
};

class Manager {
 public:
  Manager() = default;
  ~Manager();

  LazyFusionKernel *Get() {
    static bool runtime_init = false;
    if (!runtime_init) {
      bool enable_deterministic = at::globalContext().deterministicAlgorithms();
      auto &conf = dvm::Config::Instance();
      if (enable_deterministic) {
        conf.SetDeterm();
      } else {
        conf.UnsetDeterm();
      }
      ASCEND_LOGI("Set dvm deterministic = %d", enable_deterministic);
      bool enable_tuning = flags_.online_tuning;
      if (enable_deterministic) {
        enable_tuning = false;
      }
      if (enable_tuning) {
        conf.SetLazyTuner();
      } else {
        conf.UnsetLazyTuner();
      }
      ASCEND_LOGI("Set dvm online tuning = %d", enable_tuning);
      runtime_init = true;
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    if (current_ != nullptr) {
      if (current_stream_ != stream) {
        Flush();
      } else {
        return current_;
      }
    }
    current_ = NewKernel();
    current_stream_ = stream;
    current_->Reset(stream, id_.fetch_add(1, std::memory_order_relaxed));
    return current_;
  }

  void Flush() {
    if (auto k = current_; k != nullptr) {
      current_ = nullptr;
      k->Flush();
    }
  }

  bool Empty() { return current_ == nullptr; }

  void FreeKernel(LazyFusionKernel *k) {
    std::lock_guard<std::mutex> guard(mutex_);
    pool_.push(k);
  }

  LazyFusionFlags flags_;

 private:
  LazyFusionKernel *NewKernel();

  std::queue<LazyFusionKernel *> pool_;
  LazyFusionKernel *current_{nullptr};
  aclrtStream current_stream_{nullptr};
  std::mutex mutex_;
  std::atomic<size_t> id_{0};
};

extern Manager g_lazy_fusion_manager;

inline bool IsEnabled() {
  static auto enable_task_queue = c10_npu::option::OptionsManager::GetTaskQueueEnable();
  if (g_lazy_fusion_manager.flags_.opt_level > OptLevel_0 && enable_task_queue) {
    auto stream = c10_npu::getCurrentNPUStream();
    return !stream.isSyncLaunchStream();
  }
  return false;
}

bool IsEnabled(const std::string &op, unsigned int op_level);

inline void LazyFusionFlush() { g_lazy_fusion_manager.Flush(); }
}  // namespace lazy_fusion
#endif  // DVM_LAZY_FUSION_KERNEL_H
