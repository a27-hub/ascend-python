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

#ifndef DVM_LAZY_FUSION_KERNEL_H
#define DVM_LAZY_FUSION_KERNEL_H

#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <string>
#include <unordered_map>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/graph_task.h>
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/ops/dvm/lazy_fusion_flags.h"
#include "op_plugin/ops/dvm/lazy_fusion_accuracy.h"
#include "third_party/dvm/dvm/include/dvm.h"

namespace lazy_fusion {

// Thread-local re-entrancy guard for the online accuracy check. The check
// computes op_api references and compares them with aten ops (clone, sub, abs,
// clamp, isclose, min/max, ...). Several of those are themselves DVM-intercepted
// ops, so running them while a fusion kernel is in flight would recursively
// build a new fusion that reuses the pooled kernel object and corrupts the
// check's own state. While this flag is set, IsEnabled() returns false so the
// helper ops fall through to plain aclnn (no fusion).
extern thread_local bool g_in_accuracy_check;
struct AccuracyReentryGuard {
  bool prev_;
  AccuracyReentryGuard() : prev_(g_in_accuracy_check) { g_in_accuracy_check = true; }
  ~AccuracyReentryGuard() { g_in_accuracy_check = prev_; }
};

class LazyFusionKernel final : public dvm::Kernel, public dvm::WsAllocator {
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
  dvm::NDObject *ViewInput(const at::Tensor &base, void *data_ptr, dvm::ShapeRef *shape,
                            dvm::ShapeRef *stride, bool enable_cast = true);
  void Output(const at::Tensor &tensor, dvm::NDObject *obj, bool inplace = false);
  bool NeedFlushForInput(const at::Tensor &x, dvm::ShapeRef *shape = nullptr) const;
  bool NeedFlushForWritableOutput(const at::Tensor &tensor) const;
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

  // Append an op name to the running accuracy-check op list (for diagnostics).
  // Called from the free DumpOp() helper when flags_.check_accuracy is true.
  void AddAccOpName(const std::string &op_name) {
    if (!acc_ops_str_.empty()) {
      acc_ops_str_ += ", ";
    }
    acc_ops_str_ += op_name;
  }

  // Eagerly compute this op's op_api reference output(s) and register them in
  // acc_ref_map_ keyed by the DVM output data_ptr, so a later fused op resolves
  // this op's *reference* result as its input. `fn` is the inline closure built
  // by LF_ACC at the op entry: it calls op_api with each tensor argument routed
  // through the Resolver. Running here (during capture, in the op entry's stack)
  // means every argument is still alive -- no deferred-replay copies needed.
  template <typename Fn>
  void RecordAccEager(Fn &&fn) {
    size_t out_start = acc_dump_idx_;
    Resolver resolver(acc_ref_map_);
    // The reference op_api calls are direct (no dispatch), but the Resolver's
    // clone()/empty_like() are dispatched ops -- guard so they don't re-fuse.
    std::vector<at::Tensor> refs;
    {
      AccuracyReentryGuard guard;
      refs = fn(resolver);
    }
    for (size_t i = 0; i < refs.size() && (out_start + i) < outputs_.size(); ++i) {
      acc_ref_map_[outputs_[out_start + i].tensor_meta.data_ptr] = refs[i];
    }
    acc_dump_idx_ = outputs_.size();
  }

  // Run accuracy check: compare the DVM fused outputs against the eagerly
  // computed op_api references; on mismatch print the DVM graph dump + das.
  // Called from Flush() after CodeGen+Launch when check_accuracy is enabled.
  void CheckAccuracy();
  // Snapshot the (dvm output handle, reference output) pairs before ClearGraphRefs
  // wipes outputs_/acc_ref_map_. Called from Flush() before CodeGen+Launch.
  void PrepareAccuracyCheck();

 private:
  void ClearGraphRefs() {
    for (size_t i = 0; i < input_used_; i++) {
      auto p = inputs_[i]->tensor.storage().getWeakStorageImpl().lock();
      if (p) {
        auto storage = static_cast<torch_npu::NPUStorageImpl *>(p.get());
        storage->lazy_fusion_data_ = nullptr;
      }
      inputs_[i]->tensor = at::Tensor();
    }
    for (auto &output : outputs_) {
      auto p = output.storage.lock();
      if (p) {
        auto storage = static_cast<torch_npu::NPUStorageImpl *>(p.get());
        storage->lazy_fusion_data_ = nullptr;
      }
    }
    outputs_.clear();
    dump_ops_.clear();
    acc_ref_map_.clear();
    acc_ops_str_.clear();
    input_used_ = 0;
    dvm_ops_used_ = 0;
    cache_shape_used_ = 0;
    dump_idx_ = 0;
    acc_dump_idx_ = 0;
  }
  void ClearRuntimeState();
  void Clear() {
    workspace_.clear();
    ClearGraphRefs();
    ClearRuntimeState();
  }

  void CodeGenAndDump();
  int Launch();

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

  // Cache the tensor identity and metadata needed by the exact-match fast path.
  struct TensorMeta {
    c10::TensorImpl *tensor_impl{nullptr};
    void *data_ptr{nullptr};
    int64_t storage_offset{0};
    at::ScalarType dtype{at::ScalarType::Undefined};
    int64_t dim{0};
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
  };

  struct Load {
    dvm::ShapeRef shape;
    dvm::ShapeRef stride;             // populated when has_stride=true
    bool has_stride{false};           // true → strided Load form (Input non-contig path / ViewInput)
    void *data_ptr{nullptr};
    at::Tensor tensor;
  };

  struct Store {
    Store() = delete;
    Store(dvm::NDObject *p, bool is_inplace, const at::Tensor &t)
        : op(p), inplace(is_inplace),
          storage(t.storage().getWeakStorageImpl()),
          tensor_impl(t.getIntrusivePtr()) {}
    dvm::NDObject *op;
    TensorMeta tensor_meta;
    bool inplace{false};
    bool skip{false};
    c10::weak_intrusive_ptr<c10::StorageImpl> storage;
    // Weak ref to the output TensorImpl, used by accuracy check to read the
    // fused result after Launch. Does not extend the tensor's lifetime. Uses the
    // same NullType (UndefinedTensorImpl) as at::Tensor's intrusive_ptr so it can
    // be constructed from getIntrusivePtr() and locked back into an at::Tensor.
    c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl> tensor_impl;
  };

  // Helper to rebuild an at::Tensor from a weak TensorImpl ref for accuracy
  // check. Returns an undefined tensor if the impl is no longer alive.
  static at::Tensor TensorFromWeakImpl(
      const c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl> &weak) {
    auto strong = weak.lock();
    if (!strong) {
      return at::Tensor();
    }
    return at::Tensor(std::move(strong));
  }

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

  // Track whether the cached NDObject may be rebuilt from GM on an exact-match miss.
  struct DvmOp {
    dvm::NDObject *op{nullptr};
    TensorMeta tensor_meta;
    std::vector<int64_t> dvm_shape;
    bool reloadable_from_gm{false};
  };

  static void CacheTensorMeta(TensorMeta *meta, const at::Tensor &tensor);
  static bool MatchTensorMeta(const TensorMeta &meta, const at::Tensor &tensor);
  static void CacheDvmShape(DvmOp *dvm_op, dvm::ShapeRef *shape, c10::IntArrayRef default_shape);
  static bool MatchDvmShape(const DvmOp *dvm_op, dvm::ShapeRef *shape, c10::IntArrayRef default_shape);
  // Exact reuse requires both tensor metadata and logical DVM shape to match.
  static bool MatchDvmOp(const DvmOp *dvm_op, const at::Tensor &tensor, dvm::ShapeRef *shape);

  void CacheDvmOp(torch_npu::NPUStorageImpl *storage, const at::Tensor &tensor, const TensorMeta *tensor_meta,
                  dvm::NDObject *obj, dvm::ShapeRef *shape, bool reloadable_from_gm);

  std::vector<Load *> inputs_;
  std::vector<Store> outputs_;
  std::vector<at::Tensor> workspace_;
  std::vector<DvmOp *> dvm_ops_;
  std::vector<ShapeWithRef *> cached_shape_;
  std::vector<Op> dump_ops_;
  size_t input_used_{0};
  size_t dvm_ops_used_{0};
  size_t cache_shape_used_{0};
  size_t dump_idx_{0};
  size_t id_{0};
  bool flushed_{false};
  aclrtStream stream_;
  std::stringstream dump_buf_;
  // Accuracy check state (only used when flags_.check_accuracy is true).
  // Maps a DVM output data_ptr to the op_api reference result computed eagerly
  // for that output; threads references through the fused op chain.
  std::unordered_map<void *, at::Tensor> acc_ref_map_;
  size_t acc_dump_idx_{0};                       // outputs_ slice consumed by RecordAccEager
  std::string acc_ops_str_;                      // running op-name list (capture phase)
  std::string acc_report_ops_;                   // snapshot of acc_ops_str_ for CheckAccuracy
  std::string acc_dump_text_;                    // DVM graph dump captured at CodeGen
  std::string acc_das_text_;                     // DVM das captured at CodeGen
  bool check_accuracy_pending_{false};
  std::vector<at::Tensor> check_accuracy_ref_outputs_;
  std::vector<at::Tensor> check_accuracy_output_handles_;
};

class Manager {
 public:
  Manager() = default;
  ~Manager();

  LazyFusionKernel *Get() {
    static bool runtime_init = false;
    if (!runtime_init) {
      auto &conf = dvm::Config::Instance();
      bool enable_tuning = flags_.online_tuning;
      if (enable_tuning) {
        conf.SetLazyTuner();
      } else {
        conf.UnsetLazyTuner();
      }
      ASCEND_LOGI("Set dvm online tuning = %d", enable_tuning);
      runtime_init = true;
    }
    {
      static int dvm_determ_oldstatus = -1;
      int determ = at::globalContext().deterministicAlgorithms() ? 1 : 0;
      if (dvm_determ_oldstatus != determ) {
        auto &conf = dvm::Config::Instance();
        if (determ) {
          conf.SetDeterm();
        } else {
          conf.UnsetDeterm();
        }
        dvm_determ_oldstatus = determ;
        ASCEND_LOGI("Set dvm determ = %d", determ);
      }
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    if (current_ != nullptr) {
      if (current_stream_ != stream) {
        ASCEND_LOGI("dvm Manager: stream changed, flush current graph (id=%zu)", current_->id());
        Flush();
      } else {
        return current_;
      }
    }
    current_ = NewKernel();
    current_stream_ = stream;
    auto kid = id_.fetch_add(1, std::memory_order_relaxed);
    current_->Reset(stream, kid);
    ASCEND_LOGD("dvm Manager: new kernel id=%zu, stream=%p", kid, stream);
    return current_;
  }

  void Flush() {
    if (auto k = current_; k != nullptr) {
      current_ = nullptr;
      k->Flush();
    }
  }

  bool Empty() { return current_ == nullptr; }
  bool NeedFlushForInput(const at::Tensor &x, dvm::ShapeRef *shape = nullptr) {
    auto k = current_;
    return k != nullptr && k->NeedFlushForInput(x, shape);
  }
  bool NeedFlushForWritableOutput(const at::Tensor &tensor) {
    auto k = current_;
    return k != nullptr && k->NeedFlushForWritableOutput(tensor);
  }

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

inline void LazyFusionFlush() { g_lazy_fusion_manager.Flush(); }

inline bool IsEnabled() {
  // Suppress fusion while the accuracy check runs its reference/compare ops,
  // otherwise those (DVM-intercepted) aten ops recurse into the pool.
  if (g_in_accuracy_check) {
    return false;
  }
  static const bool global_enabled =
      g_lazy_fusion_manager.flags_.enabled &&
      c10_npu::option::OptionsManager::GetTaskQueueEnable();
  if (!global_enabled) {
    return false;
  }
  if (c10_npu::getCurrentNPUStream().isSyncLaunchStream()) {
    return false;
  }
  thread_local int8_t is_main_thread = -1;
  if (is_main_thread < 0) {
    is_main_thread = (static_cast<pid_t>(syscall(SYS_gettid)) == getpid()) ? 1 : 0;
  }
  if (is_main_thread == 1) {
    return true;
  }
  return torch::autograd::get_current_graph_task_id() >= 0;
}

// Per-op enable check. `required` declares the minimum optimization level the
// op needs (defaults to kO1; ops with workload-sensitive payoff pass kO2).
bool IsEnabled(const std::string &op, Level required = Level::kO1);
}  // namespace lazy_fusion
#endif  // DVM_LAZY_FUSION_KERNEL_H
