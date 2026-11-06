// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_FFT_COMMON_OP_API__
#define __TORCH_NPU_OP_PLUGIN_UTILS_FFT_COMMON_OP_API__

#include <memory>
#include <mutex>
#include "op_plugin/utils/op_api_common.h"
#include "fft_plan_op_api.h"
#include "AsdSipNpuOpApi.h"

namespace op_api {
// For torch_npu._C
void setFFTPlanCapacity(int64_t maxSize);
int64_t getFFTPlanCapacity();
int64_t getFFTPlanSize();
void clearFFTPlanCache();

// For FFTc2cKernelNpuOpApi.cpp
FFTPlanItem get_plan(int64_t prb_size, bool is_forward, PlanMode plan_mode, at::ScalarType scalar_dtype);

class FFTHandle {
 public:
  FFTHandle(const FFTParam& param, c10::DeviceIndex device, aclrtStream stream);
  ~FFTHandle() noexcept;
  FFTHandle(const FFTHandle&) = delete;
  FFTHandle& operator=(const FFTHandle&) = delete;

  size_t workspaceSize() const {
    return workspaceSize_;
  }
  int execute(FftExecApiFunc exec, const aclTensor* input, const aclTensor* output, void* workspace);

 private:
  void release(bool synchronize) noexcept;

  asdFftHandle handle_ = nullptr;
  c10::DeviceIndex device_;
  aclrtStream stream_;
  size_t workspaceSize_ = 0;
  std::mutex executionMutex_;
};

// The caller and its queued task retain ownership independently of the LRU cache.
std::shared_ptr<FFTHandle> getHandle(const FFTParam& param, const c10_npu::NPUStream& stream);

struct FFTCacheKey {
  bool isAsdSip = false;
  c10::DeviceIndex device = -1;
  aclrtStream stream = nullptr;
  FFTParam fftParam;
  PlanKey planKey;
};
struct FFTCacheValue {
  std::shared_ptr<FFTHandle> handle;
  FFTPlanItem plan;
};

inline bool operator==(const FFTCacheKey& one, const FFTCacheKey& other) {
  if (one.isAsdSip != other.isAsdSip || one.device != other.device) {
    return false;
  }
  if (one.isAsdSip) {
    return one.stream == other.stream && one.fftParam == other.fftParam;
  } else {
    return one.planKey == other.planKey;
  }
}

class FFTMixCache {
 public:
  using FFTPair = std::pair<FFTCacheKey, FFTCacheValue>;
  FFTMixCache(int64_t c);
  FFTCacheValue get(FFTCacheKey& cacheKey);
  void setCapacity(int64_t maxSize);
  int64_t getCapacity();
  int64_t getSize();
  void clear();

 private:
  int64_t capacity;
  std::list<FFTPair> list{};
  std::mutex fftMutex;
};

} // namespace op_api

#define DO_ASDSIP_COMPATIBILITY(asdSipApi, originCallExpression)                     \
  do {                                                                               \
    static const auto opApiFuncAddr = GetAsdSipApiFuncAddr("asdFftExec" #asdSipApi); \
    if (opApiFuncAddr == nullptr) {                                                  \
      ASCEND_LOGW(                                                                   \
          "%s not in %s, or %s not found. Will call %s",                             \
          #asdSipApi,                                                                \
          GetAsdSipApiLibName(),                                                     \
          GetAsdSipApiLibName(),                                                     \
          #originCallExpression);                                                    \
      return originCallExpression;                                                   \
    }                                                                                \
  } while (0)

/**
 * 异步调用npu执行, 无返回值.
 */
#define EXEC_ASDSIP_FFT_NPU_CMD(fftExecApi, inData, outData, fftParam)                             \
  do {                                                                                         \
    auto sip_stream = c10_npu::getCurrentNPUStream();                                            \
    at_npu::native::SetDeterministic();                                                         \
    static const auto asdFftExec = GetAsdSipApiFuncAddr("asdFftExec" #fftExecApi);                 \
    TORCH_CHECK(asdFftExec != nullptr, "asdFftExec" #fftExecApi " not found");                   \
    auto handle = op_api::getHandle(fftParam, sip_stream);                                      \
    size_t workspace_size = handle->workspaceSize();                                           \
    at::Tensor workspace_tensor;                                                               \
    if (workspace_size != 0) {                                                                 \
      workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size); \
    }                                                                                         \
    auto release_tensor = [](aclTensor* tensor) { Release(tensor); };                           \
    auto input = std::shared_ptr<aclTensor>(ConvertType(inData), release_tensor);                \
    auto output = std::shared_ptr<aclTensor>(ConvertType(outData), release_tensor);              \
    auto sip_call = [handle, input, output, workspace_tensor]() -> int {                         \
      void* workspace_addr = workspace_tensor.defined()                                        \
          ? const_cast<void*>(workspace_tensor.storage().data()) : nullptr;                    \
      FftExecApiFunc fftExecApiFunc = reinterpret_cast<FftExecApiFunc>(asdFftExec);               \
      auto api_ret = handle->execute(fftExecApiFunc, input.get(), output.get(), workspace_addr); \
      TORCH_CHECK(api_ret == 0, "call asdFftExec" #fftExecApi " failed");                       \
      return api_ret;                                                                         \
    };                                                                                        \
    at_npu::native::OpCommand::RunOpApiV2("asdFftExec" #fftExecApi, sip_call);                    \
  } while (false)

#endif //  __TORCH_NPU_OP_PLUGIN_UTILS_FFT_COMMON_OP_API__
