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

#include "op_plugin/utils/custom_functions/opapi/FFTCommonOpApi.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

namespace op_api {

static FFTMixCache fftCache(10);

FFTHandle::FFTHandle(const FFTParam& param, c10::DeviceIndex device, aclrtStream stream)
    : device_(device), stream_(stream) {
  c10_npu::NPUGuard guard(device_);
  try {
    TORCH_CHECK(asdSipFftCreate(handle_) == 0, "asdSipFftCreate failed", OPS_ERROR(ErrCode::INTERNAL));
    if (param.fftYSize == 0) {
      TORCH_CHECK(
          asdSipFftMakePlan1D(
              handle_, param.fftXSize, param.fftType, param.direction, param.batchSize, param.dimType) == 0,
          "asdSipFftMakePlan1D failed",
          OPS_ERROR(ErrCode::INTERNAL));
    } else {
      TORCH_CHECK(
          asdSipFftMakePlan2D(
              handle_, param.fftXSize, param.fftYSize, param.fftType, param.direction, param.batchSize) == 0,
          "asdSipFftMakePlan2D failed",
          OPS_ERROR(ErrCode::INTERNAL));
    }
    TORCH_CHECK(
        asdSipFftGetWorkspaceSize(handle_, workspaceSize_) == 0,
        "asdSipFftGetWorkspaceSize failed",
        OPS_ERROR(ErrCode::INTERNAL));
    // Bind once before publishing the handle, including handles evicted before their first execution.
    TORCH_CHECK(
        asdSipFftSetStream(handle_, stream_) == 0, "asdSipFftSetStream failed", OPS_ERROR(ErrCode::INTERNAL));
  } catch (...) {
    // No FFT execution has been submitted for a partially constructed handle.
    release(false);
    throw;
  }
}

FFTHandle::~FFTHandle() noexcept {
  release(true);
}

void FFTHandle::release(bool synchronize) noexcept {
  if (handle_ == nullptr) {
    return;
  }
  try {
    // Static cache destruction may run after ACL has already been finalized.
    if (!c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
      return;
    }
    c10_npu::NPUGuard guard(device_);
    if (synchronize) {
      // All host users have released ownership. Wait only for submitted device work;
      // draining the taskqueue here could wait for the currently executing callback itself.
      auto ret = asdSipFftSynchronize(handle_);
      if (ret != 0) {
        ASCEND_LOGE("asdSipFftSynchronize failed during FFT handle release, ret=%d; handle retained", ret);
        return;
      }
    }
    auto ret = asdSipFftDestroy(handle_);
    if (ret != 0) {
      ASCEND_LOGE("asdSipFftDestroy failed during FFT handle release, ret=%d", ret);
      return;
    }
    handle_ = nullptr;
  } catch (const std::exception& e) {
    ASCEND_LOGE("FFT handle release failed: %s", e.what());
  } catch (...) {
    ASCEND_LOGE("FFT handle release failed with an unknown exception");
  }
}

int FFTHandle::execute(FftExecApiFunc exec, const aclTensor* input, const aclTensor* output, void* workspace) {
  std::lock_guard<std::mutex> guard(executionMutex_);
  c10_npu::NPUGuard deviceGuard(device_);
  // Per-invocation state must be set immediately before execution, never by a producer
  // preparing another queued FFT that happens to hit the same cached handle.
  TORCH_CHECK(
      asdSipFftSetWorkspace(handle_, workspace) == 0,
      "asdSipFftSetWorkspace failed",
      OPS_ERROR(ErrCode::INTERNAL));
  TORCH_CHECK(
      asdSipFftSetStream(handle_, stream_) == 0, "asdSipFftSetStream failed", OPS_ERROR(ErrCode::INTERNAL));
  return exec(handle_, input, output);
}

FFTMixCache::FFTMixCache(int64_t c) : capacity(c) {}

FFTCacheValue FFTMixCache::get(FFTCacheKey& cacheKey) {
  // These owners must be destroyed after the cache mutex is released, including on exceptions.
  std::list<FFTPair> retired;
  FFTCacheValue value;
  std::lock_guard<std::mutex> guard(fftMutex);
  auto match_key = [&cacheKey](FFTPair& pair) { return pair.first == cacheKey; };
  auto it = std::find_if(list.begin(), list.end(), match_key);
  if (it != list.end()) {
    list.splice(list.end(), list, it);
    return it->second;
  }

  if (cacheKey.isAsdSip) {
    value.handle = std::make_shared<FFTHandle>(cacheKey.fftParam, cacheKey.device, cacheKey.stream);
  } else {
    value.plan = make_plan(cacheKey.planKey);
  }

  list.emplace_back(cacheKey, value);
  if (static_cast<int64_t>(list.size()) > capacity) {
    retired.splice(retired.end(), list, list.begin());
  }
  return value;
}

void FFTMixCache::setCapacity(int64_t maxSize) {
  std::list<FFTPair> retired;
  std::lock_guard<std::mutex> guard(fftMutex);
  if (capacity == maxSize) {
    return;
  }

  capacity = maxSize;
  while (static_cast<int64_t>(list.size()) > capacity) {
    retired.splice(retired.end(), list, list.begin());
  }
}

int64_t FFTMixCache::getCapacity() {
  std::lock_guard<std::mutex> guard(fftMutex);
  return capacity;
}

int64_t FFTMixCache::getSize() {
  std::lock_guard<std::mutex> guard(fftMutex);
  return list.size();
}

void FFTMixCache::clear() {
  std::list<FFTPair> retired;
  std::lock_guard<std::mutex> guard(fftMutex);
  retired.swap(list);
}

void setFFTPlanCapacity(int64_t maxSize) {
  if (maxSize < 1 || maxSize > 99) {
    return;
  }

  fftCache.setCapacity(maxSize);
}

int64_t getFFTPlanCapacity() {
  return fftCache.getCapacity();
}

int64_t getFFTPlanSize() {
  return fftCache.getSize();
}

void clearFFTPlanCache() {
  fftCache.clear();
}

FFTPlanItem get_plan(int64_t prb_size, bool is_forward, PlanMode plan_mode, at::ScalarType scalar_dtype) {
  FFTCacheKey cacheKey;
  cacheKey.isAsdSip = false;
  cacheKey.device = c10_npu::current_device();
  cacheKey.planKey.prb_size = prb_size;
  cacheKey.planKey.is_forward = is_forward;
  cacheKey.planKey.plan_mode = plan_mode;
  cacheKey.planKey.scalar_dtype = scalar_dtype;
  return fftCache.get(cacheKey).plan;
}

std::shared_ptr<FFTHandle> getHandle(const FFTParam& param, const c10_npu::NPUStream& stream) {
  FFTCacheKey cacheKey;
  cacheKey.isAsdSip = true;
  cacheKey.device = stream.device_index();
  cacheKey.stream = stream.stream(false);
  cacheKey.fftParam = param;
  return fftCache.get(cacheKey).handle;
}

} // namespace op_api
