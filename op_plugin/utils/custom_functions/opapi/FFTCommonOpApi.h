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

    // For EXEC_ASDSIP_FFT_NPU_CMD
    asdFftHandle getHandle(FFTParam param);

    struct FFTCacheKey {
        bool isAsdSip;
        FFTParam fftParam;
        PlanKey planKey;
    };
    struct FFTCacheValue {
        asdFftHandle handle = nullptr;
        FFTPlanItem plan;
    };

    inline bool operator==(const FFTCacheKey &one, const FFTCacheKey &other)
    {
        if (one.isAsdSip) {
            return one.fftParam == other.fftParam;
        } else {
            return one.planKey == other.planKey;
        }
    }

    class FFTMixCache {
    public:
        using FFTPair = std::pair<FFTCacheKey, FFTCacheValue>;
        FFTMixCache(int64_t c);
        FFTCacheValue get(FFTCacheKey &cacheKey);
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

#define DO_ASDSIP_COMPATIBILITY(asdSipApi, originCallExpression)                                                       \
    do {                                                                                                               \
        static const auto opApiFuncAddr = GetAsdSipApiFuncAddr("asdFftExec" #asdSipApi);                               \
        if (opApiFuncAddr == nullptr) {                                                                                \
            ASCEND_LOGW("%s not in %s, or %s not found. Will call %s", #asdSipApi,                                     \
                        GetAsdSipApiLibName(), GetAsdSipApiLibName(), #originCallExpression);                          \
            return originCallExpression;                                                                               \
        }                                                                                                              \
    } while (0)

/**
 * 异步调用npu执行, 无返回值. (V1版本 - 对应EXEC_NPU_CMD_V1)
 */
#define EXEC_ASDSIP_FFT_NPU_CMD_V1(fftExecApi, inData, outData, fftParam)                                              \
    do {                                                                                                               \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        if (c10_npu::check_enqueue_need_use(sip_stream)) {                                                             \
            c10_npu::UseStreamResInCurrentThread(sip_stream);                                                          \
        }                                                                                                              \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);          \
        at_npu::native::SetDeterministic();                                                                            \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        asdFftHandle handle = op_api::getHandle(fftParam);                                                             \
        size_t workspace_size = 0;                                                                                     \
        asdSipFftGetWorkspaceSize(handle, workspace_size);                                                             \
        void *workspace_addr = nullptr;                                                                                \
        at::Tensor workspace_tensor;                                                                                   \
        if (workspace_size != 0) {                                                                                     \
            workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);                  \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        asdSipFftSetWorkspace(handle, workspace_addr);                                                                 \
        asdSipFftSetStream(handle, sip_stream);                                                                        \
        auto input = ConvertType(inData);                                                                         \
        auto output = ConvertType(outData);                                                                       \
        static const auto asdFftExec = GetAsdSipApiFuncAddr("asdFftExec" #fftExecApi);                                 \
        auto sip_call = [handle, input, output, sip_stream, asdFftExec, releaseMemAddr]() mutable -> int {            \
            if (c10_npu::check_dequeue_need_use(sip_stream)) {                                                         \
                c10_npu::UseStreamResInCurrentThread(sip_stream);                                                      \
            }                                                                                                          \
            FftExecApiFunc fftExecApiFunc = reinterpret_cast<FftExecApiFunc>(asdFftExec);                              \
            auto api_ret = fftExecApiFunc(handle, input, output);                                                      \
            TORCH_CHECK(api_ret == 0, "call " "asdFftExec" #fftExecApi " failed");                                     \
            ReleaseConvertTypes(input);                                                                                  \
            ReleaseConvertTypes(output);                                                                                \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                        \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                        \
            }                                                                                                          \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApiV2("asdFftExec" #fftExecApi, sip_call);                                     \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        UnInitCacheThreadLocal();                                                                                      \
    } while (false)

/**
 * 异步调用npu执行, 无返回值. (V2版本 - 对应EXEC_NPU_CMD_V2，包含deterministic_status和core control)
 */
#define EXEC_ASDSIP_FFT_NPU_CMD_V2(fftExecApi, inData, outData, fftParam)                                              \
    do {                                                                                                               \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        if (c10_npu::check_enqueue_need_use(sip_stream)) {                                                             \
            c10_npu::UseStreamResInCurrentThread(sip_stream);                                                          \
        }                                                                                                              \
        asdFftHandle handle = op_api::getHandle(fftParam);                                                             \
        size_t workspace_size = 0;                                                                                     \
        asdSipFftGetWorkspaceSize(handle, workspace_size);                                                             \
        void *workspace_addr = nullptr;                                                                                \
        at::Tensor workspace_tensor;                                                                                   \
        if (workspace_size != 0) {                                                                                     \
            workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, sip_stream);     \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        asdSipFftSetWorkspace(handle, workspace_addr);                                                                 \
        asdSipFftSetStream(handle, sip_stream);                                                                        \
        auto input = ConvertType(inData);                                                                         \
        auto output = ConvertType(outData);                                                                       \
        auto deterministic_status = at::globalContext().deterministicAlgorithms();                                     \
        uint32_t aic_num = UINT32_MAX;                                                                                  \
        uint32_t aiv_num = UINT32_MAX;                                                                                  \
        if (c10_npu::is_core_control_enabled()) {                                                                       \
            aic_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);                          \
            aiv_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);                        \
        }                                                                                                              \
        static const auto asdFftExec = GetAsdSipApiFuncAddr("asdFftExec" #fftExecApi);                                 \
        auto sip_call = [handle, input, output, sip_stream, asdFftExec, deterministic_status, aic_num, aiv_num,     \
                         initMemAddr, unInitMemAddr, releaseMemAddr]() mutable -> int {                                \
            if (c10_npu::check_dequeue_need_use(sip_stream)) {                                                         \
                c10_npu::UseStreamResInCurrentThread(sip_stream);                                                      \
            }                                                                                                          \
            InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);               \
            UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);       \
            int api_ret = 0;                                                                                            \
            at_npu::native::SetDeterministicOps(deterministic_status);                                                  \
            if (initMemFunc) {                                                                                         \
                initMemFunc(nullptr, false);                                                                           \
            }                                                                                                          \
            FftExecApiFunc fftExecApiFunc = reinterpret_cast<FftExecApiFunc>(asdFftExec);                              \
            api_ret = fftExecApiFunc(handle, input, output);                                                          \
            TORCH_CHECK(api_ret == 0, "call " "asdFftExec" #fftExecApi " failed");                                     \
            ReleaseConvertTypes(input);                                                                                \
            ReleaseConvertTypes(output);                                                                               \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                         \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                       \
            }                                                                                                          \
            if (unInitMemFunc) {                                                                                       \
                unInitMemFunc(nullptr, false);                                                                         \
            }                                                                                                          \
            UnInitCacheThreadLocal();                                                                                  \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApiV2("asdFftExec" #fftExecApi, sip_call);                                     \
    } while (false)

/**
 * 异步调用npu执行, 无返回值. (根据task_queue_enable自动选择V1或V2)
 */
#define EXEC_ASDSIP_FFT_NPU_CMD(fftExecApi, inData, outData, fftParam)                                                 \
    do {                                                                                                               \
        static const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();                   \
        if (task_queue_enable == 2) {                                                                                  \
            EXEC_ASDSIP_FFT_NPU_CMD_V2(fftExecApi, inData, outData, fftParam);                                        \
        } else {                                                                                                       \
            EXEC_ASDSIP_FFT_NPU_CMD_V1(fftExecApi, inData, outData, fftParam);                                        \
        }                                                                                                              \
    } while (false)

#endif //  __TORCH_NPU_OP_PLUGIN_UTILS_FFT_COMMON_OP_API__
