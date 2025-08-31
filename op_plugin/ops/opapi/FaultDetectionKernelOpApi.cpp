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

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>
#include <future>
#include <atomic>
#include <memory>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;
const int DIE_PER_SERVER = 16;
const int MAX_SERVER_NUM = 8;

class StreamCtxSingle {
private:
    aclrtContext detectCtx[MAX_SERVER_NUM];
    aclrtContext collectCtx;
    int device_id;
    static StreamCtxSingle* single_instance;
    static std::mutex mutex;
    StreamCtxSingle(int dev_id = 0) : device_id(dev_id) {
        for (uint32_t i = 0; i < MAX_SERVER_NUM; i++) {
            aclrtCreateContext(&detectCtx[i], device_id);
        }
        aclrtCreateContext(&collectCtx, device_id);
    }
    StreamCtxSingle(const StreamCtxSingle&) = delete;
    StreamCtxSingle& operator = (const StreamCtxSingle&) = delete;
public:
    ~StreamCtxSingle() {
    }
    static StreamCtxSingle& getInstance(int dev_id = 0) {
        if (single_instance == nullptr) {
            std::lock_guard<std::mutex> lock(mutex);
            if (single_instance == nullptr) {
                single_instance = new Singleton(dev_id);
            }
        }
        return *single_instance
    }
    int32_t GetCtx(int32_t server_num, aclrtContext *ctx) {
        if (server_num >= MAX_SERVER_NUM) {
            OP_LOG_ERROR(aclnnFaultDectect, "Get Ctx server num bigger than max server num 8.\n");
            return -1;
        }
        *ctx = detectCtx[server_num];
        return 0;
    }
    aclrtContext GetCollectCtx() {
        return collectCtx;
    }

};

class SimpleSingleThread {
private:
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<std::function<void()>> tasks;
    std::atomic<bool> stop;
    std::atomic<int> taskCounter;
    std::mutex completionMtx;
    std::condition_variable completionCv;
    SimpleSingleThread() : stop(false), taskCounter(0) {
        worker = std::thread([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [this]() { return stop || !tasks.empty(); });
                    
                    if (stop && tasks.empty()) break;
                    
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                // 执行任务
                task();
                
                // 任务完成，通知等待的线程
                {
                    std::lock_guard<std::mutex> lock(completionMtx);
                    taskCounter--;
                }
                completionCv.notify_all();
            }
        });
    }
    SimpleSingleThread(const SimpleSingleThread&) = delete;
    SimpleSingleThread& operator = (const SimpleSingleThread&) = delete;
public:
    ~SimpleSingleThread() {
      shutDown();
    }

    // 获取单例实例的静态方法
    static SimpleSingleThread& getInstance() {
        static SimpleSingleThread instance;
        return instance;
    }
    void shutDown() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }
    // 阻塞执行（等待任务完成）
    void executeAndWait(std::function<void()> task) {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        
        // 包装任务，使其完成后设置promise
        auto wrappedTask = [task, &promise]() {
            try {
                task();
                promise.set_value();
            } catch (...) {
                promise.set_exception(std::current_exception());
            }
        };
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (stop) {
                throw std::runtime_error("线程已停止");
            }
            tasks.push(wrappedTask);
            taskCounter++;
        }
        cv.notify_one();
        
        // 阻塞等待任务完成
        future.get();
    }
};

void DetecTionThreadFun(at::Tensor &elastic_rank_table, at::Tensor &output_elastic_rank)
{
    int32_t serverNum = detection_world_size / DIE_PER_SERVER;
      int32_t lastServerDieNum = detection_world_size % DIE_PER_SERVER;
      std::vector<aclStream> streamPool;
      std::vector<bool> statusVec(serverNum, false);
      StreamCtxSingle& streamCtx = StreamCtxSingle::getInstance(static_cast<int>(elastic_rank_table.device().index()));
      try {
            for (int32_t server = 0; server < serverNum; server++) {
                aclrtContext cur_ctx;
                int ret = streamCtx.GetCtx(server, &cur_ctx);
                aclrtSetCurrentContext(cur_ctx);
                static const auto getWorkspaceSizeFuncAddr =
                    GetOpApiFuncAddr("aclnnFaultDectectGetWorkspaceSize");
                static const auto opApiFuncAddr = GetOpApiFuncAddr(aclnnFaultDectect);
                TORCH_CHECK(
                    getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                    #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ",
                    GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.",
                    OPS_ERROR(ErrCode::PTR));
                auto copied_params = CopyTypesV2(detection_rank_id, group_detection, detection_world_size, );
                auto convertedParams =
                    ConvertTypesV2(copied_params, workspace_size_addr, executor_addr);
                auto getWorkspaceSizeFunc =
                    ConvertToOpApiFunc(convertedParams, getWorkspaceSizeFuncAddr);
                auto workspace_status = call(getWorkspaceSizeFunc, convertedParams);
            }
      for (int32_t server = 0; server < serverNum; server++) {
          try {
            aclrtContext cur_ctx;
            int ret = streamCtx.GetCtx(server, &cur_ctx);
            aclrtSetCurrentContext(cur_ctx);
            c10_npu::acl::AclrtSynchronizeStreamWithTimeout(streamPool[i]);
          } catch (...) {
            // do notiing
            statusVec[i] = true;
            continue;
          }
      }
    } catch (...) {
      OP_LOG_ERROR(aclnnFaultDectect, "Get unExpected Error.\n");
    }
    time.sleep(100);  // 100ms
    aclrtSetCurrentContext(streamCtx.GetCollectCtx());
    EXEC_NPU_CMD_SYNC(aclnnFaultCollect, detection_rank_id, group_detection,
                      detection_world_size, elastic_rank_table);

    bool needClean =
        CalValidRankListInfo(detection_world_size, output_elastic_rank);
    if (needClean) {
      EXEC_NPU_CMD(aclnnClean, group_ep, ep_world_size, ep_rank_id,
                  output_elastic_rank);
    }
}
bool CalValidRankListInfo(int32_t rankId) { return true; }
at::Tensor npu_fault_detection(c10::string_view group_ep, int64_t ep_world_size,
                               int64_t ep_rank_id,
                               c10::string_view group_detection,
                               int64_t detection_world_size,
                               int64_t detection_rank_id) {
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
              "ep_rank_id should be in [0, ep_world_size), but got",
              " ep_world_size: ", ep_world_size, ", ep_rank_id: ", ep_rank_id,
              ". " + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        (detection_rank_id >= 0) && (detection_rank_id < detection_world_size),
        "detection_rank_id should be in [0, detection_world_size), but got",
        " detection_world_size: ", detection_world_size,
        ", detection_rank_id: ", detection_rank_id,
        ". " + OPS_ERROR(ErrCode::PARAM));
    at::Tensor output_elastic_rank = npu_preparation::apply_tensor_without_format(
        {detection_world_size}, x.options().dtype(at::kBool));
    at::Tensor elastic_rank_table = npu_preparation::apply_tensor_without_format(
        {detection_world_size * detection_world_size},
        x.options().dtype(at::kBool));
    at::Tensor ranks_per_server = npu_preparation::apply_tensor_without_format(
        {DIE_PER_SERVER},x.options().dtype(at::kInt32));
    SingletonThread& threadPool = SingletonThread::getInstance();
    threadPool.execute(std::bind(DetecTionThreadFun, elastic_rank_table, output_elastic_rank));
    return output_elastic_rank;
}
}  // namespace op_api