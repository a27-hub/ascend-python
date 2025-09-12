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

#include <torch/torch.h>
#include <unordered_set>
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
using aclnn_get_workspace_func = int32_t (*)(const aclTensor *, const char *, const int, const int, uint64_t *, aclOpExecutor**);
using aclnn_get_func = int32_t (*)(void *, const uint64_t, aclOpExecutor *, aclrtStream);
using aclnn_collect_get_workspace_func = int32_t (*)(const char *, const int, const aclTensor *, uint64_t *, aclOpExecutor**);

class StreamCtxSingle {
private:
    aclrtContext detectCtx[MAX_SERVER_NUM];
    aclrtContext collectCtx;
    int device_id;
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
        static StreamCtxSingle single_instance(dev_id);
        return single_instance;
    }
    int32_t GetCtx(int32_t server_num, aclrtContext *ctx) {
        if (server_num >= MAX_SERVER_NUM) {
            OP_LOG_ERROR(aclnnElasticReceivableTest, "Get Ctx server num bigger than max server num 8.\n");
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

static bool CheckRankValid(uint32_t rank_id, uint32_t world_size, std::unordered_set<int>&server_set, at::Tensor &elastic_rank_table) {
    int* elastic_rank_table_ptr = elastic_rank_table.data_ptr<int>();
    for (int32_t i = 0; i < world_size / DIE_PER_SERVER; i++) {
        if (server_set.count(i) > 0) {
            continue;
        }
        for (int32_t j = 0; j < DIE_PER_SERVER; j++) {
            if (elastic_rank_table_ptr[rank_id * world_size + i * DIE_PER_SERVER + j] == 0) {
                return false;
            }
        }
    }
    for (int32_t i = 0; i <= world_size / DIE_PER_SERVER; i++) {
        if (server_set.count(i) > 0) {
            continue;
        }
        for (int32_t j = 0; j < DIE_PER_SERVER; j++) {
            int32_t offset = 0;
            if (i + j > 0) {
                offset = ((i * DIE_PER_SERVER + j) - 1) * world_size;
            }
            if (elastic_rank_table_ptr[offset + rank_id] == 0) {
                return false;
            }
        }
    }
    return true;
}

static bool CalValidRankListInfo(int32_t rankId, at::Tensor &elastic_rank_table, int64_t world_size, at::Tensor &output_elastic_rank) { 
    auto elastic_table_cpu = elastic_rank_table.cpu();
    std::unordered_set<int>server_set;
    bool* output_elastic_rank_ptr = output_elastic_rank.data_ptr<bool>();
    for (int rank_id = 0; rank_id < world_size; rank_id++) {
        bool is_valid = CheckRankValid(rank_id, world_size, server_set, elastic_table_cpu);
        if (is_valid) {
            output_elastic_rank_ptr[rank_id] = true;
        } else {
            // roll back
            server_set.insert(rank_id / DIE_PER_SERVER);
            uint32_t begin_rank = rank_id / DIE_PER_SERVER * DIE_PER_SERVER;
            for (int i = begin_rank; i < rank_id; i++) {
                output_elastic_rank_ptr[i] = false;
            }
            rank_id = begin_rank + DIE_PER_SERVER - 1;
        }
    }
    return output_elastic_rank_ptr[rankId] == true;
}

void DetecTionThreadFun(at::Tensor &elastic_rank_table, at::Tensor &output_elastic_rank, at::Tensor &arange_value, c10::string_view group_ep,
    int64_t ep_world_size, int64_t ep_rank_id, c10::string_view group_detection, int64_t detection_world_size, int64_t detection_rank_id)
{
    int32_t serverNum = detection_world_size / DIE_PER_SERVER;
    int32_t lastServerDieNum = detection_world_size % DIE_PER_SERVER;
    std::vector<bool> statusVec(serverNum, false);
    StreamCtxSingle& streamCtx = StreamCtxSingle::getInstance(static_cast<int>(elastic_rank_table.device().index()));
    try {
        for (int32_t server = 0; server < serverNum; server++) {
            aclrtContext cur_ctx;
            int ret = streamCtx.GetCtx(server, &cur_ctx);
            aclrtSetCurrentContext(cur_ctx);
            static const auto getWorkspaceSizeFuncAddr =
                reinterpret_cast<aclnn_get_workspace_func>(GetOpApiFuncAddr("aclnnElasticReceivableTestGetWorkspaceSize"));
            static const auto opApiFuncAddr = reinterpret_cast<aclnn_get_func>(GetOpApiFuncAddr("aclnnElasticReceivableTest"));
            TORCH_CHECK(
                getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnElasticReceivableTestGetWorkspaceSize or aclnnElasticReceivableTest symbol not found.",
                OPS_ERROR(ErrCode::PTR));
            static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
            const int64_t stride = 0;
            const int64_t storage_dims = detection_world_size;
            const int64_t view_dims = DIE_PER_SERVER;
            aclTensor *input_rank_tensor = aclCreateTensor(&view_dims, 1, ACL_INT32, &stride, 0,
                    ACL_FORMAT_ND, &storage_dims, 1, arange_value.data_ptr() + (server * DIE_PER_SERVER * 4));
            uint64_t workspace_size = 0;
            void *workspace_addr = nullptr;
            aclOpExecutor *executor = nullptr;
            // auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
            getWorkspaceSizeFuncAddr(input_rank_tensor, const_cast<char*>(group_detection.data()),
                detection_world_size, DIE_PER_SERVER, &workspace_size, &executor);
            if (workspace_size > 0) {
                auto ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
                TORCH_CHECK(ret == ACL_SUCCESS, "[ERROR] aclrtMalloc workspace failed.", OPS_ERROR(ErrCode::MEMORY));
            }
            opApiFuncAddr(workspace_addr, workspace_size, executor, nullptr);
        }
        for (int32_t server = 0; server < serverNum; server++) {
            try {
                aclrtContext cur_ctx;
                int ret = streamCtx.GetCtx(server, &cur_ctx);
                aclrtSetCurrentContext(cur_ctx);
                aclrtSynchronizeStreamWithTimeout(nullptr, 10000);
            } catch (...) {
                // do notiing
                statusVec[server] = true;
                continue;
            }
        }
    } catch (...) {
      OP_LOG_ERROR(aclnnElasticReceivableTest, "Get unExpected Error.\n");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); //100ms
    aclrtSetCurrentContext(streamCtx.GetCollectCtx());
    static const auto getWorkspaceSizeFuncAddr =
        reinterpret_cast<aclnn_get_workspace_func>(GetOpApiFuncAddr("aclnnElasticReceivableInfoCollectGetWorkspaceSize"));
    static const auto opApiFuncAddr = reinterpret_cast<aclnn_get_func>(GetOpApiFuncAddr("aclnnElasticReceivableInfoCollect"));
    TORCH_CHECK(
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
        "aclnnElasticReceivaclnnElasticReceivableInfoCollectGetWorkspaceSizeableTestGetWorkspaceSize or aclnnElasticReceivableInfoCollect symbol not found.",
        OPS_ERROR(ErrCode::PTR));
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    const int64_t stride = 0;
    const int64_t storage_dims = detection_world_size * detection_world_size;
    const int64_t view_dims = detection_world_size * detection_world_size;
    aclTensor *input_rank_tensor = aclCreateTensor(&view_dims, 1, ACL_INT32, &stride, 0,
            ACL_FORMAT_ND, &storage_dims, 1, elastic_rank_table.data_ptr());
    uint64_t workspace_size = 0;
    void *workspace_addr = nullptr;
    aclOpExecutor *executor = nullptr;
    getWorkspaceSizeFuncAddr(const_cast<char*>(group_detection.data()),
        detection_world_size, input_rank_tensor, &workspace_size, &executor);
    if (workspace_size > 0) {
        auto ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "[ERROR] aclrtMalloc workspace failed.", OPS_ERROR(ErrCode::MEMORY));
    }
    opApiFuncAddr(workspace_addr, workspace_size, executor, nullptr);
    
    aclrtSynchronizeStreamWithTimeout(nullptr, 10000);
    bool needClean =
        CalValidRankListInfo(detection_rank_id, elastic_rank_table, detection_world_size, output_elastic_rank);
    if (needClean) {
        at::Tensor elastic_rank = output_elastic_rank.to(elastic_rank_table.device());
        EXEC_NPU_CMD(aclnnClean, group_ep, ep_world_size, ep_rank_id,
            elastic_rank);
    }
}

at::Tensor npu_fault_detection(const at::Tensor &expand_x, c10::string_view group_ep, int64_t ep_world_size,
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
    at::Tensor output_elastic_rank = torch::full({detection_world_size}, false, torch::kBool);
    at::Tensor elastic_rank_table = npu_preparation::apply_tensor_without_format(
        {detection_world_size * detection_world_size}, dtype(at::kInt));
    at::Tensor arange_value = torch::arange(0, detection_world_size, 1, torch::kInt32).to(elastic_rank_table.device());
    SimpleSingleThread& threadPool = SimpleSingleThread::getInstance();
    threadPool.executeAndWait(std::bind(DetecTionThreadFun, elastic_rank_table, output_elastic_rank, arange_value, group_ep,
        ep_world_size, ep_rank_id, group_detection, detection_world_size, detection_rank_id));
    return output_elastic_rank;
}
}  // namespace op_api