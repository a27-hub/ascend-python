#ifndef OPPLUGIN_UTILS_ATB_PARAM_CACHE_H
#define OPPLUGIN_UTILS_ATB_PARAM_CACHE_H

#include <unordered_map>
#include <mutex>
#include <memory>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "op_plugin/third_party/atb/inc/atb_infer.h"
#include "op_cache_compute.h"


namespace op_api {

template <typename ParamType>
class OpParamCache {
public:

    static OpParamCache& getInstance();

    atb::Operation* getOperation(const ParamType& param, const std::string& name);

private:
    OpParamCache() = default;

    OpParamCache(const OpParamCache&) = delete;
    OpParamCache& operator=(const OpParamCache&) = delete;

    ~OpParamCache();

    std::unordered_map<size_t, atb::Operation*> opMap_; 
    mutable std::mutex mutex_; 
};


template <typename ParamType>
OpParamCache<ParamType>& OpParamCache<ParamType>::getInstance() {
    static OpParamCache instance;
    return instance;
}


template <typename ParamType>
atb::Operation* OpParamCache<ParamType>::getOperation(const ParamType& param, const std::string& name) {

    size_t hashValue = computeHash(param);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = opMap_.find(hashValue);
        if (it != opMap_.end()) {
            return it->second;
        }
        
        atb::Operation* op = nullptr;
        atb::CreateOperation(param, &op);
        TORCH_CHECK(op != nullptr, name, " CreateOperation failed!");
        opMap_[hashValue] = op;
        return op;
    }
}


template <typename ParamType>
OpParamCache<ParamType>::~OpParamCache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : opMap_) {
        DestroyOperation(entry.second);
    }
}


}

#endif // OPPLUGIN_UTILS_ATB_PARAM_CACHE_H