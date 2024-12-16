#include "op_param_cache.h"
#include <iostream>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace op_api {

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
