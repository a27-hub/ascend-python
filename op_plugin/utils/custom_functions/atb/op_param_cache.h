#ifndef OPPLUGIN_UTILS_ATB_PARAM_CACHE_H
#define OPPLUGIN_UTILS_ATB_PARAM_CACHE_H

#include <unordered_map>
#include <mutex>
#include <memory>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "op_plugin/third_party/atb/inc/atb_infer.h"


namespace op_api {

template <typename ParamType>
class OpParamCache {
public:

    static OpParamCache& getInstance();

    atb::Operation* getOperation(const ParamType& param, const std::string& name);

    ~OpParamCache();

private:
    OpParamCache() = default;

    OpParamCache(const OpParamCache&) = delete;
    OpParamCache& operator=(const OpParamCache&) = delete;

    std::unordered_map<size_t, atb::Operation*> opMap_; 
    mutable std::mutex mutex_; 
};


// Each operator implements its own hash function calculation. 
// It is possible to hash only the attributes that may change in the parameters of the calculation.
// The example is as follows:
// template <>
// struct HashOpParam<atb::infer::XXXParam> {   //if XXXParam's transposeA and hasBias need hash
//     size_t operator()(const atb::infer::XXXParam& param) const {
//         size_t h = 0;
//         h ^= std::hash<std::string>{}("transposeA") ^ std::hash<int>{}(param.transposeA);
//         h ^= std::hash<std::string>{}("hasBias") ^ std::hash<int>{}(param.hasBias);  
//         return h;
//     }
// };

template <typename T>
struct HashOpParam {
    size_t operator()(const T& param) const;
};

template <>
struct HashOpParam<atb::infer::LinearParam> {
    size_t operator()(const atb::infer::LinearParam& param) const {
        size_t h = 0;
        return h;
    }
};

template <>
struct HashOpParam<atb::infer::PagedAttentionParam> {
    size_t operator()(const atb::infer::PagedAttentionParam& param) const {
        size_t h = 0;
        h ^= std::hash<std::string>{}("kvHeadNum") ^ std::hash<int>{}(param.kvHeadNum);
        h ^= std::hash<std::string>{}("headNum") ^ std::hash<int>{}(param.headNum); 
        h ^= std::hash<std::string>{}("qkScale") ^ std::hash<int>{}(param.qkScale);
        return h;
    }
};

template <>
struct HashOpParam<atb::infer::ReshapeAndCacheParam> {
    size_t operator()(const atb::infer::ReshapeAndCacheParam& param) const {
        size_t h = 0;
        return h;
    }
};

template <>
struct HashOpParam<atb::infer::RopeParam> {
    size_t operator()(const atb::infer::RopeParam& param) const {
        size_t h = std::hash<int64_t>{}(param.rotaryCoeff);
        return h;
    }
};

template <>
struct HashOpParam<atb::infer::SelfAttentionParam> {
    size_t operator()(const atb::infer::SelfAttentionParam& param) const {
        size_t h = 0;
        h ^= std::hash<std::string>{}("headNum") ^ std::hash<int>{}(param.headNum);
        h ^= std::hash<std::string>{}("kvHeadNum") ^ std::hash<int>{}(param.kvHeadNum); 
        h ^= std::hash<std::string>{}("scale") ^ std::hash<int>{}(param.qkScale);
        return h;
    }
};

template <typename T>
size_t computeHash(const T& obj) {
    return HashOpParam<T>{}(obj);
}

template class OpParamCache<atb::infer::LinearParam>;
template class OpParamCache<atb::infer::PagedAttentionParam>;
template class OpParamCache<atb::infer::ReshapeAndCacheParam>;
template class OpParamCache<atb::infer::RopeParam>;
template class OpParamCache<atb::infer::SelfAttentionParam>;

}

#endif // OPPLUGIN_UTILS_ATB_PARAM_CACHE_H