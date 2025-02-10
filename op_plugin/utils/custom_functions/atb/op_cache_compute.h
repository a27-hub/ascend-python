#ifndef OPPLUGIN_UTILS_ATB_PARAM_OP_CACHE_COMPUTE_H
#define OPPLUGIN_UTILS_ATB_PARAM_OP_CACHE_COMPUTE_H

#include <unordered_map>
#include <mutex>
#include <memory>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "op_plugin/third_party/atb/inc/atb_infer.h"


namespace op_api {

template <typename T>
struct HashOpParam {
    size_t operator()(const T& param) const;
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
        h ^= std::hash<std::string>{}("maskType") ^ std::hash<int>{}(param.maskType);
        h ^= std::hash<std::string>{}("batchRunStatusEnable") ^ std::hash<bool>{}(param.batchRunStatusEnable);
        h ^= std::hash<std::string>{}("quantType") ^ std::hash<int>{}(param.quantType);
        h ^= std::hash<std::string>{}("outDataType") ^ std::hash<int>{}(param.outDataType);
        h ^= std::hash<std::string>{}("hasQuantOffset") ^ std::hash<bool>{}(param.hasQuantOffset);
        h ^= std::hash<std::string>{}("compressType") ^ std::hash<int>{}(param.compressType);
        h ^= std::hash<std::string>{}("calcType") ^ std::hash<int>{}(param.calcType);
        h ^= std::hash<std::string>{}("scaleType") ^ std::hash<int>{}(param.scaleType);
        h ^= std::hash<std::string>{}("inputLayout") ^ std::hash<int>{}(param.inputLayout);
        h ^= std::hash<std::string>{}("mlaVHeadSize") ^ std::hash<int>{}(param.mlaVHeadSize);
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

template <>
struct HashOpParam<atb::infer::ElewiseParam> {
    size_t operator()(const atb::infer::ElewiseParam& param) const {
        size_t h = 0;
        h ^= std::hash<std::string>{}("elewiseType") ^ std::hash<int>{}(param.elewiseType);
        h ^= std::hash<std::string>{}("inputScale") ^ std::hash<int>{}(param.quantParam.inputScale); 
        h ^= std::hash<std::string>{}("asymmetric") ^ std::hash<int>{}(param.quantParam.asymmetric);
        h ^= std::hash<std::string>{}("inputOffset") ^ std::hash<int>{}(param.quantParam.inputOffset);
        h ^= std::hash<std::string>{}("varAttr") ^ std::hash<int>{}(param.mulsParam.varAttr);
        h ^= std::hash<std::string>{}("outTensorType") ^ std::hash<int>{}(param.outTensorType);
        return h;
    }
};

template <typename T>
size_t computeHash(const T& obj) {
    return HashOpParam<T>{}(obj);
}

}

#endif // OPPLUGIN_UTILS_ATB_PARAM_OP_CACHE_COMPUTE_H