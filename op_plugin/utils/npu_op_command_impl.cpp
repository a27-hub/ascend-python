// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "npu_op_command.h"
#include "op_api_common_base.h"
#include <vector>
#include <tuple>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace op_plugin {
namespace {

template<int N>
int CallAclFunc(void* func, void** args);

template<> int CallAclFunc<1>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*)>(func)(args[0]);
}
template<> int CallAclFunc<2>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*)>(func)(args[0], args[1]);
}
template<> int CallAclFunc<3>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*)>(func)(args[0], args[1], args[2]);
}
template<> int CallAclFunc<4>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3]);
}
template<> int CallAclFunc<5>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4]);
}
template<> int CallAclFunc<6>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5]);
}
template<> int CallAclFunc<7>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
}
template<> int CallAclFunc<8>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
}
template<> int CallAclFunc<9>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
}
template<> int CallAclFunc<10>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]);
}
template<> int CallAclFunc<11>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]);
}
template<> int CallAclFunc<12>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]);
}
template<> int CallAclFunc<13>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]);
}
template<> int CallAclFunc<14>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13]);
}
template<> int CallAclFunc<15>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14]);
}
template<> int CallAclFunc<16>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15]);
}
template<> int CallAclFunc<17>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16]);
}
template<> int CallAclFunc<18>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17]);
}
template<> int CallAclFunc<19>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18]);
}
template<> int CallAclFunc<20>(void* func, void** args) {
    return reinterpret_cast<int(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19]);
}

inline int InvokeAclFunc(void* func, void** args, int n) {
    switch(n) {
        case 1: return CallAclFunc<1>(func, args);
        case 2: return CallAclFunc<2>(func, args);
        case 3: return CallAclFunc<3>(func, args);
        case 4: return CallAclFunc<4>(func, args);
        case 5: return CallAclFunc<5>(func, args);
        case 6: return CallAclFunc<6>(func, args);
        case 7: return CallAclFunc<7>(func, args);
        case 8: return CallAclFunc<8>(func, args);
        case 9: return CallAclFunc<9>(func, args);
        case 10: return CallAclFunc<10>(func, args);
        case 11: return CallAclFunc<11>(func, args);
        case 12: return CallAclFunc<12>(func, args);
        case 13: return CallAclFunc<13>(func, args);
        case 14: return CallAclFunc<14>(func, args);
        case 15: return CallAclFunc<15>(func, args);
        case 16: return CallAclFunc<16>(func, args);
        case 17: return CallAclFunc<17>(func, args);
        case 18: return CallAclFunc<18>(func, args);
        case 19: return CallAclFunc<19>(func, args);
        case 20: return CallAclFunc<20>(func, args);
        default:
            TORCH_CHECK(false, "Parameter count ", n, " exceeds max supported (20)");
            return -1;
    }
}

bool NeedsRelease(NpuParamType type) {
    return type == NpuParamType::kTensor ||
           type == NpuParamType::kScalar ||
           type == NpuParamType::kOptionalTensor ||
           type == NpuParamType::kOptionalScalar ||
           type == NpuParamType::kTensorList ||
           type == NpuParamType::kScalarList ||
           type == NpuParamType::kIntArrayRef ||
           type == NpuParamType::kOptionalIntArrayRef ||
           type == NpuParamType::kSymIntArrayRef ||
           type == NpuParamType::kOptionalSymIntArrayRef ||
           type == NpuParamType::kOptionalIntArrayRefV2 ||
           type == NpuParamType::kBoolArray;
}

void ReleaseAclParam(void* acl_param, NpuParamType type) {
    if (acl_param == nullptr) return;
    
    switch (type) {
        case NpuParamType::kTensor:
        case NpuParamType::kOptionalTensor:
            Release(static_cast<aclTensor*>(acl_param));
            break;
        case NpuParamType::kScalar:
        case NpuParamType::kOptionalScalar:
            Release(static_cast<aclScalar*>(acl_param));
            break;
        case NpuParamType::kIntArrayRef:
        case NpuParamType::kOptionalIntArrayRef:
        case NpuParamType::kSymIntArrayRef:
        case NpuParamType::kOptionalSymIntArrayRef:
        case NpuParamType::kOptionalIntArrayRefV2:
            Release(static_cast<aclIntArray*>(acl_param));
            break;
        case NpuParamType::kTensorList:
            Release(static_cast<aclTensorList*>(acl_param));
            break;
        case NpuParamType::kScalarList:
            Release(static_cast<aclScalarList*>(acl_param));
            break;
        case NpuParamType::kBoolArray:
            Release(static_cast<aclBoolArray*>(acl_param));
            break;
        default:
            break;
    }
}

void ReleaseAllAclParams(std::vector<std::pair<void*, NpuParamType>>& converted_params) {
    for (auto& p : converted_params) {
        ReleaseAclParam(p.first, p.second);
    }
}

struct ConvertedParam {
    std::vector<void*> acl_params;
    std::vector<std::pair<void*, NpuParamType>> need_release_params;
};

struct CopiedDataStorage {
    std::vector<TensorStructPtr> tensors;
    std::vector<std::vector<int64_t>> int_arrays;
    std::vector<std::vector<bool>> bool_arrays;
    std::vector<std::vector<TensorStructPtr>> tensor_lists;
    std::vector<std::vector<at::Scalar>> scalars;
    std::vector<c10::optional<std::vector<int64_t>>> opt_int_arrays;
};

void* ConvertParamToAcl(const NpuParam& param) {
    switch (param.type) {
        case NpuParamType::kTensor:
            return ConvertType(*static_cast<const at::Tensor*>(param.data));
        case NpuParamType::kScalar:
            return ConvertType(*static_cast<const at::Scalar*>(param.data));
        case NpuParamType::kIntArrayRef:
            return ConvertType(*static_cast<const at::IntArrayRef*>(param.data));
        case NpuParamType::kBool:
            return const_cast<void*>(param.data);
        case NpuParamType::kInt64:
            return const_cast<void*>(param.data);
        case NpuParamType::kDouble:
            return const_cast<void*>(param.data);
        case NpuParamType::kOptionalTensor:
            return ConvertType(*static_cast<const c10::optional<at::Tensor>*>(param.data));
        case NpuParamType::kOptionalIntArrayRef:
            return ConvertType(*static_cast<const c10::optional<at::IntArrayRef>*>(param.data));
        case NpuParamType::kOptionalScalar:
            return ConvertType(*static_cast<const c10::optional<at::Scalar>*>(param.data));
        case NpuParamType::kScalarType:
            return reinterpret_cast<void*>(static_cast<uintptr_t>(ConvertType(*static_cast<const at::ScalarType*>(param.data))));
        case NpuParamType::kString:
            return const_cast<void*>(param.data);
        case NpuParamType::kTensorList:
            return ConvertType(*static_cast<const at::TensorList*>(param.data));
        case NpuParamType::kScalarList:
            return ConvertType(*static_cast<const at::ArrayRef<at::Scalar>*>(param.data));
        case NpuParamType::kBoolArray:
            return ConvertType(*static_cast<const at::ArrayRef<bool>*>(param.data));
        case NpuParamType::kSymIntArrayRef:
            return ConvertType(*static_cast<const at::ArrayRef<c10::SymInt>*>(param.data));
        case NpuParamType::kOptionalSymIntArrayRef:
            return ConvertType(*static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data));
        case NpuParamType::kOptionalIntArrayRefV2:
            return ConvertType(*static_cast<const c10::OptionalIntArrayRef*>(param.data));
        case NpuParamType::kCharArray:
            return const_cast<void*>(param.data);
        default:
            TORCH_CHECK(false, "Unsupported parameter type: ", static_cast<int>(param.type));
            return nullptr;
    }
}

ConvertedParam ConvertParamsV1(const NpuParam* params, int nparams) {
    ConvertedParam result;
    result.acl_params.resize(nparams);
    
    for (int i = 0; i < nparams; i++) {
        void* acl_param = ConvertParamToAcl(params[i]);
        result.acl_params[i] = acl_param;
        if (acl_param != nullptr && NeedsRelease(params[i].type)) {
            result.need_release_params.push_back({acl_param, params[i].type});
        }
    }
    return result;
}

ConvertedParam ConvertParamsV2(const NpuParam* params, int nparams, CopiedDataStorage& storage) {
    ConvertedParam result;
    result.acl_params.resize(nparams);
    
    for (int i = 0; i < nparams; i++) {
        void* acl_param = nullptr;
        switch (params[i].type) {
            case NpuParamType::kTensor: {
                auto tensor_ptr = CopyTypeV2(*static_cast<const at::Tensor*>(params[i].data));
                storage.tensors.push_back(tensor_ptr);
                acl_param = ConvertTypeV2(tensor_ptr);
                break;
            }
            case NpuParamType::kScalar:
                acl_param = ConvertTypeV2(*static_cast<const at::Scalar*>(params[i].data));
                break;
            case NpuParamType::kIntArrayRef: {
                auto arr = CopyTypeV2(*static_cast<const at::IntArrayRef*>(params[i].data));
                storage.int_arrays.push_back(arr);
                acl_param = ConvertTypeV2(storage.int_arrays.back());
                break;
            }
            case NpuParamType::kBool:
                acl_param = const_cast<void*>(params[i].data);
                break;
            case NpuParamType::kInt64:
                acl_param = const_cast<void*>(params[i].data);
                break;
            case NpuParamType::kDouble:
                acl_param = const_cast<void*>(params[i].data);
                break;
            case NpuParamType::kOptionalTensor: {
                auto tensor_ptr = CopyTypeV2(*static_cast<const c10::optional<at::Tensor>*>(params[i].data));
                storage.tensors.push_back(tensor_ptr);
                acl_param = ConvertTypeV2(tensor_ptr);
                break;
            }
            case NpuParamType::kOptionalIntArrayRef: {
                auto arr = CopyTypeV2(*static_cast<const c10::optional<at::IntArrayRef>*>(params[i].data));
                storage.opt_int_arrays.push_back(arr);
                acl_param = ConvertTypeV2(storage.opt_int_arrays.back());
                break;
            }
            case NpuParamType::kOptionalScalar:
                acl_param = ConvertTypeV2(*static_cast<const c10::optional<at::Scalar>*>(params[i].data));
                break;
            case NpuParamType::kScalarType:
                acl_param = reinterpret_cast<void*>(static_cast<uintptr_t>(
                    ConvertTypeV2(*static_cast<const at::ScalarType*>(params[i].data))));
                break;
            case NpuParamType::kString:
                acl_param = const_cast<void*>(params[i].data);
                break;
            case NpuParamType::kTensorList: {
                auto tensor_list = CopyTypeV2(*static_cast<const at::TensorList*>(params[i].data));
                storage.tensor_lists.push_back(tensor_list);
                acl_param = ConvertTypeV2(storage.tensor_lists.back());
                break;
            }
            case NpuParamType::kScalarList: {
                auto scalar_list = CopyTypeV2(*static_cast<const at::ArrayRef<at::Scalar>*>(params[i].data));
                storage.scalars.push_back(scalar_list);
                acl_param = ConvertTypeV2(storage.scalars.back());
                break;
            }
            case NpuParamType::kBoolArray: {
                auto bools = CopyTypeV2(*static_cast<const at::ArrayRef<bool>*>(params[i].data));
                storage.bool_arrays.push_back(bools);
                acl_param = ConvertTypeV2(storage.bool_arrays.back());
                break;
            }
            case NpuParamType::kSymIntArrayRef: {
                auto arr = CopyTypeV2(*static_cast<const at::ArrayRef<c10::SymInt>*>(params[i].data));
                storage.int_arrays.push_back(arr);
                acl_param = ConvertTypeV2(storage.int_arrays.back());
                break;
            }
            case NpuParamType::kOptionalSymIntArrayRef: {
                auto arr = CopyTypeV2(*static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(params[i].data));
                storage.opt_int_arrays.push_back(arr);
                acl_param = ConvertTypeV2(storage.opt_int_arrays.back());
                break;
            }
            case NpuParamType::kOptionalIntArrayRefV2: {
                auto arr = CopyTypeV2(*static_cast<const c10::OptionalIntArrayRef*>(params[i].data));
                storage.opt_int_arrays.push_back(arr);
                acl_param = ConvertTypeV2(storage.opt_int_arrays.back());
                break;
            }
            case NpuParamType::kCharArray:
                acl_param = const_cast<void*>(params[i].data);
                break;
            default:
                TORCH_CHECK(false, "Unsupported parameter type in V2: ", static_cast<int>(params[i].type));
        }
        result.acl_params[i] = acl_param;
        if (acl_param != nullptr && NeedsRelease(params[i].type)) {
            result.need_release_params.push_back({acl_param, params[i].type});
        }
    }
    return result;
}

void AddParamToBufFromNpuParam(const NpuParam& param) {
    switch (param.type) {
        case NpuParamType::kTensor:
            add_param_to_buf(*static_cast<const at::Tensor*>(param.data));
            break;
        case NpuParamType::kScalar:
            add_param_to_buf(*static_cast<const at::Scalar*>(param.data));
            break;
        case NpuParamType::kIntArrayRef:
            add_param_to_buf(*static_cast<const at::IntArrayRef*>(param.data));
            break;
        case NpuParamType::kBool:
            add_param_to_buf(*static_cast<const bool*>(param.data));
            break;
        case NpuParamType::kInt64:
            add_param_to_buf(*static_cast<const int64_t*>(param.data));
            break;
        case NpuParamType::kDouble:
            add_param_to_buf(*static_cast<const double*>(param.data));
            break;
        case NpuParamType::kOptionalTensor:
            add_param_to_buf(*static_cast<const c10::optional<at::Tensor>*>(param.data));
            break;
        case NpuParamType::kOptionalIntArrayRef:
            add_param_to_buf(*static_cast<const c10::optional<at::IntArrayRef>*>(param.data));
            break;
        case NpuParamType::kOptionalScalar:
            add_param_to_buf(*static_cast<const c10::optional<at::Scalar>*>(param.data));
            break;
        case NpuParamType::kScalarType:
            add_param_to_buf(*static_cast<const at::ScalarType*>(param.data));
            break;
        case NpuParamType::kString:
            add_param_to_buf(*static_cast<const std::string*>(param.data));
            break;
        case NpuParamType::kTensorList:
            add_param_to_buf(*static_cast<const at::TensorList*>(param.data));
            break;
        case NpuParamType::kScalarList:
            add_param_to_buf(*static_cast<const at::ArrayRef<at::Scalar>*>(param.data));
            break;
        case NpuParamType::kBoolArray:
            add_param_to_buf(*static_cast<const at::ArrayRef<bool>*>(param.data));
            break;
        case NpuParamType::kSymIntArrayRef:
            add_param_to_buf(*static_cast<const at::ArrayRef<c10::SymInt>*>(param.data));
            break;
        case NpuParamType::kOptionalSymIntArrayRef:
            add_param_to_buf(*static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data));
            break;
        case NpuParamType::kOptionalIntArrayRefV2:
            add_param_to_buf(*static_cast<const c10::OptionalIntArrayRef*>(param.data));
            break;
        case NpuParamType::kCharArray:
            add_param_to_buf(static_cast<const char*>(param.data));
            break;
        default:
            TORCH_CHECK(false, "Unsupported parameter type for cache hash: ", static_cast<int>(param.type));
    }
}

void AddParamToBufV2FromNpuParam(const NpuParam& param) {
    switch (param.type) {
        case NpuParamType::kTensor: {
            auto tensor_ptr = CopyTypeV2(*static_cast<const at::Tensor*>(param.data));
            add_param_to_buf_v2(tensor_ptr);
            break;
        }
        case NpuParamType::kScalar:
            add_param_to_buf_v2(*static_cast<const at::Scalar*>(param.data));
            break;
        case NpuParamType::kIntArrayRef: {
            auto arr = CopyTypeV2(*static_cast<const at::IntArrayRef*>(param.data));
            add_param_to_buf_v2(arr);
            break;
        }
        case NpuParamType::kBool:
            add_param_to_buf_v2(*static_cast<const bool*>(param.data));
            break;
        case NpuParamType::kInt64:
            add_param_to_buf_v2(*static_cast<const int64_t*>(param.data));
            break;
        case NpuParamType::kDouble:
            add_param_to_buf_v2(*static_cast<const double*>(param.data));
            break;
        case NpuParamType::kOptionalTensor: {
            auto tensor_ptr = CopyTypeV2(*static_cast<const c10::optional<at::Tensor>*>(param.data));
            add_param_to_buf_v2(tensor_ptr);
            break;
        }
        case NpuParamType::kOptionalIntArrayRef: {
            auto arr = CopyTypeV2(*static_cast<const c10::optional<at::IntArrayRef>*>(param.data));
            add_param_to_buf_v2(arr);
            break;
        }
        case NpuParamType::kOptionalScalar:
            add_param_to_buf_v2(*static_cast<const c10::optional<at::Scalar>*>(param.data));
            break;
        case NpuParamType::kScalarType:
            add_param_to_buf_v2(*static_cast<const at::ScalarType*>(param.data));
            break;
        case NpuParamType::kString:
            add_param_to_buf_v2(*static_cast<const std::string*>(param.data));
            break;
        case NpuParamType::kTensorList: {
            auto tensor_list = CopyTypeV2(*static_cast<const at::TensorList*>(param.data));
            add_param_to_buf_v2(tensor_list);
            break;
        }
        case NpuParamType::kScalarList: {
            auto scalars = CopyTypeV2(*static_cast<const at::ArrayRef<at::Scalar>*>(param.data));
            add_param_to_buf_v2(scalars);
            break;
        }
        case NpuParamType::kBoolArray: {
            auto bools = CopyTypeV2(*static_cast<const at::ArrayRef<bool>*>(param.data));
            add_param_to_buf_v2(bools);
            break;
        }
        case NpuParamType::kSymIntArrayRef: {
            auto arr = CopyTypeV2(*static_cast<const at::ArrayRef<c10::SymInt>*>(param.data));
            add_param_to_buf_v2(arr);
            break;
        }
        case NpuParamType::kOptionalSymIntArrayRef: {
            auto arr = CopyTypeV2(*static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data));
            add_param_to_buf_v2(arr);
            break;
        }
        case NpuParamType::kOptionalIntArrayRefV2: {
            auto arr = CopyTypeV2(*static_cast<const c10::OptionalIntArrayRef*>(param.data));
            add_param_to_buf_v2(arr);
            break;
        }
        case NpuParamType::kCharArray:
            add_param_to_buf_v2(static_cast<const char*>(param.data));
            break;
        default:
            TORCH_CHECK(false, "Unsupported parameter type for cache hash V2: ", static_cast<int>(param.type));
    }
}

struct ApiFuncCache {
    void* opApiFuncAddr = nullptr;
    void* getWorkspaceSizeFuncAddr = nullptr;
};

std::unordered_map<std::string, ApiFuncCache>& GetApiFuncCacheMap() {
    static std::unordered_map<std::string, ApiFuncCache> cache;
    return cache;
}

std::mutex& GetApiFuncCacheMutex() {
    static std::mutex mtx;
    return mtx;
}

ApiFuncCache& GetOrInitApiFuncCache(const char* api_name) {
    std::lock_guard<std::mutex> lock(GetApiFuncCacheMutex());
    auto& cache = GetApiFuncCacheMap();
    auto it = cache.find(api_name);
    if (it != cache.end()) {
        return it->second;
    }
    
    auto& entry = cache[api_name];
    std::string workspace_api_name = std::string(api_name) + "GetWorkspaceSize";
    GetApiFunc(api_name, workspace_api_name.c_str(), entry.opApiFuncAddr, entry.getWorkspaceSizeFuncAddr);
    return entry;
}

struct WorkspaceAndExecutor {
    uint64_t size;
    uint64_t* size_addr;
    aclOpExecutor* executor;
    aclOpExecutor** executor_addr;
};

WorkspaceAndExecutor CreateWorkspaceAndExecutor() {
    WorkspaceAndExecutor result;
    result.size = 0;
    result.size_addr = &result.size;
    result.executor = nullptr;
    result.executor_addr = &result.executor;
    return result;
}

int CallGetWorkspaceSize(void* getWorkspaceSizeFuncAddr, 
                         const std::vector<void*>& acl_params,
                         WorkspaceAndExecutor& we) {
    std::vector<void*> ws_params = acl_params;
    ws_params.push_back(static_cast<void*>(we.size_addr));
    ws_params.push_back(static_cast<void*>(we.executor_addr));
    return InvokeAclFunc(getWorkspaceSizeFuncAddr, ws_params.data(), ws_params.size());
}

}

TORCH_NPU_API bool HitCacheExtFromParams(aclrtStream acl_stream, const char* aclnn_api, void* phrase2,
                                          const NpuParam* params, int nparams);
TORCH_NPU_API bool HitCacheV2ExtFromParams(aclrtStream acl_stream, const char* aclnn_api, void* phrase2,
                                            const NpuParam* params, int nparams, int* api_ret,
                                            const CacheParams& cache_params);

}

extern "C" {

void ExecNpuCmdExtImpl(const char* api_name, const op_plugin::NpuParam* params, int nparams) {
    using namespace op_plugin;
    
    static const auto task_queue_enable = OpApiGetTaskQueueEnable();
    
    auto& funcCache = GetOrInitApiFuncCache(api_name);
    void* opApiFuncAddr = funcCache.opApiFuncAddr;
    void* getWorkspaceSizeFuncAddr = funcCache.getWorkspaceSizeFuncAddr;
    
    if (task_queue_enable == 2) {
        CopiedDataStorage storage;
        ConvertedParam converted = ConvertParamsV2(params, nparams, storage);
        
        auto acl_stream = GetAclStream();
        CacheParams cache_params = GetCacheParams();
        
        auto acl_call = [=]() mutable -> int {
            InitExecSubTheadCtx(acl_stream);
            int api_ret = 0;
            
            if (HitCacheV2ExtFromParams(acl_stream, api_name, opApiFuncAddr, params, nparams, &api_ret, cache_params)) {
                return api_ret;
            }
            
            SetExecConfigV2(cache_params);
            InitExecCommonCtx();
            
            auto we = CreateWorkspaceAndExecutor();
            CallGetWorkspaceSize(getWorkspaceSizeFuncAddr, converted.acl_params, we);
            
            api_ret = ExecuteApiFunc(opApiFuncAddr, acl_stream, we.size, we.executor);
            
            ReleaseAllAclParams(converted.need_release_params);
            UnInitExecCommonCtx();
            return api_ret;
        };
        
        RunAclCall(api_name, acl_call);
    } else {
        ConvertedParam converted = ConvertParamsV1(params, nparams);
        
        InitExecCommonCtx();
        auto acl_stream = GetAclStream();
        
        if (HitCacheExtFromParams(acl_stream, api_name, opApiFuncAddr, params, nparams)) {
            ReleaseAllAclParams(converted.need_release_params);
            UnInitExecCommonCtx();
            return;
        }
        
        SetExecConfig();
        
        auto we = CreateWorkspaceAndExecutor();
        CallGetWorkspaceSize(getWorkspaceSizeFuncAddr, converted.acl_params, we);
        
        auto acl_call = [=]() mutable -> int {
            InitExecSubTheadCtx(acl_stream);
            auto api_ret = ExecuteApiFunc(opApiFuncAddr, acl_stream, we.size, we.executor);
            ReleaseAllAclParams(converted.need_release_params);
            UnInitExecCommonCtx();
            return api_ret;
        };
        
        RunAclCall(api_name, acl_call);
    }
}

}

namespace op_plugin {

TORCH_NPU_API bool HitCacheExtFromParams(aclrtStream acl_stream, const char* aclnn_api, void* phrase2,
                                          const NpuParam* params, int nparams) {
    if (!CheckAndInitFunc(aclnn_api)) {
        return false;
    }
    CacheParams cache_params = GetCacheParams();
    AddCacheConfigParams(acl_stream, cache_params);
    for (int i = 0; i < nparams; i++) {
        AddParamToBufFromNpuParam(params[i]);
    }
    add_param_to_buf(std::string(aclnn_api));
    return ExecuteCachedOp(acl_stream, aclnn_api, phrase2);
}

TORCH_NPU_API bool HitCacheV2ExtFromParams(aclrtStream acl_stream, const char* aclnn_api, void* phrase2,
                                            const NpuParam* params, int nparams, int* api_ret,
                                            const CacheParams& cache_params) {
    if (!CheckAndInitFuncV2(aclnn_api)) {
        return false;
    }
    AddCacheConfigParamsV2(acl_stream, cache_params, aclnn_api);
    for (int i = 0; i < nparams; i++) {
        AddParamToBufV2FromNpuParam(params[i]);
    }
    return ExecuteCachedOpV2(acl_stream, aclnn_api, phrase2, api_ret);
}

}
