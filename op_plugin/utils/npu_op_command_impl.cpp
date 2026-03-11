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

#include "op_plugin/utils/npu_op_command.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/op_log.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include <tuple>
#include <vector>
#include <memory>
#include <string>

namespace op_plugin {

namespace {

struct ConvertedParamsHolder {
    std::vector<aclTensor*> acl_tensors;
    std::vector<aclScalar*> acl_scalars;
    std::vector<aclIntArray*> acl_int_arrays;
    std::vector<aclBoolArray*> acl_bool_arrays;
    std::vector<aclTensorList*> acl_tensor_lists;
    std::vector<aclScalarList*> acl_scalar_lists;
    std::vector<TensorStructPtr> tensor_structs;
    std::vector<std::vector<int64_t>> int_vectors;
    std::vector<std::vector<bool>> bool_vectors;
    std::vector<std::vector<TensorStructPtr>> tensor_list_structs;
    std::vector<std::vector<at::Scalar>> scalar_vectors;
    std::vector<c10::optional<std::vector<int64_t>>> opt_int_vectors;
    std::vector<char*> char_ptrs;
    
    ~ConvertedParamsHolder() {
        for (auto* p : acl_tensors) { Release(p); }
        for (auto* p : acl_scalars) { Release(p); }
        for (auto* p : acl_int_arrays) { Release(p); }
        for (auto* p : acl_bool_arrays) { Release(p); }
        for (auto* p : acl_tensor_lists) { Release(p); }
        for (auto* p : acl_scalar_lists) { Release(p); }
    }
};

void* ConvertNpuParam(const NpuParam& param, ConvertedParamsHolder& holder) {
    switch (param.type) {
        case kTensor: {
            auto* tensor = static_cast<const at::Tensor*>(param.data);
            auto* acl_tensor = ConvertType(*tensor);
            holder.acl_tensors.push_back(acl_tensor);
            return acl_tensor;
        }
        case kScalar: {
            auto* scalar = static_cast<const at::Scalar*>(param.data);
            auto* acl_scalar = ConvertType(*scalar);
            holder.acl_scalars.push_back(acl_scalar);
            return acl_scalar;
        }
        case kIntArrayRef: {
            auto* array = static_cast<const at::IntArrayRef*>(param.data);
            auto* acl_array = ConvertType(*array);
            holder.acl_int_arrays.push_back(acl_array);
            return acl_array;
        }
        case kBool:
            return const_cast<void*>(param.data);
        case kInt64:
            return const_cast<void*>(param.data);
        case kDouble:
            return const_cast<void*>(param.data);
        case kTensorList: {
            auto* tensor_list = static_cast<const at::TensorList*>(param.data);
            auto* acl_tensor_list = ConvertType(*tensor_list);
            holder.acl_tensor_lists.push_back(acl_tensor_list);
            return acl_tensor_list;
        }
        case kScalarList: {
            auto* scalar_list = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            auto* acl_scalar_list = ConvertType(*scalar_list);
            holder.acl_scalar_lists.push_back(acl_scalar_list);
            return acl_scalar_list;
        }
        case kString: {
            const std::string* str = static_cast<const std::string*>(param.data);
            holder.char_ptrs.push_back(const_cast<char*>(str->c_str()));
            return holder.char_ptrs.back();
        }
        case kScalarType: {
            auto scalar_type = *static_cast<const at::ScalarType*>(param.data);
            auto acl_dtype = ConvertType(scalar_type);
            return reinterpret_cast<void*>(static_cast<intptr_t>(acl_dtype));
        }
        case kOptionalTensor: {
            auto* opt_tensor = static_cast<const c10::optional<at::Tensor>*>(param.data);
            if (opt_tensor->has_value() && opt_tensor->value().defined()) {
                auto* acl_tensor = ConvertType(opt_tensor->value());
                holder.acl_tensors.push_back(acl_tensor);
                return acl_tensor;
            }
            return nullptr;
        }
        case kOptionalIntArrayRef: {
            auto* opt_array = static_cast<const c10::optional<at::IntArrayRef>*>(param.data);
            if (opt_array->has_value()) {
                auto* acl_array = ConvertType(opt_array->value());
                holder.acl_int_arrays.push_back(acl_array);
                return acl_array;
            }
            return nullptr;
        }
        case kOptionalScalar: {
            auto* opt_scalar = static_cast<const c10::optional<at::Scalar>*>(param.data);
            if (opt_scalar->has_value()) {
                auto* acl_scalar = ConvertType(opt_scalar->value());
                holder.acl_scalars.push_back(acl_scalar);
                return acl_scalar;
            }
            return nullptr;
        }
        case kBoolArray: {
            auto* array = static_cast<const at::ArrayRef<bool>*>(param.data);
            auto* acl_array = ConvertType(*array);
            holder.acl_bool_arrays.push_back(acl_array);
            return acl_array;
        }
        case kSymIntArrayRef: {
            auto* array = static_cast<const at::ArrayRef<c10::SymInt>*>(param.data);
            auto* acl_array = ConvertType(*array);
            holder.acl_int_arrays.push_back(acl_array);
            return acl_array;
        }
        case kOptionalSymIntArrayRef: {
            auto* opt_array = static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data);
            if (opt_array->has_value()) {
                auto* acl_array = ConvertType(opt_array->value());
                holder.acl_int_arrays.push_back(acl_array);
                return acl_array;
            }
            return nullptr;
        }
        case kNullptr:
        default:
            return nullptr;
    }
}

void* CopyNpuParamV2(const NpuParam& param, ConvertedParamsHolder& holder) {
    switch (param.type) {
        case kTensor: {
            auto* tensor = static_cast<const at::Tensor*>(param.data);
            auto tensor_struct = CopyTypeV2(*tensor);
            holder.tensor_structs.push_back(tensor_struct);
            auto* acl_tensor = ConvertTypeV2(tensor_struct);
            holder.acl_tensors.push_back(acl_tensor);
            return acl_tensor;
        }
        case kScalar: {
            auto* scalar = static_cast<const at::Scalar*>(param.data);
            auto* acl_scalar = ConvertTypeV2(*scalar);
            holder.acl_scalars.push_back(acl_scalar);
            return acl_scalar;
        }
        case kIntArrayRef: {
            auto* array = static_cast<const at::IntArrayRef*>(param.data);
            holder.int_vectors.push_back(array->vec());
            auto* acl_array = ConvertTypeV2(holder.int_vectors.back());
            holder.acl_int_arrays.push_back(acl_array);
            return acl_array;
        }
        case kBool:
            return const_cast<void*>(param.data);
        case kInt64:
            return const_cast<void*>(param.data);
        case kDouble:
            return const_cast<void*>(param.data);
        case kTensorList: {
            auto* tensor_list = static_cast<const at::TensorList*>(param.data);
            std::vector<TensorStructPtr> tensor_struct_list;
            for (const auto& tensor : *tensor_list) {
                auto tensor_struct = CopyTypeV2(tensor);
                holder.tensor_structs.push_back(tensor_struct);
                tensor_struct_list.push_back(tensor_struct);
            }
            holder.tensor_list_structs.push_back(tensor_struct_list);
            auto* acl_tensor_list = ConvertTypeV2(tensor_struct_list);
            holder.acl_tensor_lists.push_back(acl_tensor_list);
            return acl_tensor_list;
        }
        case kScalarList: {
            auto* scalar_list = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            holder.scalar_vectors.push_back(scalar_list->vec());
            auto* acl_scalar_list = ConvertTypeV2(holder.scalar_vectors.back());
            holder.acl_scalar_lists.push_back(acl_scalar_list);
            return acl_scalar_list;
        }
        case kString: {
            const std::string* str = static_cast<const std::string*>(param.data);
            holder.char_ptrs.push_back(const_cast<char*>(str->c_str()));
            return holder.char_ptrs.back();
        }
        case kScalarType: {
            auto scalar_type = *static_cast<const at::ScalarType*>(param.data);
            auto acl_dtype = ConvertTypeV2(scalar_type);
            return reinterpret_cast<void*>(static_cast<intptr_t>(acl_dtype));
        }
        case kOptionalTensor: {
            auto* opt_tensor = static_cast<const c10::optional<at::Tensor>*>(param.data);
            if (opt_tensor->has_value() && opt_tensor->value().defined()) {
                auto tensor_struct = CopyTypeV2(opt_tensor->value());
                holder.tensor_structs.push_back(tensor_struct);
                auto* acl_tensor = ConvertTypeV2(tensor_struct);
                holder.acl_tensors.push_back(acl_tensor);
                return acl_tensor;
            }
            return nullptr;
        }
        case kOptionalIntArrayRef: {
            auto* opt_array = static_cast<const c10::optional<at::IntArrayRef>*>(param.data);
            if (opt_array->has_value()) {
                holder.opt_int_vectors.push_back(opt_array->value().vec());
                auto* acl_array = ConvertTypeV2(holder.opt_int_vectors.back());
                holder.acl_int_arrays.push_back(acl_array);
                return acl_array;
            }
            return nullptr;
        }
        case kOptionalScalar: {
            auto* opt_scalar = static_cast<const c10::optional<at::Scalar>*>(param.data);
            if (opt_scalar->has_value()) {
                auto* acl_scalar = ConvertTypeV2(opt_scalar->value());
                holder.acl_scalars.push_back(acl_scalar);
                return acl_scalar;
            }
            return nullptr;
        }
        case kBoolArray: {
            auto* array = static_cast<const at::ArrayRef<bool>*>(param.data);
            holder.bool_vectors.push_back(array->vec());
            auto* acl_array = ConvertTypeV2(holder.bool_vectors.back());
            holder.acl_bool_arrays.push_back(acl_array);
            return acl_array;
        }
        case kSymIntArrayRef: {
            auto* array = static_cast<const at::ArrayRef<c10::SymInt>*>(param.data);
            holder.int_vectors.push_back(CopyTypeV2(*array));
            auto* acl_array = ConvertTypeV2(holder.int_vectors.back());
            holder.acl_int_arrays.push_back(acl_array);
            return acl_array;
        }
        case kOptionalSymIntArrayRef: {
            auto* opt_array = static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data);
            if (opt_array->has_value()) {
                holder.opt_int_vectors.push_back(CopyTypeV2(opt_array->value()));
                auto* acl_array = ConvertTypeV2(holder.opt_int_vectors.back());
                holder.acl_int_arrays.push_back(acl_array);
                return acl_array;
            }
            return nullptr;
        }
        case kNullptr:
        default:
            return nullptr;
    }
}

int CallGetWorkspaceSize(void* func_addr, const std::vector<void*>& params) {
    typedef int (*GetWorkspaceSizeFunc)(...);
    auto func = reinterpret_cast<GetWorkspaceSizeFunc>(func_addr);
    switch (params.size()) {
        case 1: return ((int (*)(void*))func)(params[0]);
        case 2: return ((int (*)(void*, void*))func)(params[0], params[1]);
        case 3: return ((int (*)(void*, void*, void*))func)(params[0], params[1], params[2]);
        case 4: return ((int (*)(void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3]);
        case 5: return ((int (*)(void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4]);
        case 6: return ((int (*)(void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5]);
        case 7: return ((int (*)(void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6]);
        case 8: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]);
        case 9: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]);
        case 10: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9]);
        case 11: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10]);
        case 12: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11]);
        case 13: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12]);
        case 14: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13]);
        case 15: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14]);
        case 16: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15]);
        case 17: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16]);
        case 18: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17]);
        case 19: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18]);
        case 20: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19]);
        case 21: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20]);
        case 22: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21]);
        case 23: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22]);
        case 24: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23]);
        case 25: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24]);
        case 26: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25]);
        case 27: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26]);
        case 28: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27]);
        case 29: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28]);
        case 30: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29]);
        case 31: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30]);
        case 32: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31]);
        case 33: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32]);
        case 34: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33]);
        case 35: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34]);
        case 36: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35]);
        case 37: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36]);
        case 38: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37]);
        case 39: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38]);
        case 40: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39]);
        case 41: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40]);
        case 42: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41]);
        case 43: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42]);
        case 44: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43]);
        case 45: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44]);
        case 46: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45]);
        case 47: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45], params[46]);
        case 48: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45], params[46], params[47]);
        case 49: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45], params[46], params[47], params[48]);
        case 50: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45], params[46], params[47], params[48], params[49]);
        default:
            TORCH_CHECK(false, "Too many parameters: ", params.size(), OPS_ERROR(ErrCode::PARAM));
            return -1;
    }
}

}

extern "C" {

TORCH_NPU_API void ExecNpuCmdExtImpl(const char* api_name, const NpuParam* params, int nparams) {
    static const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
    
    std::string workspace_api_name = std::string(api_name) + "GetWorkspaceSize";
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workspace_api_name.c_str());
    static const auto opApiFuncAddr = GetOpApiFuncAddr(api_name);
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
    
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
        api_name, " or ", workspace_api_name, " not in ", GetOpApiLibName(),
        ", or ", GetOpApiLibName(), " not found.",
        OPS_ERROR(ErrCode::PTR));
    
    if (task_queue_enable == 2) {
        OP_EXEC_LOG_WITH_TASK_QUEUE(api_name, "EXEC_NPU_CMD_EX", "2", "params_count:", nparams);
        
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        if (c10_npu::check_enqueue_need_use(acl_stream)) {
            c10_npu::UseStreamResInCurrentThread(acl_stream);
        }
        
        ConvertedParamsHolder holder;
        std::vector<void*> copied_params;
        copied_params.reserve(nparams);
        for (int i = 0; i < nparams; ++i) {
            copied_params.push_back(CopyNpuParamV2(params[i], holder));
        }
        
        auto deterministic_status = at::globalContext().deterministicAlgorithms();
        uint32_t aic_num = UINT32_MAX;
        uint32_t aiv_num = UINT32_MAX;
        if (c10_npu::is_core_control_enabled()) {
            aic_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);
            aiv_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);
        }
        
        auto acl_call = [api_name, copied_params, acl_stream, deterministic_status, aic_num, aiv_num,
                         getWorkspaceSizeFuncAddr, opApiFuncAddr, initMemAddr, unInitMemAddr, releaseMemAddr,
                         holder = std::move(holder)]() mutable -> int {
            if (c10_npu::check_dequeue_need_use(acl_stream)) {
                c10_npu::UseStreamResInCurrentThread(acl_stream);
            }
            
            uint64_t workspace_size = 0;
            uint64_t* workspace_size_addr = &workspace_size;
            aclOpExecutor* executor = nullptr;
            aclOpExecutor** executor_addr = &executor;
            InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
            UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
            int api_ret = 0;
            
            at_npu::native::SetDeterministicOps(deterministic_status);
            if (initMemFunc) {
                initMemFunc(nullptr, false);
            }
            
            std::vector<void*> converted_params = copied_params;
            converted_params.push_back(workspace_size_addr);
            converted_params.push_back(executor_addr);
            
            auto workspace_status = CallGetWorkspaceSize(getWorkspaceSizeFuncAddr, converted_params);
            NPU_CHECK_ERROR(workspace_status, (std::string("call ") + api_name + " failed").c_str());
            
            void* workspace_addr = nullptr;
            at::Tensor workspace_tensor;
            if (workspace_size != 0) {
                workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, acl_stream);
                workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
            }
            
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
            api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
            NPU_CHECK_ERROR(api_ret, (std::string("call ") + api_name + " failed").c_str());
            
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
            if (releaseMemFunc) {
                releaseMemFunc(nullptr, false);
            }
            if (unInitMemFunc) {
                unInitMemFunc(nullptr, false);
            }
            UnInitCacheThreadLocal();
            
            return api_ret;
        };
        
        at_npu::native::OpCommand::RunOpApiV2(api_name, acl_call);
    } else {
        OP_EXEC_LOG_WITH_TASK_QUEUE(api_name, "EXEC_NPU_CMD_EX", "1", "params_count:", nparams);
        
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        if (c10_npu::check_enqueue_need_use(acl_stream)) {
            c10_npu::UseStreamResInCurrentThread(acl_stream);
        }
        
        uint64_t workspace_size = 0;
        uint64_t* workspace_size_addr = &workspace_size;
        aclOpExecutor* executor = nullptr;
        aclOpExecutor** executor_addr = &executor;
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
        
        at_npu::native::SetDeterministic();
        if (initMemFunc) {
            initMemFunc(nullptr, false);
        }
        
        ConvertedParamsHolder holder;
        std::vector<void*> converted_params;
        converted_params.reserve(nparams + 2);
        for (int i = 0; i < nparams; ++i) {
            converted_params.push_back(ConvertNpuParam(params[i], holder));
        }
        converted_params.push_back(workspace_size_addr);
        converted_params.push_back(executor_addr);
        
        auto workspace_status = CallGetWorkspaceSize(getWorkspaceSizeFuncAddr, converted_params);
        NPU_CHECK_ERROR(workspace_status, (std::string("call ") + api_name + " failed").c_str());
        
        void* workspace_addr = nullptr;
        at::Tensor workspace_tensor;
        if (workspace_size != 0) {
            workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);
            workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
        }
        
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor,
                         opApiFuncAddr, releaseMemAddr, api_name, holder = std::move(holder)]() mutable -> int {
            if (c10_npu::check_dequeue_need_use(acl_stream)) {
                c10_npu::UseStreamResInCurrentThread(acl_stream);
            }
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
            NPU_CHECK_ERROR(api_ret, (std::string("call ") + api_name + " failed").c_str());
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
            if (releaseMemFunc) {
                releaseMemFunc(nullptr, false);
            }
            return api_ret;
        };
        
        at_npu::native::OpCommand::RunOpApiV2(api_name, acl_call);
        
        if (unInitMemFunc) {
            unInitMemFunc(nullptr, false);
        }
        UnInitCacheThreadLocal();
    }
}

}

}
