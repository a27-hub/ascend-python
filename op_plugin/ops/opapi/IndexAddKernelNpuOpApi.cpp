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

#include <ATen/ATen.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

// 辅助函数：将 tensor 数据保存到二进制文件
void save_tensor_to_bin(const std::string& filename, const at::Tensor& tensor) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open file: " << filename << std::endl;
        return;
    }
    
    // 获取 tensor 数据（确保在 CPU 上且连续）
    at::Tensor cpu_tensor = tensor.cpu().contiguous();
    
    // 获取数据指针和大小
    void* data_ptr = cpu_tensor.data_ptr();
    size_t numel = cpu_tensor.numel();
    size_t element_size = cpu_tensor.element_size();
    size_t total_bytes = numel * element_size;
    
    // 写入元数据：shape, dtype, numel
    auto sizes = cpu_tensor.sizes();
    int64_t dim = sizes.size();
    file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    for (int64_t i = 0; i < dim; ++i) {
        int64_t size = sizes[i];
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    int64_t dtype = static_cast<int64_t>(cpu_tensor.scalar_type());
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    file.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
    
    // 写入实际数据
    file.write(reinterpret_cast<const char*>(data_ptr), total_bytes);
    
    file.close();
    std::cout << "[DEBUG] Saved tensor to " << filename << ", numel=" << numel 
              << ", total_bytes=" << total_bytes << std::endl;
}

// 辅助函数：保存 indices 信息
void save_indices_to_bin(const std::string& filename_prefix, 
                         const c10::List<c10::optional<at::Tensor>>& indices) {
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i].has_value()) {
            std::string filename = filename_prefix + "_index_" + std::to_string(i) + ".bin";
            save_tensor_to_bin(filename, indices[i].value());
        }
    }
}

// 辅助函数：保存标量 alpha
void save_alpha_to_bin(const std::string& filename, const at::Scalar& alpha) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open file: " << filename << std::endl;
        return;
    }
    
    double alpha_value = alpha.toDouble();
    file.write(reinterpret_cast<const char*>(&alpha_value), sizeof(alpha_value));
    file.close();
    std::cout << "[DEBUG] Saved alpha to " << filename << ", value=" << alpha_value << std::endl;
}

at::Tensor& index_add_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha,
    at::Tensor& result) {
    DO_COMPATIBILITY(aclnnIndexAdd, acl_op::index_add_out(self, dim, index, source, alpha, result));
    auto self_sizes = self.sizes().vec();
    auto source_sizes = source.sizes().vec();
    if (source.dim() != 0 && self.dim() != 0) {
        auto wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
        self_sizes.erase(self_sizes.begin() + wrapped_dim);
        source_sizes.erase(source_sizes.begin() + wrapped_dim);
    }
    TORCH_CHECK(self_sizes == source_sizes,
                "source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = ",
                self.sizes(),
                " source.shape = ",
                source.sizes(),
                OPS_ERROR(ErrCode::PARAM));
    at_npu::native::OpPreparation::check_tensor({self, index, source},
                                                result,
                                                result.scalar_type(),
                                                self.sizes());
    if (!result.is_same(self)) {
        result.copy_(self);
    }

    if (at::globalContext().deterministicAlgorithms()) {
        auto wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
        c10::List<c10::optional<at::Tensor>> indices;
        indices.reserve(self.dim());
        for (const auto i : c10::irange(self.dim())) {
            if (i == wrapped_dim) {
                indices.emplace_back(index.to(at::kLong));
            } else {
                indices.emplace_back(c10::nullopt);
            }
        }
        
        // 保存输入数据到二进制文件
        std::cout << "[DEBUG] ========== Saving input data for index_add_out ==========" << std::endl;
        save_tensor_to_bin("input_self.bin", self);
        save_tensor_to_bin("input_source.bin", source);
        save_tensor_to_bin("input_index.bin", index);
        save_tensor_to_bin("input_source_alpha.bin", source * alpha);
        save_tensor_to_bin("input_result_before.bin", result);
        save_indices_to_bin("input_indices", indices);
        save_alpha_to_bin("input_alpha.bin", alpha);
        
        // 保存额外信息：dim 和 wrapped_dim
        std::ofstream meta_file("input_meta.txt");
        if (meta_file.is_open()) {
            meta_file << "dim: " << dim << std::endl;
            meta_file << "wrapped_dim: " << wrapped_dim << std::endl;
            meta_file << "self.dim(): " << self.dim() << std::endl;
            meta_file << "deterministic: true" << std::endl;
            meta_file << "self.shape: " << self.sizes() << std::endl;
            meta_file << "source.shape: " << source.sizes() << std::endl;
            meta_file << "index.shape: " << index.sizes() << std::endl;
            meta_file << "result.shape: " << result.sizes() << std::endl;
            meta_file.close();
        }
        std::cout << "[DEBUG] ========== Input data saved ==========" << std::endl;
        
        result.index_put_(indices, source * alpha, true);
        return result;
    }

    EXEC_NPU_CMD(aclnnIndexAdd, result, dim, index, source, alpha, result);
    return result;
}

at::Tensor index_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
    DO_COMPATIBILITY(aclnnIndexAdd, acl_op::index_add(self, dim, index, source, alpha));
    auto self_sizes = self.sizes().vec();
    auto source_sizes = source.sizes().vec();
    if (source.dim() != 0 && self.dim() != 0) {
        auto wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
        self_sizes.erase(self_sizes.begin() + wrapped_dim);
        source_sizes.erase(source_sizes.begin() + wrapped_dim);
    }
    TORCH_CHECK(self_sizes == source_sizes,
                "source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = ",
                self.sizes(),
                " source.shape = ",
                source.sizes(),
                OPS_ERROR(ErrCode::PARAM));
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self.sizes(), self.options());

    if (at::globalContext().deterministicAlgorithms()) {
        auto wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
        result.copy_(self);
        c10::List<c10::optional<at::Tensor>> indices;
        indices.reserve(self.dim());
        for (const auto i : c10::irange(self.dim())) {
            if (i == wrapped_dim) {
                indices.emplace_back(index.to(at::kLong));
            } else {
                indices.emplace_back(c10::nullopt);
            }
        }
        
        // 保存输入数据到二进制文件
        std::cout << "[DEBUG] ========== Saving input data for index_add ==========" << std::endl;
        save_tensor_to_bin("input_self.bin", self);
        save_tensor_to_bin("input_source.bin", source);
        save_tensor_to_bin("input_index.bin", index);
        save_tensor_to_bin("input_source_alpha.bin", source * alpha);
        save_tensor_to_bin("input_result_before.bin", result);
        save_indices_to_bin("input_indices", indices);
        save_alpha_to_bin("input_alpha.bin", alpha);
        
        // 保存额外信息：dim 和 wrapped_dim
        std::ofstream meta_file("input_meta.txt");
        if (meta_file.is_open()) {
            meta_file << "dim: " << dim << std::endl;
            meta_file << "wrapped_dim: " << wrapped_dim << std::endl;
            meta_file << "self.dim(): " << self.dim() << std::endl;
            meta_file << "deterministic: true" << std::endl;
            meta_file << "self.shape: " << self.sizes() << std::endl;
            meta_file << "source.shape: " << source.sizes() << std::endl;
            meta_file << "index.shape: " << index.sizes() << std::endl;
            meta_file << "result.shape: " << result.sizes() << std::endl;
            meta_file.close();
        }
        std::cout << "[DEBUG] ========== Input data saved ==========" << std::endl;
        
        result.index_put_(indices, source * alpha, true);
        return result;
    }

    EXEC_NPU_CMD(aclnnIndexAdd, result.copy_(self), dim, index, source, alpha, result.copy_(self));
    return result;
}

at::Tensor index_add(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
    return op_api::index_add(self, dimname_to_position(self, dim), index, source, alpha);
}

} // namespace op_api