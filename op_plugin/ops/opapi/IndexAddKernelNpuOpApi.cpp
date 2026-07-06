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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>

namespace op_api {

namespace {
    // 生成带时间戳的文件名
    std::string get_timestamp_filename(const std::string& prefix = "tensor_data") {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << prefix << "_" 
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
           << "_" << std::setfill('0') << std::setw(3) << ms.count()
           << ".txt";
        return ss.str();
    }
    
    // 保存张量完整数据到文件
    void save_tensor_to_file(const at::Tensor& tensor, 
                             const std::string& name,
                             const std::string& prefix = "index_add_debug") {
        if (!tensor.defined()) {
            std::cout << "  " << name << ": [undefined tensor]" << std::endl;
            return;
        }
        
        // 先打印前10个元素到控制台
        std::cout << "  " << name << ": shape=" << tensor.sizes() 
                  << ", dtype=" << tensor.dtype() 
                  << ", device=" << tensor.device()
                  << ", numel=" << tensor.numel() << std::endl;
        
        // 打印前10个元素
        if (tensor.numel() > 0) {
            auto flat = tensor.flatten();
            int64_t print_num = std::min<int64_t>(10, flat.numel());
            auto slice = flat.slice(0, 0, print_num);
            std::cout << "  " << name << " first " << print_num << " values: " << slice << std::endl;
        }
        
        // 保存完整数据到文件
        try {
            std::string filename = get_timestamp_filename(prefix + "_" + name);
            std::ofstream file(filename);
            if (file.is_open()) {
                file << "=== Tensor: " << name << " ===" << std::endl;
                file << "shape: " << tensor.sizes() << std::endl;
                file << "dtype: " << tensor.dtype() << std::endl;
                file << "device: " << tensor.device() << std::endl;
                file << "numel: " << tensor.numel() << std::endl;
                file << "values:" << std::endl;
                
                // 将张量转为CPU以便保存
                at::Tensor cpu_tensor = tensor;
                if (tensor.device().type() != at::kCPU) {
                    cpu_tensor = tensor.cpu();
                }
                
                // 保存所有数据
                auto flat = cpu_tensor.flatten();
                for (int64_t i = 0; i < flat.numel(); ++i) {
                    if (i > 0) file << ", ";
                    if (i % 10 == 0 && i > 0) file << std::endl;
                    
                    // 根据类型打印
                    if (flat.scalar_type() == at::kFloat) {
                        file << flat[i].item<float>();
                    } else if (flat.scalar_type() == at::kDouble) {
                        file << flat[i].item<double>();
                    } else if (flat.scalar_type() == at::kLong) {
                        file << flat[i].item<long>();
                    } else if (flat.scalar_type() == at::kInt) {
                        file << flat[i].item<int>();
                    } else if (flat.scalar_type() == at::kBool) {
                        file << (flat[i].item<bool>() ? "true" : "false");
                    } else {
                        file << flat[i].item<float>();
                    }
                }
                file << std::endl;
                file.close();
                std::cout << "  " << name << " full data saved to: " << filename << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  Failed to save " << name << " data: " << e.what() << std::endl;
        }
    }
    
    void print_debug_info(const std::string& location,
                         const at::Tensor& self,
                         const at::Tensor& result,
                         int64_t dim,
                         int64_t wrapped_dim,
                         const at::Tensor& index,
                         const at::Tensor& source,
                         const at::Scalar& alpha,
                         bool has_self = true) {
        std::cout << "========== " << location << " ==========" << std::endl;
        std::cout << "dim: " << dim << ", wrapped_dim: " << wrapped_dim << std::endl;
        std::cout << "alpha: " << alpha.to<double>() << std::endl;
        
        if (has_self) {
            save_tensor_to_file(self, "self", "index_add_debug");
        }
        save_tensor_to_file(result, "result_before", "index_add_debug");
        save_tensor_to_file(index, "index", "index_add_debug");
        save_tensor_to_file(source, "source", "index_add_debug");
        
        // 计算并打印 source * alpha
        auto source_alpha = source * alpha;
        save_tensor_to_file(source_alpha, "source_alpha", "index_add_debug");
        
        std::cout << "=========================================" << std::endl;
    }
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
        
        // 打印调试信息并保存数据
        print_debug_info("index_add_out deterministic", self, result, dim, wrapped_dim, index, source, alpha, true);
        
        result.index_put_(indices, source * alpha, true);
        
        // 打印结果
        std::cout << "========== " << "index_add_out deterministic (After)" << " ==========" << std::endl;
        save_tensor_to_file(result, "result_after", "index_add_debug");
        std::cout << "=========================================" << std::endl;
        
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
        
        // 打印调试信息并保存数据
        print_debug_info("index_add deterministic", self, result, dim, wrapped_dim, index, source, alpha, true);
        
        result.index_put_(indices, source * alpha, true);
        
        // 打印结果
        std::cout << "========== " << "index_add deterministic (After)" << " ==========" << std::endl;
        save_tensor_to_file(result, "result_after", "index_add_debug");
        std::cout << "=========================================" << std::endl;
        
        return result;
    }

    EXEC_NPU_CMD(aclnnIndexAdd, result.copy_(self), dim, index, source, alpha, result);
    return result;
}

#if !VERSION_BETWEEN(V2R13, VERSION_NEWEST)
at::Tensor index_add(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
    return op_api::index_add(self, dimname_to_position(self, dim), index, source, alpha);
}
#endif

} // namespace op_api