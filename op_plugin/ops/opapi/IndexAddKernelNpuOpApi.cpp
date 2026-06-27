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
#include <cstdio>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

namespace {
void print_sizes(const char* prefix, at::IntArrayRef sizes) {
    printf("%s [", prefix);
    for (int64_t i = 0; i < static_cast<int64_t>(sizes.size()); i++) {
        if (i > 0) printf(", ");
        printf("%lld", (long long)sizes[i]);
    }
    printf("]");
}
void print_strides(const char* prefix, at::IntArrayRef strides) {
    printf("%s [", prefix);
    for (int64_t i = 0; i < static_cast<int64_t>(strides.size()); i++) {
        if (i > 0) printf(", ");
        printf("%lld", (long long)strides[i]);
    }
    printf("]");
}
const char* dtype_to_str(at::ScalarType t) {
    switch (t) {
        case at::kFloat:   return "Float";
        case at::kDouble:  return "Double";
        case at::kHalf:    return "Half";
        case at::kBFloat16:return "BFloat16";
        case at::kInt:     return "Int";
        case at::kLong:    return "Long";
        case at::kBool:    return "Bool";
        case at::kByte:    return "Byte";
        case at::kComplexHalf: return "ComplexHalf";
        default:           return "Other";
    }
}
} // namespace

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
        printf("[DEBUG] ========== index_add_out deterministic ==========\n");
        printf("[DEBUG] self: numel=%lld dim=%lld dtype=%s\n",
               (long long)self.numel(), (long long)self.dim(), dtype_to_str(self.scalar_type()));
        print_sizes("[DEBUG] self.sizes=", self.sizes());
        print_strides(" strides=", self.strides());
        printf("\n");
        printf("[DEBUG] dim=%lld self.dim()=%lld\n", (long long)dim, (long long)self.dim());
        printf("[DEBUG] index: numel=%lld dim=%lld dtype=%s\n",
               (long long)index.numel(), (long long)index.dim(), dtype_to_str(index.scalar_type()));
        print_sizes("[DEBUG] index.sizes=", index.sizes());
        printf("\n");
        printf("[DEBUG] source: numel=%lld dim=%lld dtype=%s\n",
               (long long)source.numel(), (long long)source.dim(), dtype_to_str(source.scalar_type()));
        print_sizes("[DEBUG] source.sizes=", source.sizes());
        printf("\n");
        printf("[DEBUG] alpha type=%d\n", (int)alpha.type());
        printf("[DEBUG] result: numel=%lld dim=%lld dtype=%s\n",
               (long long)result.numel(), (long long)result.dim(), dtype_to_str(result.scalar_type()));
        print_sizes("[DEBUG] result.sizes=", result.sizes());
        printf("\n");
        auto wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
        printf("[DEBUG] wrapped_dim=%lld\n", (long long)wrapped_dim);

        c10::List<c10::optional<at::Tensor>> indices;
        indices.reserve(self.dim());
        for (const auto i : c10::irange(self.dim())) {
            if (i == wrapped_dim) {
                indices.emplace_back(index.to(at::kLong));
            } else {
                indices.emplace_back(c10::nullopt);
            }
        }
        printf("[DEBUG] indices list size=%zu\n", indices.size());
        for (const auto i : c10::irange(indices.size())) {
            if (indices[i].has_value()) {
                printf("[DEBUG]   indices[%lld]: defined dtype=%s ",
                       (long long)i, dtype_to_str(indices[i]->scalar_type()));
                print_sizes("sizes=", indices[i]->sizes());
                printf("\n");
            } else {
                printf("[DEBUG]   indices[%lld]: nullopt\n", (long long)i);
            }
        }
        auto value = source * alpha;
        printf("[DEBUG] value=source*alpha: numel=%lld dim=%lld dtype=%s\n",
               (long long)value.numel(), (long long)value.dim(), dtype_to_str(value.scalar_type()));
        print_sizes("[DEBUG] value.sizes=", value.sizes());
        print_strides(" strides=", value.strides());
        printf("\n");
        printf("[DEBUG] calling result.index_put_(indices, value, accumulate=true) ...\n");
        result.index_put_(indices, value, accumulate=true);
        printf("[DEBUG] index_put_ returned. result numel=%lld\n", (long long)result.numel());
        printf("[DEBUG] ========== index_add_out deterministic END ==========\n");
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
        printf("[DEBUG] ========== index_add deterministic ==========\n");
        printf("[DEBUG] self: numel=%lld dim=%lld dtype=%s\n",
               (long long)self.numel(), (long long)self.dim(), dtype_to_str(self.scalar_type()));
        print_sizes("[DEBUG] self.sizes=", self.sizes());
        print_strides(" strides=", self.strides());
        printf("\n");
        printf("[DEBUG] dim=%lld self.dim()=%lld\n", (long long)dim, (long long)self.dim());
        printf("[DEBUG] index: numel=%lld dim=%lld dtype=%s\n",
               (long long)index.numel(), (long long)index.dim(), dtype_to_str(index.scalar_type()));
        print_sizes("[DEBUG] index.sizes=", index.sizes());
        printf("\n");
        printf("[DEBUG] source: numel=%lld dim=%lld dtype=%s\n",
               (long long)source.numel(), (long long)source.dim(), dtype_to_str(source.scalar_type()));
        print_sizes("[DEBUG] source.sizes=", source.sizes());
        printf("\n");
        printf("[DEBUG] alpha type=%d\n", (int)alpha.type());
        auto wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
        printf("[DEBUG] wrapped_dim=%lld\n", (long long)wrapped_dim);

        result.copy_(self);
        printf("[DEBUG] result after copy_: numel=%lld\n", (long long)result.numel());

        c10::List<c10::optional<at::Tensor>> indices;
        indices.reserve(self.dim());
        for (const auto i : c10::irange(self.dim())) {
            if (i == wrapped_dim) {
                indices.emplace_back(index.to(at::kLong));
            } else {
                indices.emplace_back(c10::nullopt);
            }
        }
        printf("[DEBUG] indices list size=%zu\n", indices.size());
        for (const auto i : c10::irange(indices.size())) {
            if (indices[i].has_value()) {
                printf("[DEBUG]   indices[%lld]: defined dtype=%s ",
                       (long long)i, dtype_to_str(indices[i]->scalar_type()));
                print_sizes("sizes=", indices[i]->sizes());
                printf("\n");
            } else {
                printf("[DEBUG]   indices[%lld]: nullopt\n", (long long)i);
            }
        }
        auto value = source * alpha;
        printf("[DEBUG] value=source*alpha: numel=%lld dim=%lld dtype=%s\n",
               (long long)value.numel(), (long long)value.dim(), dtype_to_str(value.scalar_type()));
        print_sizes("[DEBUG] value.sizes=", value.sizes());
        print_strides(" strides=", value.strides());
        printf("\n");
        printf("[DEBUG] calling result.index_put_(indices, value, accumulate=true) ...\n");
        result.index_put_(indices, value, accumulate=true);
        printf("[DEBUG] index_put_ returned. result numel=%lld\n", (long long)result.numel());
        printf("[DEBUG] ========== index_add deterministic END ==========\n");
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
