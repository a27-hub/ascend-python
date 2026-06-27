// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include <cstdio>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/AdvancedIndex.h"
#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
void check_no_overlap(const at::Tensor& a, const at::Tensor& b)
{
    const auto overlap_status = at_npu::native::get_overlap_status(a, b);
    TORCH_CHECK(overlap_status != at_npu::native::MemOverlapStatus::PARTIAL &&
                    overlap_status != at_npu::native::MemOverlapStatus::FULL,
                "unsupported operation: some elements of the input tensor and "
                "the written-to tensor refer to a single memory location. "
                "Please clone() the tensor before performing the operation.",
                OPS_ERROR(ErrCode::NOT_SUPPORT));
}

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

const char* dtype_str(at::ScalarType t) {
    switch (t) {
        case at::kFloat:   return "Float";
        case at::kDouble:  return "Double";
        case at::kHalf:    return "Half";
        case at::kBFloat16:return "BFloat16";
        case at::kInt:     return "Int";
        case at::kLong:    return "Long";
        case at::kBool:    return "Bool";
        case at::kByte:    return "Byte";
        default:           return "Other";
    }
}
} // namespace

at::Tensor index_put(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    bool accumulate) {
    DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::index_put(self, indices, value, accumulate));
    return self.clone(at::MemoryFormat::Contiguous).index_put_(indices, value, accumulate);
}

at::Tensor& index_put_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate) {
    printf("[DEBUG] ========== index_put_ entry ==========\n");
    printf("[DEBUG] self: numel=%lld dim=%lld dtype=%s\n",
           (long long)self.numel(), (long long)self.dim(), dtype_str(self.scalar_type()));
    print_sizes("[DEBUG] self.sizes=", self.sizes());
    print_strides(" strides=", self.strides());
    printf("\n");
    printf("[DEBUG] value: numel=%lld dim=%lld dtype=%s\n",
           (long long)value.numel(), (long long)value.dim(), dtype_str(value.scalar_type()));
    print_sizes("[DEBUG] value.sizes=", value.sizes());
    print_strides(" strides=", value.strides());
    printf("\n");
    printf("[DEBUG] accumulate=%d\n", (int)accumulate);
    printf("[DEBUG] indices list size=%zu\n", indices.size());
    for (const auto i : c10::irange(indices.size())) {
        if (indices[i].has_value() && indices[i]->defined()) {
            printf("[DEBUG]   indices[%lld]: defined dtype=%s ",
                   (long long)i, dtype_str(indices[i]->scalar_type()));
            print_sizes("sizes=", indices[i]->sizes());
            printf("\n");
        } else {
            printf("[DEBUG]   indices[%lld]: nullopt/undefined\n", (long long)i);
        }
    }
    DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::index_put_(self, indices, value, accumulate));
    printf("[DEBUG] calling _index_put_impl_ ...\n");
    return at::_index_put_impl_(self, indices, value, accumulate, false);
}

at::Tensor& _index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe) {
    DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::_index_put_impl_(self, indices, value, accumulate, unsafe));
    printf("[DEBUG] ========== _index_put_impl_ entry ==========\n");
    printf("[DEBUG] self: numel=%lld dim=%lld dtype=%s\n",
           (long long)self.numel(), (long long)self.dim(), dtype_str(self.scalar_type()));
    print_sizes("[DEBUG] self.sizes=", self.sizes());
    print_strides(" strides=", self.strides());
    printf("\n");
    printf("[DEBUG] value: numel=%lld dim=%lld dtype=%s\n",
           (long long)value.numel(), (long long)value.dim(), dtype_str(value.scalar_type()));
    print_sizes("[DEBUG] value.sizes=", value.sizes());
    print_strides(" strides=", value.strides());
    printf("\n");
    printf("[DEBUG] accumulate=%d unsafe=%d deterministic=%d\n",
           (int)accumulate, (int)unsafe, (int)at::globalContext().deterministicAlgorithms());
    printf("[DEBUG] indices list size=%zu\n", indices.size());
    for (const auto i : c10::irange(indices.size())) {
        if (indices[i].has_value() && indices[i]->defined()) {
            printf("[DEBUG]   indices[%lld]: defined dtype=%s ",
                   (long long)i, dtype_str(indices[i]->scalar_type()));
            print_sizes("sizes=", indices[i]->sizes());
            printf("\n");
        } else {
            printf("[DEBUG]   indices[%lld]: nullopt/undefined\n", (long long)i);
        }
    }
    if (self.device().type() == at::kCPU) {
        printf("[DEBUG] CPU device -> at::native::_index_put_impl_\n");
        return at::native::_index_put_impl_(self, indices, value, accumulate, unsafe);
    }
    if (at_npu::native::has_internal_overlap(self) == at_npu::native::MemOverlap::YES) {
        TORCH_WARN(
            "Use of index_put_ on expanded tensors is deprecated. "
            "Please clone() the tensor before performing this operation. "
            "This also applies to advanced indexing e.g. tensor[indices] = tensor");
    }
    check_no_overlap(self, value);
    for (const c10::optional<at::Tensor>& index : indices) {
        if (index.has_value()) {
            check_no_overlap(self, *index);
        }
    }

    bool needCast = op_plugin::AdvanceIndex::checkIndexTensorTypes(indices);
    auto indices_after = op_plugin::AdvanceIndex::npu_expand_tensors(self, indices, needCast, true);
    std::vector<at::Tensor> all_defined_indices;
    at::SmallVector<int64_t, op_infer::N> zeroSize = {0};
    at::Tensor emptyTensor = npu_preparation::apply_tensor_without_format(self, zeroSize);
    c10::List<c10::optional<at::Tensor>> indices_expand_list;
    for (at::Tensor index_opt : indices_after) {
        indices_expand_list.push_back(index_opt);
    }
    auto info = op_plugin::AdvanceIndex::make_info(self, indices_expand_list);
    TORCH_CHECK(op_plugin::AdvanceIndex::is_expandable_to(value.sizes(), info.src.sizes()),
        "shape mismatch: value tensor of shape ", value.sizes(),
        " cannot be broadcast to indexing result of shape ", info.src.sizes(),
        OPS_ERROR(ErrCode::PARAM));
    for (int i = 0; i < static_cast<int>(indices_after.size()); i++) {
        if (indices_after[i].defined()) {
            all_defined_indices.emplace_back(indices_after[i]);
        continue;
        }
    all_defined_indices.emplace_back(emptyTensor);
    }

  for (auto &all_defined_indice : all_defined_indices) {
      if (all_defined_indice.device() != self.device()) {
          all_defined_indice = all_defined_indice.to(self.device());
      }
  }

  printf("[DEBUG] after processing:\n");
  printf("[DEBUG]   all_defined_indices count=%zu\n", all_defined_indices.size());
  for (size_t i = 0; i < all_defined_indices.size(); i++) {
      printf("[DEBUG]     [%zu]: dtype=%s ", i, dtype_str(all_defined_indices[i].scalar_type()));
      print_sizes("sizes=", all_defined_indices[i].sizes());
      printf("\n");
  }
  printf("[DEBUG]   indices_after count=%zu\n", indices_after.size());
  for (size_t i = 0; i < indices_after.size(); i++) {
      if (indices_after[i].defined()) {
          printf("[DEBUG]     [%zu]: defined ", i);
          print_sizes("sizes=", indices_after[i].sizes());
          printf("\n");
      } else {
          printf("[DEBUG]     [%zu]: undefined\n", i);
      }
  }
  printf("[DEBUG]   info.src.sizes: "); print_sizes("", info.src.sizes()); printf("\n");
  printf("[DEBUG]   info.indexed_sizes: "); print_sizes("", info.indexed_sizes); printf("\n");
  printf("[DEBUG]   info.indexed_strides: "); print_sizes("", info.indexed_strides); printf("\n");
  printf("[DEBUG]   info.dims_before=%lld dims_after=%lld\n",
         (long long)info.dims_before, (long long)info.dims_after);

//   // 这个if 判断 有啥用
//   if (accumulate && at::globalContext().deterministicAlgorithms()) {
//       return acl_op::_index_put_impl_(self, indices, value, accumulate, unsafe);
//   }
//   //

  at::TensorList indices_tensor_list = all_defined_indices;
  if (self.numel() != 0 && value.numel() != 0) {
      printf("[DEBUG] calling EXEC_NPU_CMD(aclnnIndexPutImpl) ...\n");
      printf("[DEBUG]   indices_tensor_list size=%zu\n", indices_tensor_list.size());
      EXEC_NPU_CMD(aclnnIndexPutImpl, self, indices_tensor_list, value, accumulate, unsafe);
      printf("[DEBUG] EXEC_NPU_CMD(aclnnIndexPutImpl) returned\n");
  } else {
      printf("[DEBUG] skip EXEC_NPU_CMD: self.numel=%lld value.numel=%lld\n",
             (long long)self.numel(), (long long)value.numel());
  }
  printf("[DEBUG] _index_put_impl_ returning. self numel=%lld\n", (long long)self.numel());
  printf("[DEBUG] ========== _index_put_impl_ exit ==========\n");
  return self;
}

} // namespace op_api
