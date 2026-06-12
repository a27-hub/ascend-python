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
    DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::index_put_(self, indices, value, accumulate));
    return at::_index_put_impl_(self, indices, value, accumulate, false);
}

at::Tensor& _index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe) {
    DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::_index_put_impl_(self, indices, value, accumulate, unsafe));
    if (self.device().type() == at::kCPU) {
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
    c10::List<c10::optional<at::Tensor>> indices_expand_list;
    for (at::Tensor index_opt : indices_after) {
        indices_expand_list.push_back(index_opt);
    }
    auto info = op_plugin::AdvanceIndex::make_info(self, indices_expand_list);
    TORCH_CHECK(op_plugin::AdvanceIndex::is_expandable_to(value.sizes(), info.src.sizes()),
        "shape mismatch: value tensor of shape ", value.sizes(),
        " cannot be broadcast to indexing result of shape ", info.src.sizes(),
        OPS_ERROR(ErrCode::PARAM));

    // aclnnIndexPutImpl crashes (null deref in its internal Transpose) if any
    // undefined index is forwarded as an empty placeholder tensor. It also
    // requires exactly self.dim() index tensors – passing fewer (e.g. after
    // transposeToFront) breaks value broadcasting because the kernel aligns
    // value with the trailing result dimensions instead of the index dimensions.
    //
    // Instead: keep self in its original layout and craft explicit "select all"
    // sentinel indices for every None dimension. The sentinels and the defined
    // indices are reshaped so that they broadcast to the Cartesian-product grid
    // [index_bcast_shape, none_dim_0_size, none_dim_1_size, …].  This ensures
    // aclnnIndexPutImpl always sees self.dim() index tensors and the value
    // broadcasts correctly (it aligns from the right with the index dimension
    // that sits last in the original result layout).
    std::vector<at::Tensor> indices_padded = indices_after;
    while (indices_padded.size() < static_cast<size_t>(self.dim())) {
        indices_padded.emplace_back();
    }

    // Separate defined / None positions.
    std::vector<at::Tensor> defined_indices;
    std::vector<int64_t> defined_positions;
    std::vector<int64_t> none_positions;
    for (int64_t i = 0; i < self.dim(); i++) {
        if (indices_padded[i].defined()) {
            defined_indices.push_back(indices_padded[i]);
            defined_positions.push_back(i);
        } else {
            none_positions.push_back(i);
        }
    }
    int64_t n_defined = static_cast<int64_t>(defined_indices.size());
    int64_t n_none = self.dim() - n_defined;

    // Fast path: no None dims — use indices directly.
    if (n_none == 0) {
        for (auto& idx : defined_indices) {
            if (idx.device() != self.device()) {
                idx = idx.to(self.device());
            }
        }
        at::TensorList indices_tensor_list = defined_indices;
        if (self.numel() != 0 && value.numel() != 0) {
            EXEC_NPU_CMD(aclnnIndexPutImpl, self, indices_tensor_list, value, accumulate, unsafe);
        }
        return self;
    }

    // Broadcast all defined indices to a common shape.
    std::vector<at::Tensor> defined_bcast = defined_indices;
    std::vector<int64_t> bcast_shape_vec;  // owning copy – safe across reshape
    {
        std::vector<int64_t> bcast_shape;
        for (size_t i = 0; i < defined_bcast.size(); i++) {
            if (bcast_shape.empty()) {
                bcast_shape = defined_bcast[i].sizes().vec();
            } else {
                bcast_shape = at::infer_size(bcast_shape, defined_bcast[i].sizes());
            }
        }
        for (size_t i = 0; i < defined_bcast.size(); i++) {
            if (!defined_bcast[i].sizes().equals(bcast_shape)) {
                defined_bcast[i] = defined_bcast[i].expand(bcast_shape, /*implicit=*/true);
            }
        }
        bcast_shape_vec = std::move(bcast_shape);
    }

    // Reshape each defined index to [bcast_shape, 1, …, 1]  (n_none trailing 1s).
    for (auto& idx : defined_bcast) {
        std::vector<int64_t> new_shape(bcast_shape_vec.begin(), bcast_shape_vec.end());
        new_shape.insert(new_shape.end(), n_none, 1);
        idx = idx.reshape(new_shape);
    }

    // Create sentinel indices for None dims.
    // sentinel_j shape: [1]*|bcast_shape|  +  [dim_size at position j]  +  [1]*(n_none - j - 1)
    std::vector<at::Tensor> sentinel_indices;
    sentinel_indices.reserve(n_none);
    for (int64_t j = 0; j < n_none; j++) {
        int64_t none_dim = none_positions[j];
        int64_t dim_size = self.size(none_dim);

        std::vector<int64_t> sentinel_shape(bcast_shape_vec.size(), 1);
        for (int64_t k = 0; k < n_none; k++) {
            sentinel_shape.push_back(k == j ? dim_size : 1);
        }

        at::Tensor sentinel = at::arange(
            dim_size, at::TensorOptions().dtype(at::kLong).device(self.device()));
        sentinel = sentinel.reshape(sentinel_shape);
        sentinel_indices.push_back(sentinel);
    }

    // Assemble final index list in the original dimension order.
    std::vector<at::Tensor> final_indices(self.dim());
    int defined_idx = 0, none_idx = 0;
    for (int64_t i = 0; i < self.dim(); i++) {
        if (indices_padded[i].defined()) {
            final_indices[i] = defined_bcast[defined_idx++];
        } else {
            final_indices[i] = sentinel_indices[none_idx++];
        }
    }

    // Ensure all indices are on the correct device.
    for (auto& idx : final_indices) {
        if (idx.device() != self.device()) {
            idx = idx.to(self.device());
        }
    }

    at::TensorList indices_tensor_list = final_indices;
    if (self.numel() != 0 && value.numel() != 0) {
        EXEC_NPU_CMD(aclnnIndexPutImpl, self, indices_tensor_list, value, accumulate, unsafe);
    }
    return self;
}

}
