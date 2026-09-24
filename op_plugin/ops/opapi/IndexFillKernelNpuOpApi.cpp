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

namespace op_api {

constexpr int MAX_DIM = 1;

// aclnnInplaceIndexFill / aclnnIndexFill only support a limited dtype list. For dtypes
// outside that list (e.g. int8/int16/uint8/float64), rewrite index_fill as the equivalent
// index_put_ with a single advanced index at `dim` (all other dims sliced with `:`).
// index_put_ is dispatched to aclnnIndexPutImpl which has a wider dtype coverage and is
// available on all platforms (incl. Ascend 950) without depending on aclop.
static bool is_supported_by_aclnn_index_fill(at::ScalarType dtype)
{
    return dtype == at::kInt || dtype == at::kHalf || dtype == at::kFloat ||
           dtype == at::kLong || dtype == at::kBool || dtype == at::kBFloat16;
}

static c10::List<c10::optional<at::Tensor>> make_index_put_indices(int64_t self_dim, int64_t dim,
                                                                   const at::Tensor& index)
{
    c10::List<c10::optional<at::Tensor>> indices;
    for (int64_t d = 0; d < self_dim; ++d) {
        if (d == dim) {
            indices.push_back(index);
        } else {
            indices.push_back(c10::optional<at::Tensor>());
        }
    }
    return indices;
}

static at::Tensor& index_fill_fallback_impl_(at::Tensor& self, int64_t dim,
                                             const at::Tensor& index, const at::Scalar& value,
                                             at::ScalarType dtype)
{
    at::Tensor value_tensor = at::scalar_tensor(value, self.options().dtype(dtype));
    return self.index_put_(make_index_put_indices(self.dim(), dim, index), value_tensor, false);
}

static std::vector<int64_t> get_index_vector(const at::Tensor& index)
{
    std::vector<int64_t> idx_vec;
    if (index.dim() == 0) {
        idx_vec.emplace_back(static_cast<int64_t>(index.item().to<int>()));
    } else {
        for (int64_t i = 0; i < index.sizes()[0]; i++) {
            int64_t idx = static_cast<int64_t>(index[i].item().to<int>());
            idx_vec.emplace_back(idx);
        }
    }
    return idx_vec;
}

at::Tensor& index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                        const at::Tensor& value)
{
    TORCH_CHECK(value.dim() == 0, "Value should be a 0-dimensional tensor, but got ",
                value.dim(), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(self.is_complex() || !value.is_complex(),
                "index_fill_(): Converting complex Scalar to non-complex type is not supported",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
    if (!is_supported_by_aclnn_index_fill(self.scalar_type())) {
        index_fill_fallback_impl_(self, dim, index, value.item(), self.scalar_type());
        return self;
    }
    static const bool is_inplace_index_fill_available = check_aclnn_kernel_available("aclnnInplaceIndexFill");
    if (!is_inplace_index_fill_available ||
        (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950 && !op_plugin::utils::is_gte_cann_version_910())) {
        DO_COMPATIBILITY(aclnnInplaceIndexFillTensor, acl_op::index_fill_(self, dim, index, value));
        TORCH_CHECK(index.dim() <= MAX_DIM, "Index has to be a vector/scalar.", OPS_ERROR(ErrCode::TYPE));
        at::Scalar value_scalar = value.item();
        std::vector<int64_t> idx_vec = get_index_vector(index);
        at::IntArrayRef index_array = at::IntArrayRef(idx_vec);
        EXEC_NPU_CMD(aclnnInplaceIndexFillTensor, self, dim, index_array, value_scalar);
    } else {
        DO_COMPATIBILITY(aclnnInplaceIndexFill, acl_op::index_fill_(self, dim, index, value));
        at::Scalar value_scalar = value.item();
        EXEC_NPU_CMD(aclnnInplaceIndexFill, self, dim, index, value_scalar);
    }
    return self;
}

at::Tensor index_fill(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                      const at::Tensor& value)
{
    TORCH_CHECK(value.dim() == 0, "Value should be a 0-dimensional tensor, but got ",
                value.dim(), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(self.is_complex() || !value.is_complex(),
                "index_fill(): Converting complex Scalar to non-complex type is not supported",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
    if (!is_supported_by_aclnn_index_fill(self.scalar_type())) {
        at::Tensor result = self.clone(at::MemoryFormat::Contiguous);
        index_fill_fallback_impl_(result, dim, index, value.item(), self.scalar_type());
        at::namedinference::propagate_names(result, self);
        return result;
    }
    static const bool is_index_fill_available = check_aclnn_kernel_available("aclnnIndexFill");
    if (!is_index_fill_available || c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        DO_COMPATIBILITY(aclnnIndexFillTensor, acl_op::index_fill(self, dim, index, value));
        TORCH_CHECK(index.dim() <= MAX_DIM, "Index has to be a vector/scalar.", OPS_ERROR(ErrCode::TYPE));
        at::Scalar value_scalar = value.item();
        std::vector<int64_t> idx_vec = get_index_vector(index);
        at::IntArrayRef index_array = at::IntArrayRef(idx_vec);
        at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
        EXEC_NPU_CMD(aclnnIndexFillTensor, self, dim, index_array, value_scalar, result);
        at::namedinference::propagate_names(result, self);
        return result;
    } else {
        DO_COMPATIBILITY(aclnnIndexFill, acl_op::index_fill(self, dim, index, value));
        at::Scalar value_scalar = value.item();
        at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
        EXEC_NPU_CMD(aclnnIndexFill, self, dim, index, value_scalar, result);
        at::namedinference::propagate_names(result, self);
        return result;
    }
}

at::Tensor& index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                        const at::Scalar& value)
{
    TORCH_CHECK(self.is_complex() || !value.isComplex(),
                "index_fill_(): Converting complex Scalar to non-complex type is not supported",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
    if (!is_supported_by_aclnn_index_fill(self.scalar_type())) {
        index_fill_fallback_impl_(self, dim, index, value, self.scalar_type());
        return self;
    }
    static const bool is_inplace_index_fill_available = check_aclnn_kernel_available("aclnnInplaceIndexFill");
    if (!is_inplace_index_fill_available ||
        (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950 && !op_plugin::utils::is_gte_cann_version_910())) {
        DO_COMPATIBILITY(aclnnInplaceIndexFillTensor, acl_op::index_fill_(self, dim, index, value));
        TORCH_CHECK(index.dim() <= MAX_DIM, "Index has to be a vector/scalar.", OPS_ERROR(ErrCode::TYPE));
        std::vector<int64_t> idx_vec = get_index_vector(index);
        at::IntArrayRef index_array = at::IntArrayRef(idx_vec);
        EXEC_NPU_CMD(aclnnInplaceIndexFillTensor, self, dim, index_array, value);
    } else {
        DO_COMPATIBILITY(aclnnInplaceIndexFill, acl_op::index_fill_(self, dim, index, value));
        EXEC_NPU_CMD(aclnnInplaceIndexFill, self, dim, index, value);
    }
    return self;
}

at::Tensor index_fill(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                      const at::Scalar& value)
{
    TORCH_CHECK(self.is_complex() || !value.isComplex(),
                "index_fill(): Converting complex Scalar to non-complex type is not supported",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
    if (!is_supported_by_aclnn_index_fill(self.scalar_type())) {
        at::Tensor result = self.clone(at::MemoryFormat::Contiguous);
        index_fill_fallback_impl_(result, dim, index, value, self.scalar_type());
        at::namedinference::propagate_names(result, self);
        return result;
    }
    static const bool is_index_fill_available = check_aclnn_kernel_available("aclnnIndexFill");
    if (!is_index_fill_available || c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        DO_COMPATIBILITY(aclnnIndexFillTensor, acl_op::index_fill(self, dim, index, value));
        TORCH_CHECK(index.dim() <= MAX_DIM, "Index has to be a vector/scalar.", OPS_ERROR(ErrCode::TYPE));
        std::vector<int64_t> idx_vec = get_index_vector(index);
        at::IntArrayRef index_array = at::IntArrayRef(idx_vec);
        // construct output tensor
        at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
        EXEC_NPU_CMD(aclnnIndexFillTensor, self, dim, index_array, value, result);
        at::namedinference::propagate_names(result, self);
        return result;
    } else {
        DO_COMPATIBILITY(aclnnIndexFill, acl_op::index_fill(self, dim, index, value));
        at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
        EXEC_NPU_CMD(aclnnIndexFill, self, dim, index, value, result);
        at::namedinference::propagate_names(result, self);
        return result;
    }
}
}