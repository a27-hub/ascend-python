#include <ATen/native/TypeProperties.h>
#include <ATen/MemoryOverlap.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using namespace op_plugin::utils;
using namespace op_infer;

namespace {
inline void linalg_cross_check(const at::Tensor &self, const at::Tensor &other, int64_t dim)
{
    auto x_d = self.dim();
    auto y_d = other.dim();
    TORCH_CHECK(x_d == y_d, "linalg.cross: inputs must have the same number of dimensions.",
                OPS_ERROR(ErrCode::PARAM));
    auto wrap_dim = at::maybe_wrap_dim(dim, x_d);
    TORCH_CHECK(self.size(wrap_dim) == 3 && other.size(wrap_dim) == 3,
                "linalg.cross: inputs dimension ", wrap_dim, " must have length 3. Got ", self.size(wrap_dim),
                " and ", other.size(wrap_dim), OPS_ERROR(ErrCode::PARAM));
}
} // namespace

at::Tensor linalg_cross(const at::Tensor & self, const at::Tensor & other, int64_t dim)
{
    DO_COMPATIBILITY(aclnnLinalgCross, acl_op::linalg_cross(self, other, dim));
    linalg_cross_check(self, other, dim);
    auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                  self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLinalgCross, self, other, wrap_dim, out);
    return out;
}

at::Tensor & linalg_cross_out(const at::Tensor & self, const at::Tensor & other, int64_t dim, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLinalgCross, acl_op::linalg_cross_out(self, other, dim, out));
    linalg_cross_check(self, other, dim);
    auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, self);
    at::assert_no_overlap(out, other);
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self, other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLinalgCross, self, other, wrap_dim, out);
    return out;
}
} // namespace op_api
