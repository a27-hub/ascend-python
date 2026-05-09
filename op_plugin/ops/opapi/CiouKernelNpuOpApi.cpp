#include "torch_npu/csrc/aten/CustomFunctions.h"

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
c10::SmallVector<int64_t, op_infer::SIZE> ciou_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool is_cross)
{
    TORCH_CHECK(self.dim() == 2, "ciou expected input in 2D, "
        "but input self has sizes ", self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gtboxes.dim() == 2, "ciou expected input in 2D, "
        "but input gtboxes has sizes ", gtboxes.dim(),
        OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, op_infer::SIZE> output_size;
    if (is_cross) {
        output_size = {gtboxes.size(1), self.size(1)};
    } else {
        output_size = {1, self.size(1)};
    }
    return output_size;
}
} // namespace

std::tuple<at::Tensor, at::Tensor> _npu_ciou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag)
{
    DO_COMPATIBILITY(aclnnCIoU, acl_op::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag));

    bool self_is_half = self.scalar_type() == at::kHalf;
    bool gtboxes_is_half = gtboxes.scalar_type() == at::kHalf;
    at::Tensor self_cp = self_is_half ? at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat) : self;
    at::Tensor gtboxes_cp = gtboxes_is_half ? at_npu::native::custom_ops::npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;

    auto output_size = ciou_npu_output_size(self_cp, gtboxes_cp, is_cross);
    at::Tensor overlap = npu_preparation::apply_tensor_with_format(output_size, self_cp.options(), ACL_FORMAT_ND);
    at::Tensor atan_sub = npu_preparation::apply_tensor_with_format(output_size, self_cp.options(), ACL_FORMAT_ND);

    string mode_str = mode == 1 ? "iof" : "iou";
    EXEC_NPU_CMD(aclnnCIoU, self_cp, gtboxes_cp, trans, is_cross, mode_str, overlap, atan_sub);

    if (self_is_half || gtboxes_is_half) {
        overlap = at_npu::native::custom_ops::npu_dtype_cast(overlap, at::kHalf);
        atan_sub = at_npu::native::custom_ops::npu_dtype_cast(atan_sub, at::kHalf);
    }
    return std::tie(overlap, atan_sub);
}

at::Tensor npu_ciou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag)
{
    DO_COMPATIBILITY(aclnnCIoU, acl_op::npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag));

    auto results = at_npu::native::custom_ops::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
    return std::get<0>(results);
}

std::tuple<at::Tensor, at::Tensor> npu_ciou_backward(
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    const c10::optional<at::Tensor>& atan_sub_opt,
    bool trans,
    bool is_cross,
    int64_t mode)
{
    DO_COMPATIBILITY(aclnnCIoUGrad,
        acl_op::npu_ciou_backward(grad, bboxes, gtboxes, atan_sub_opt, trans, is_cross, mode));

    const at::Tensor& atan_sub = c10::value_or_else(atan_sub_opt, [] { return at::Tensor(); });
    at::Tensor grad_cp = at::squeeze(grad, 0);
    if (grad_cp.scalar_type() == at::kHalf) {
        grad_cp = at_npu::native::custom_ops::npu_dtype_cast(grad_cp, at::kFloat);
    }
    bool bboxes_is_half = bboxes.scalar_type() == at::kHalf;
    bool gtboxes_is_half = gtboxes.scalar_type() == at::kHalf;
    at::Tensor bboxes_cp = bboxes_is_half ? at_npu::native::custom_ops::npu_dtype_cast(bboxes, at::kFloat) : bboxes;
    at::Tensor gtboxes_cp = gtboxes_is_half ? at_npu::native::custom_ops::npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;

    at::Tensor dbboxes = npu_preparation::apply_tensor_with_format(bboxes_cp.sizes(), bboxes_cp.options(), ACL_FORMAT_ND);
    at::Tensor dgtboxes = npu_preparation::apply_tensor_with_format(gtboxes_cp.sizes(), gtboxes_cp.options(), ACL_FORMAT_ND);

    string mode_str = mode == 1 ? "iof" : "iou";
    EXEC_NPU_CMD(aclnnCIoUGrad, grad_cp, bboxes_cp, gtboxes_cp, atan_sub, trans, is_cross, mode_str, dbboxes, dgtboxes);

    if (bboxes_is_half || gtboxes_is_half) {
        dbboxes = at_npu::native::custom_ops::npu_dtype_cast(dbboxes, at::kHalf);
        dgtboxes = at_npu::native::custom_ops::npu_dtype_cast(dgtboxes, at::kHalf);
    }
    return std::tie(dbboxes, dgtboxes);
}
} // namespace op_api
