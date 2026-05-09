#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_iou(
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    int64_t mode)
{
    DO_COMPATIBILITY(aclnnIou, acl_op::npu_iou(bboxes, gtboxes, mode));

    const at::Tensor& input1 = (mode == 1) ? gtboxes : bboxes;
    const at::Tensor& input2 = (mode == 1) ? bboxes : gtboxes;
    c10::SmallVector<int64_t, SIZE> output_size = {input1.size(0), input2.size(0)};
    at::Tensor overlap = npu_preparation::apply_tensor_with_format(output_size, bboxes.options(), ACL_FORMAT_ND);

    const char *mode_str = mode == 1 ? "iof" : "iou";
    float eps = 0.01;
    bool aligned = false;
    EXEC_NPU_CMD(aclnnIou, input1, input2, mode_str, eps, aligned, overlap);

    return overlap;
}
} // namespace op_api
