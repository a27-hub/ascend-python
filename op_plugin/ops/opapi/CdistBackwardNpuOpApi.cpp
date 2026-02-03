#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_cdist_backward(const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2,const at::Tensor & cdist,double p)
{
    float p_cast = static_cast<float>(p);
    auto output_size_0 = x1.sizes();
    auto output_dtype_0 = grad.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                            grad.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCdistBackward, grad, x1, x2, cdist, p_cast, out);
    return out;
}
}