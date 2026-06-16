#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    using namespace op_plugin::utils;
    using namespace op_infer;
}

at::Tensor top_k_top_p_softmax(const at::Tensor &logits, const c10::optional<at::Tensor> &p,
                               const c10::optional<at::Tensor> &k)
{
    auto output_size_0 = logits.sizes();
    auto output_dtype_0 = logits.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0, logits.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnApplyTopKTopP, logits, p, k, out);
    return at::_softmax(out, -1, false);
}  // namespace op_api