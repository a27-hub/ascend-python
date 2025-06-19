#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> fused_add_topk_div(
    const at::Tensor& x,
    const at::Tensor& add_num,
    const at::Tensor& mapping_num,
    const at::Tensor& mapping_table,
    int64_t group_num,
    int64_t group_topk,
    int64_t n,
    int64_t k,
    int64_t activate_type,
    bool is_norm,
    bool enable_expert_mapping,
    double scale
    )
{
    int64_t  a = x.sizes().data()[0];
    at::Tensor y = at::empty({a, k}, at::TensorOptions().dtype(at::kFloat).device(x.options().device()));
    at::Tensor indices = at::empty({a, k}, at::TensorOptions().dtype(at::kInt).device(x.options().device()));

    EXEC_NPU_CMD(aclnnFusedAddTopkDiv, x, add_num, mapping_num, mapping_table, group_num, group_topk, n, k, activate_type, is_norm, enable_expert_mapping, scale, y, indices);
    return std::make_tuple(y, indices);
}
}