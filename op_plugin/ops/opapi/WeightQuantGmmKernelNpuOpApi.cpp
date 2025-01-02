#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(v2R1, VERSION_NEWEST)
std::vector<at::Tensor> npu_weight_quant_gmm(const at::TensorList x,
                                             const at::TensorList weight,
                                             const at::TensorList antiquant_scale,
                                             const c10::optional<at::TensorList> antiquant_offset,
                                             const c10::optional<at::TensorList> bias,
                                             const c10::optional<at::Tensor>& group_list,
                                             c10::optional<int64_t> group_list_type,
                                             c10::optional<int64_t> act_type)
{
    auto group_list_real = group_list.value_or(at::Tensor());
    int64_t split_item_value = 3;
    int64_t group_type_value = 0;
    int64_t group_list_type_value = group_list_type.value_or(0);
    int64_t act_type_value = act_type.value_or(0);

    auto bias_real = bias.value_or(at::TensorList());
    auto antiquant_offset_real = antiquant_offset.value_or(at::TensorList());

    c10::TensorOptions options = x[0].options().dtype(x[0].scalar_type());

    size_t dim_num_w = weight[0].sizes().size();
    auto output_size = op_infer::array_to_small_vector({x[0].sizes()[0], weight[0].sizes()[dim_num_w - 1]});
    std::vector<at::Tensor> y{at::empty(output_size, options)};
    at::TensorList result = at::TensorList(y);
    auto scale = nullptr;
    auto offset = nullptr;
    auto per_token_scale = nullptr;
    auto act_in = nullptr;
    auto act_quant_scale = nullptr;
    auto act_quant_offset = nullptr;
    auto act_out = nullptr;
    auto dynamic_quant_scale_out = nullptr;
    EXEC_NPU_CMD(aclnnGroupedMatmulV4, x, weight, bias_real, scale, offset, antiquant_scale,
            antiquant_offset_real, per_token_scale, group_list_real, act_in, act_quant_scale, act_quant_offset,
            split_item_value, group_type_value, group_list_type_value, act_type_value,
            result, act_out, dynamic_quant_scale_out);

    return y;
}
#endif

} // namespace op_api