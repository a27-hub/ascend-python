#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;


namespace {
constexpr int SIZE = 8;

c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> small_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        small_shape.emplace_back(shape[i]);
    }
    return small_shape;
}
}

#if VERSION_BETWEEN(v2R1, VERSION_NEWEST)

std::vector<at::Tensor> npu_quant_gmm(const at::TensorList x,
                                      const at::TensorList weight,
                                      const at::TensorList scale,
                                      const c10::optional<at::TensorList> offset,
                                      const c10::optional<at::TensorList> per_token_scale,
                                      const c10::optional<at::TensorList> bias,
                                      const c10::optional<at::Tensor>& group_list,
                                      c10::optional<int64_t> group_list_type,
                                      c10::optional<int64_t> output_dtype,
                                      c10::optional<int64_t> act_type)
{
    TORCH_CHECK(x[0].scalar_type() == at::kChar && weight[0].scalar_type() == at::kChar,
                "Input x and weight dtype must be int8.");
    auto group_list_real = group_list.value_or(at::Tensor());
    int64_t split_item_value = 3;
    int64_t group_type_value = 0;
    int64_t group_list_type_value = group_list_type.value_or(0);
    int64_t act_type_value = act_type.value_or(0);

    auto bias_real = bias.value_or(at::TensorList());
    auto offset_real = offset.value_or(at::TensorList());
    auto per_token_scale_real = per_token_scale.value_or(at::TensorList());

    c10::TensorOptions options;
    int64_t output_dtype_value = output_dtype.value_or(0);
    if (output_dtype_value == 1) {
        options = x[0].options().dtype(at::kBFloat16);
    } else if (output_dtype_value == 0) {
        options = x[0].options().dtype(at::kHalf);
    } else if (output_dtype_value == -1) {
        options = x[0].options().dtype(at::kChar);
    } else {
        AT_ERROR("Value of output_dtype must be one of -1/0/1, but got ", output_dtype_value);
    }

    size_t dim_num_w = weight[0].sizes().size();
    auto output_size = array_to_small_vector({x[0].sizes()[0], weight[0].sizes()[dim_num_w - 1]});
    std::vector<at::Tensor> y{at::empty(output_size, options)};
    at::TensorList result = at::TensorList(y);
    auto antiquant_scale = nullptr;
    auto antiquant_offset = nullptr;
    auto act_in = nullptr;
    auto act_quant_scale = nullptr;
    auto act_quant_offset = nullptr;
    auto act_out = nullptr;
    auto dynamic_quant_scale_out = nullptr;
    EXEC_NPU_CMD(aclnnGroupedMatmulV4, x, weight, bias_real, scale, offset_real, antiquant_scale,
            antiquant_offset, per_token_scale_real, group_list_real, act_in, act_quant_scale, act_quant_offset,
            split_item_value, group_type_value, group_list_type_value, act_type_value,
            result, act_out, dynamic_quant_scale_out);

    return y;
}

#endif

} // namespace op_api