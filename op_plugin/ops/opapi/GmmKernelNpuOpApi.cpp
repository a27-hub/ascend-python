#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace {
    void _check_dims(size_t num_x, const at::TensorList &weight, size_t num_group_list)
    {
        size_t num_w = weight.size();
        TORCH_CHECK(num_x > 0 && num_w > 0,
            "Neither x nor weight could be empty.");
        size_t dim_num_w = weight[0].sizes().size();
        size_t dim_0_w = weight[0].sizes()[0];
    }

    void _create_new_tensor(std::vector<at::Tensor> &y, int64_t dim_m, int64_t dim_n, c10::TensorOptions options,
                            int64_t group_type_value, int64_t num_group_list)
    {
        auto output_size = (2 == group_type_value) ? op_infer::array_to_small_vector({num_group_list, dim_m, dim_n})
                                                : op_infer::array_to_small_vector({dim_m, dim_n});
        y.emplace_back(at::empty(output_size, options));
    }
}

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

bool _is_transposed(at::Tensor &tensors)
{
    int dim_sum = tensors.dim();
    TORCH_CHECK(dim_sum >= 2 && dim_sum <= 3, // 2/3: gmm weight only support 2- or 3-dimensional
        "input tensor of is_tensor_transposed should be either 2- or 3-dimensional.");
    int shape_dim = tensors.sizes().size() - 2;
    if (tensors.stride(dim_sum - 2) == 1 && tensors.stride(dim_sum - 1) == tensors.sizes().at(shape_dim)) {
        return true;
    } else {
        return false;
    }
}

#if VERSION_BETWEEN(v2R1, VERSION_NEWEST)
std::vector<at::Tensor> npu_gmm(const at::TensorList x,
                                const at::TensorList weight,
                                const c10::optional<at::TensorList> bias,
                                c10::OptionalIntArrayRef group_list,
                                c10::optional<int64_t> group_type,
                                c10::optional<int64_t> group_list_type)
{
    auto num_x = x.size();
    auto num_w = weight.size();
    auto group_list_real = group_list.value_or(at::IntArrayRef(std::vector<int64_t>{}));
    auto num_group_list = group_list_real.size();
    int64_t split_item_value = 3;
    int64_t group_type_value = group_type.value_or(-1);

    _check_dims(num_x, weight, num_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(x[0].scalar_type());

    size_t dim_num_w = weight[0].sizes().size();
    _create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[dim_num_w - 1], options, group_type_value,
                        num_group_list);

    at::TensorList result = at::TensorList(y);
    auto scale_real = at::TensorList();
    auto offset_real = at::TensorList();
    auto antiquant_scale_real = at::TensorList();
    auto antiquant_offset_real = at::TensorList();
    auto bias_real = bias.value_or(at::TensorList());
    EXEC_NPU_CMD(aclnnGroupedMatmulV2, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real,
        antiquant_offset_real, group_list_real, split_item_value, group_type_value, result);
    return y;
}

std::vector<at::Tensor> npu_gmm(const at::TensorList x,
                                const at::TensorList weight,
                                const c10::optional<at::TensorList> bias,
                                const c10::optional<at::Tensor>& group_list,
                                c10::optional<int64_t> group_type,
                                c10::optional<int64_t> group_list_type)
{
    auto num_x = x.size();
    auto num_w = weight.size();
    auto group_list_real = group_list.value_or(at::Tensor());
    auto num_group_list = group_list_real.sizes()[0];
    int64_t split_item_value = 3;
    int64_t group_type_value = group_type.value_or(-1);
    int64_t group_list_type_value = group_list_type.value_or(0);
    int64_t act_type_value = 0;

    _check_dims(num_x, weight, num_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(x[0].scalar_type());

    size_t dim_num_w = weight[0].sizes().size();
    _create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[dim_num_w - 1], options, group_type_value,
                        num_group_list);

    at::TensorList result = at::TensorList(y);
    auto scale_real = at::TensorList();
    auto offset_real = at::TensorList();
    auto antiquant_scale_real = at::TensorList();
    auto antiquant_offset_real = at::TensorList();
    auto perToken_scale_real = at::TensorList();
    auto activation_input_real = at::TensorList();
    auto activation_quant_scale_real = at::TensorList();
    auto activation_quant_offset_real = at::TensorList();
    auto activation_feature_out_real = at::TensorList();
    auto dynQuant_scale_out_real = at::TensorList();
    auto bias_real = bias.value_or(at::TensorList());

    EXEC_NPU_CMD(aclnnGroupedMatmulV4, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real,
            antiquant_offset_real, perToken_scale_real, group_list_real, activation_input_real,
            activation_quant_scale_real, activation_quant_offset_real, split_item_value, group_type_value,
            group_list_type_value, act_type_value, result, activation_feature_out_real, dynQuant_scale_out_real);

    return y;
}
#endif

} // namespace op_api