#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> npu_split_with_size(
    const at::Tensor& x,
    c10::ArrayRef<int64_t> split_sizes,
    int64_t dim
)
{
    const size_t split_size = split_sizes.size();
    std::vector<at::Tensor> result;
    result.reserve(split_size);

    const auto base_size_vec = op_infer::input_same_output_size(x).vec();
    const auto dim_size = base_size_vec.size();
    if (dim < 0) {
        dim += dim_size;
    }
    for (size_t i = 0; i < split_size; i++) {
        auto new_size = base_size_vec;
        new_size[dim] = split_sizes[i];
        result.emplace_back(npu_preparation::apply_tensor_without_format(new_size, x.options()));
    }
    at::TensorList out = at::TensorList(result);

    EXEC_NPU_CMD(aclnnSplitWithSize, x, split_sizes, dim, out);
    return result;
}
}