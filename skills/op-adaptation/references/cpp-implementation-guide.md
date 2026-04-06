# C++ Implementation Guide

## Critical Requirements

**Function signature must match generated interface in `build/pytorch/third_party/op-plugin/op_plugin/OpApiInterface.h`**

Type mapping rules:
- `int[]?` → `c10::OptionalIntArrayRef` (NOT `c10::optional<at::IntArrayRef>`)
- `str` → `c10::string_view` (NOT `std::string`)
- `ScalarType?` → `c10::optional<at::ScalarType>`
- `Tensor?` → `const c10::optional<at::Tensor>&`

## Standard Headers

```cpp
#include "op_plugin/AclOpsInterface.h"      // For DO_COMPATIBILITY fallback
#include "op_plugin/OpApiInterface.h"        // aclnn operator interface
#include "op_plugin/utils/op_api_common.h"   // Macros and utilities
```

## Find Similar Operators

Before implementing, search for similar operators:

**By category:**
- Quantization: `*Quant*.cpp`
- Matrix ops: `*Matmul*.cpp`, `*Mm*.cpp`
- Attention: `*Attention*.cpp`, `*Flash*.cpp`

**By parameter patterns:**
- IntArray: Search for `OptionalIntArrayRef`
- string_view: Search for `c10::string_view`
- Custom dtypes: Search for `TensorWrapper`

**Recommended reading:**
- `TanhKernelNpuOpApi.cpp` - simplest example
- `TopKKernelNpuOpApi.cpp` - multiple outputs
- `DynamicBlockQuantNpuOpApi.cpp` - quantization with TensorWrapper

## Standard Code Structure

```cpp
namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// Helper functions
c10::SmallVector<int64_t, SIZE> calc_output_size(const at::Tensor& x) {
    auto shape = op_infer::array_to_small_vector(x.sizes());
    return shape;
}
}

ReturnType npu_operator_name(
    const at::Tensor& input,
    c10::string_view str_param,
    c10::OptionalIntArrayRef int_array_param,
    const c10::optional<at::Tensor>& optional_tensor) {

    // 1. Validate inputs
    TORCH_CHECK(input.dim() >= 2, "input must be 2D+", OPS_ERROR(ErrCode::PARAM));

    // 2. Handle optional parameters
    at::Tensor optional_val;
    if (optional_tensor.has_value() && optional_tensor.value().defined()) {
        optional_val = optional_tensor.value();
    }

    // 3. Calculate output shapes
    auto output_shape = calc_output_size(input);

    // 4. Create output tensors
    at::Tensor output = npu_preparation::apply_tensor_without_format(
        output_shape, input.options());

    // 5. Execute aclnn operator
    EXEC_NPU_CMD(aclnnOperatorName, input, str_param.data(),
                 int_array_param, optional_val, output);

    return output;
}

}  // namespace op_api
```

## Key Patterns

### Custom Output Dtypes (Quantization)

```cpp
aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
at::ScalarType y_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
at::Tensor y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(y_dtype));
TensorWrapper y_wrapper = {y, y_acltype};
EXEC_NPU_CMD(aclnnOperator, input, y_wrapper, output);
```

### Workspace Management

```cpp
uint64_t workspace_size = 0;
aclOpExecutor* executor = nullptr;
aclnnOperatorNameGetWorkspaceSize(..., &workspace_size, &executor);

at::Tensor workspace;
if (workspace_size > 0) {
    workspace = npu_preparation::apply_tensor_without_format(
        {static_cast<int64_t>(workspace_size)}, input.options().dtype(at::kByte));
}
EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnOperatorName, workspace, workspace_size, executor, ...);
```

For complete examples, refer to similar operators in the codebase.
