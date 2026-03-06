---
name: op-adaptation
description: Integrate CANN (Ascend) operators into torch_npu backend following the standardized workflow. Use when the user asks to integrate, adapt, or add CANN/aclnn operators to torch_npu, implement NPU operators, or work with operator definition files (.md documentation and _def.cpp files). Handles the complete workflow from parsing operator specifications to building, installing, and testing the integrated operator.
---

# CANN Operator Integration

Integrate CANN (Ascend) operators into the torch_npu backend following a standardized workflow.

## Workflow

Given operator documentation (.md) and definition file (_def.cpp):

1. **Parse inputs** - Extract parameters, types, constraints from both files
2. **Generate YAML** - Add entry to `op_plugin/config/op_plugin_functions.yaml`
3. **Generate C++** - Create `op_plugin/ops/opapi/{Name}KernelNpuOpApi.cpp`
4. **Build** - Run `bash ci/build.sh --python=3.10`
5. **Test** - Create and run `test_{operator_name}.py`

## Step 1: Parse Inputs

**From .md file:**
- aclnn function signature (GetWorkspaceSize parameters)
- Parameter types, input/output markers
- Dtype/shape constraints

**From _def.cpp file:**
- REQUIRED/OPTIONAL markers
- Default values from Attr(...)
- SymInt types

**Conflict resolution:**
- Optionality: Use _def.cpp ParamType
- Defaults: Use _def.cpp Attr
- Naming: Use _def.cpp snake_case
- Constraints: Use .md documentation

## Step 2: Generate YAML

Add to `op_plugin/config/op_plugin_functions.yaml`:

```yaml
custom:
  - func: npu_operator_name(Tensor input, *, Tensor? optional=None, int attr=1) -> (Tensor, Tensor)
    op_api: all_version
    exposed: all_version
```

### YAML Configuration Fields

| Field | Description |
|-------|-------------|
| `all_version` | All currently supported PyTorch versions, can be referenced directly |
| `official` | PyTorch native operators |
| `custom` | Custom operators (prefixed with npu_) |
| `symint` | Operators supporting SymInt type parameters, must be configured together with custom |
| `func` | Operator schema: `name(parameter_list) -> return_type` |
| `op_api` | Version range supporting aclnn calls |
| `exposed` | Version range for commercial operators (typically only set for forward operators) |

Rules:
- Required params before `*`, optional after
- Optional tensors: `Tensor?`
- SymInt needs both `custom:` and `symint:` sections

## Step 3: Generate C++ Implementation

Create `op_plugin/ops/opapi/{Name}KernelNpuOpApi.cpp`:

**CRITICAL: Function signature must match generated interface**

After adding YAML, check the generated signature in `build/pytorch/third_party/op-plugin/op_plugin/OpApiInterface.h`:
- Use `at::OptionalIntArrayRef` for `int[]?` parameters (NOT `c10::optional<at::IntArrayRef>`)
- Use `c10::string_view` for `str` parameters (NOT `std::string`)
- Use `c10::optional<at::ScalarType>` for `ScalarType?` parameters

```cpp
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// Helper functions for shape calculation
c10::SmallVector<int64_t, SIZE> calc_output_size(const at::Tensor& x) {
    auto shape = op_infer::array_to_small_vector(x.sizes());
    // Modify as needed
    return shape;
}
}

ReturnType npu_operator_name(
    const at::Tensor& input,
    c10::string_view str_param,              // Use c10::string_view for str
    at::OptionalIntArrayRef int_array_param, // Use at::OptionalIntArrayRef for int[]?
    const c10::optional<at::Tensor>& optional_tensor,
    int64_t int_param,
    c10::optional<at::ScalarType> dtype_param) {

    // 1. Validate inputs
    TORCH_CHECK(input.dim() >= 2 && input.dim() <= 8,
                "input must be 2-8 dimensional",
                OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
                "input must be FLOAT16",
                OPS_ERROR(ErrCode::TYPE));

    // 2. Handle string_view parameters
    std::string str_param_str(str_param);  // Convert to std::string if needed
    const char* str_param_cstr = str_param_str.c_str();

    // 3. Handle OptionalIntArrayRef parameters
    c10::optional<at::IntArrayRef> int_array_for_exec;
    if (int_array_param.has_value()) {
        int_array_for_exec = int_array_param.value();
    }

    // 4. Handle optional tensor parameters
    at::Tensor optional_tensor_val;
    if (optional_tensor.has_value() && optional_tensor.value().defined()) {
        optional_tensor_val = optional_tensor.value();
    }

    // 5. Calculate output shapes
    auto output_shape = calc_output_size(input);

    // 6. Create output tensors
    at::Tensor output = npu_preparation::apply_tensor_without_format(
        output_shape, input.options());

    // 7. Execute aclnn operator
    EXEC_NPU_CMD(aclnnOperatorName, input, str_param_cstr,
                 int_array_for_exec, optional_tensor_val, output);

    return output;
}

}  // namespace op_api
```

**For custom output dtypes:**
```cpp
aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
at::ScalarType y_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
at::Tensor y = npu_preparation::apply_tensor_without_format(
    y_shape, c10::dtype(y_dtype));
TensorWrapper y_wrapper = {y, y_acltype};
EXEC_NPU_CMD(aclnnOperator, input, y_wrapper, output);
```

## Step 4: Build and Install

```bash
# Note: --pytorch parameter is not supported, use only --python
bash ci/build.sh --python=3.10
pip install --force-reinstall build/pytorch/dist/torch_npu-*.whl
```

**Build troubleshooting:**
- If build fails, check `build_*.log` for errors
- For signature mismatch errors, verify function signature matches generated interface
- Clean build if needed: `rm -rf build && bash ci/build.sh --python=3.10`

## Step 5: Create Tests

Create `test_{operator_name}.py`:

```python
import torch
import torch_npu

def test_basic():
    x = torch.randn(4, 32, dtype=torch.float16, device='npu')
    output = torch_npu.npu_operator_name(x)
    assert output.shape == expected_shape
    print("✓ Basic test passed")

def test_optional_params():
    x = torch.randn(4, 32, dtype=torch.float16, device='npu')
    param = torch.ones(32, dtype=torch.float16, device='npu')
    output = torch_npu.npu_operator_name(x, optional_param=param)
    print("✓ Optional parameter test passed")

if __name__ == "__main__":
    test_basic()
    test_optional_params()
    print("\n✓ All tests passed!")
```

## Complete Example: DynamicQuant

**YAML:**
```yaml
- func: npu_dynamic_quant(Tensor x, *, Tensor? smooth_scales=None, int dst_type=1) -> (Tensor y, Tensor scale)
  op_api: all_version
  exposed: all_version
```

**C++ (key parts):**
```cpp
std::tuple<at::Tensor, at::Tensor> npu_dynamic_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& smooth_scales,
    int64_t dst_type)
{
    // Validate
    TORCH_CHECK(x.dim() >= 2 && x.dim() <= 8, "x must be 2-8D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||
                x.scalar_type() == at::ScalarType::BFloat16,
                "x must be FP16/BF16", OPS_ERROR(ErrCode::TYPE));

    // Validate optional smooth_scales
    if (smooth_scales.has_value() && smooth_scales.value().defined()) {
        const auto& smooth = smooth_scales.value();
        TORCH_CHECK(smooth.scalar_type() == x.scalar_type(),
                    "smooth_scales dtype must match x", OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(smooth.dim() == 1, "smooth_scales must be 1D", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(smooth.size(0) == x.size(-1),
                    "smooth_scales size must match x last dim", OPS_ERROR(ErrCode::PARAM));
    }

    // Validate dst_type constraints
    if (dst_type == 23) {  // INT4
        TORCH_CHECK(x.size(-1) % 2 == 0, "INT4: last dim must be even", OPS_ERROR(ErrCode::PARAM));
    }
    if (dst_type == 6) {  // INT32
        TORCH_CHECK(x.size(-1) % 8 == 0, "INT32: last dim divisible by 8", OPS_ERROR(ErrCode::PARAM));
    }

    // Calculate shapes (scale = x.shape without last dim)
    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    auto scale_shape = y_shape;
    if (scale_shape.size() > 0) scale_shape.pop_back();

    // Create outputs
    aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
    at::ScalarType y_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    at::Tensor y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(y_dtype));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(
        scale_shape, c10::dtype(at::ScalarType::Float));

    // Prepare optional param
    at::Tensor smooth_tensor;
    if (smooth_scales.has_value() && smooth_scales.value().defined()) {
        smooth_tensor = smooth_scales.value();
    }

    // Execute
    TensorWrapper y_wrapper = {y, y_acltype};
    EXEC_NPU_CMD(aclnnDynamicQuant, x, smooth_tensor, y_wrapper, scale);

    return std::make_tuple(y, scale);
}
```

## Complete Example: FFN (with IntArray and string_view)

**YAML:**
```yaml
- func: npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, *, int[]? expert_tokens=None, int[]? expert_tokens_index=None, Tensor? bias1=None, Tensor? bias2=None, int inner_precise=0) -> Tensor
  op_api: all_version
  exposed: all_version
```

**C++ implementation:**
```cpp
at::Tensor npu_ffn(
    const at::Tensor& x,
    const at::Tensor& weight1,
    const at::Tensor& weight2,
    c10::string_view activation,                    // str → c10::string_view
    at::OptionalIntArrayRef expert_tokens,          // int[]? → at::OptionalIntArrayRef
    at::OptionalIntArrayRef expert_tokens_index,
    const c10::optional<at::Tensor>& bias1,
    const c10::optional<at::Tensor>& bias2,
    int64_t inner_precise)
{
    // 1. Validate inputs
    TORCH_CHECK(x.dim() >= 2 && x.dim() <= 8,
                "x must be 2-8 dimensional", OPS_ERROR(ErrCode::PARAM));

    // 2. Handle string_view - convert to std::string for c_str()
    std::string activation_str(activation);
    const char* activation_cstr = activation_str.c_str();

    // 3. Handle OptionalIntArrayRef - pass directly to EXEC_NPU_CMD
    c10::optional<at::IntArrayRef> expert_tokens_for_exec;
    std::vector<int64_t> expert_tokens_counts;

    if (expert_tokens.has_value()) {
        expert_tokens_for_exec = expert_tokens.value();
    } else if (expert_tokens_index.has_value()) {
        // Convert cumulative indices to counts
        auto index_ref = expert_tokens_index.value();
        if (index_ref.size() > 0) {
            expert_tokens_counts.push_back(index_ref[0]);
            for (size_t i = 1; i < index_ref.size(); i++) {
                expert_tokens_counts.push_back(index_ref[i] - index_ref[i-1]);
            }
            expert_tokens_for_exec = expert_tokens_counts;
        }
    }

    // 4. Handle optional tensors
    at::Tensor bias1_tensor, bias2_tensor;
    if (bias1.has_value() && bias1.value().defined()) {
        bias1_tensor = bias1.value();
    }
    if (bias2.has_value() && bias2.value().defined()) {
        bias2_tensor = bias2.value();
    }

    // 5. Calculate output shape (same as x)
    auto output_shape = op_infer::array_to_small_vector(x.sizes());

    // 6. Create output tensor
    at::Tensor y = npu_preparation::apply_tensor_without_format(
        output_shape, x.options());

    // 7. Execute - EXEC_NPU_CMD handles IntArray conversion automatically
    EXEC_NPU_CMD(aclnnFFN, x, weight1, weight2, expert_tokens_for_exec,
                 bias1_tensor, bias2_tensor, activation_cstr, inner_precise, y);

    return y;
}
```

**Key points for FFN example:**
- `str` in YAML → `c10::string_view` in C++ (convert to `std::string` for `c_str()`)
- `int[]?` in YAML → `at::OptionalIntArrayRef` in C++ (NOT `c10::optional<at::IntArrayRef>`)
- Pass `c10::optional<at::IntArrayRef>` directly to `EXEC_NPU_CMD` - no manual ACL API calls
- Support both `expert_tokens` (counts) and `expert_tokens_index` (cumulative) by converting

## Common Issues

**KeyError: 'Found an invalid operator name: npu_xxx'**
- Operator added to wrong YAML section (e.g., `quant:` instead of `custom:`)
- Solution: Move operator to `custom:` section for all npu_ prefixed operators

**Compilation error: `'QInt4x2' is not a member of 'c10::ScalarType'`**
- Use dst_type numeric values: `if (dst_type == 23)` not `if (y_dtype == ScalarType::QInt4x2)`

**Runtime error: `Tensor not implemented for DT_FLOAT`**
- Use TensorWrapper for non-standard dtypes (INT4/INT8/INT32/FLOAT8)

**Build error: `undefined reference to op_api::npu_xxx`**
- YAML function signature must exactly match C++ implementation
- Check generated signature in `build/pytorch/third_party/op-plugin/op_plugin/OpApiInterface.h`
- Common mismatches:
  - `int[]?` in YAML → `at::OptionalIntArrayRef` in C++ (NOT `c10::optional<at::IntArrayRef>`)
  - `str` in YAML → `c10::string_view` in C++ (NOT `std::string`)
  - `ScalarType?` in YAML → `c10::optional<at::ScalarType>` in C++

**Compilation error: `cannot convert '_aclCreateIntArray' to 'aclIntArray*'`**
- Do NOT manually call `_aclCreateIntArray` or `_aclDestroyIntArray`
- Let EXEC_NPU_CMD handle IntArray conversion automatically
- Pass `at::OptionalIntArrayRef` directly to EXEC_NPU_CMD

**Test fails with shape mismatch**
- For INT32 output: last_dim = input_last_dim / 8
- For reduction ops: remove reduced dimensions from output shape

## Best Practices

- **Always check generated signature first**: After adding YAML, build once and check `build/pytorch/third_party/op-plugin/op_plugin/OpApiInterface.h` for the exact function signature
- **Type mapping rules**:
  - `int[]?` → `at::OptionalIntArrayRef` (NOT `c10::optional<at::IntArrayRef>`)
  - `str` → `c10::string_view` (NOT `std::string`)
  - `ScalarType?` → `c10::optional<at::ScalarType>`
  - `Tensor?` → `const c10::optional<at::Tensor>&`
- **IntArray handling**: Pass `at::OptionalIntArrayRef` directly to `EXEC_NPU_CMD` - do NOT manually call `_aclCreateIntArray` or `_aclDestroyIntArray`
- **string_view handling**: Convert to `std::string` first if you need `c_str()`: `std::string str(string_view_param)`
- Validate inputs before aclnn call (dimensions, dtypes, constraints)
- Use helper functions in anonymous namespace for shape calculations
- Follow _def.cpp naming (snake_case), not .md (camelCase)
- Check `has_value() && defined()` for optional tensors
- Use TensorWrapper for custom output dtypes
- Test basic functionality, optional params, and edge cases
- Clean build (`rm -rf build`) if you encounter persistent signature mismatch errors
