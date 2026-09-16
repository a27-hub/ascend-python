# torch\_npu.npu\_anti\_mx\_quant

## Supported Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## Function

- Dequantizes a `float4/float8` tensor, which is quantized by `npu_dynamic_mx_quant`, to the `float16/bfloat16/float32` format. This API is the inverse process of `npu_dynamic_mx_quant`.

- Formula:
  $$
  X_{dq} = X_q \times 2^{sf - bias}
  $$

  where $sf$ is the scale factor provided by the input `mxscale`; $bias$ is the exponent bias, which is 127 for the `float8_e8m0` format; $X_q$ is the quantized `float4/float8` tensor; and $X_{dq}$ is the dequantized `float16/bfloat16/float32` tensor.

## Prototype

```python
torch_npu.npu_anti_mx_quant(x, mxscale, *, axis=-1, dst_type=15, src_type=292) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor to be dequantized. The data type can be `float4_e2m1`, `float4_e1m2`, `float8_e5m2`, or `float8_e4m3fn`, and the data layout can be `ND`. The shape can be 1D to 7D. For `float8` inputs, the data type of `x` is the corresponding `float8` type and non-contiguous tensors are supported. For `float4` inputs, the actual data type of `x` is `uint8` (every two `float4` values are packed into one `uint8`, and the size of the last dimension is half the number of actual elements), and non-contiguous tensors are not supported.
- **`mxscale`** (`Tensor`): Required. Quantization scale that participates in the dequantization computation, output by `npu_dynamic_mx_quant`. The logical data type is `float8_e8m0` (actual storage data type is `uint8`), and the data layout can be `ND`. The shape can be 2D to 8D.
- **`*`**: Position delimiter indicating preceding variables are position-dependent arguments that must be entered in order, while succeeding variables are optional keyword arguments that must be assigned using key-value pairs in any order (omitted arguments will use default values).
- **`axis`** (`int`): Optional. Dequantization axis. Currently, only the tail axis is supported, that is, the value can be `-1` or `D-1`, where `D` is the number of dimensions of the input `x`. The default value is `-1`.
- **`dst_type`** (`int`): Optional. Data type of the output `y`. Currently, the supported values are `{5, 6, 15}`, indicating that the data type of the output `y` is `{5: float16, 6: float32, 15: bfloat16}` respectively. The default value is `15` (`bfloat16`). Alternatively, `torch.float16`, `torch.float32`, or `torch.bfloat16` can be directly passed in.
- **`src_type`** (`int`): Optional. Data type of the input `x`. Currently, the supported values are `{291, 292, 296, 297}`, indicating that the data type of the input `x` is `{291: float8_e5m2, 292: float8_e4m3fn, 296: float4_e2m1, 297: float4_e1m2}` respectively. The default value is `292` (`float8_e4m3fn`).

## Return Values

- **`y`** (`Tensor`): Dequantization result. The data type is specified by `dst_type` and can be `float16/bfloat16/float32`, and the data layout can be `ND`. For `float8` inputs, the shape of `y` is the same as that of the input `x`. For `float4` inputs, the size of the last dimension of `y` is twice that of the input `x` (stored as `uint8`), and the other dimensions are the same as those of the input `x`.

## Constraints

- API supported in inference scenarios.
- API supports single-operator mode and graph mode calls.
- The inputs `x` and `mxscale` must be used as a pair. It is recommended that `mxscale` be obtained through `torch_npu.npu_dynamic_mx_quant` (with `block_size` set to 32).
- Shape constraints between the inputs `x` and `mxscale` (when `block_size` is 32):
  - rank\(mxscale\) = rank\(x\) + 1
  - axis\_change = axis if axis >= 0 else axis + rank\(x\) (only the tail axis is currently supported)
  - mxscale.shape\[axis\_change\] = \(ceil\(x.shape\[axis\] / block\_size\) + 2 - 1\) / 2
  - mxscale.shape\[-1\] = 2
  - Other dimensions are identical to those of `x`.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    def anti_mx_quant_test(x_dtype, dst_type, src_type):
        # method 1: generate inputs manually
        x = torch.randn((16, 128), dtype=torch.float32).to(dtype=torch.float8_e5m2).npu()
        mxscale = torch.randint(120, 140, (16, 2, 2), dtype=torch.float32).to(dtype=torch.float8_e8m0fnu).npu()
        y = torch_npu.npu_anti_mx_quant(x, mxscale, axis=-1, dst_type=dst_type, src_type=src_type)
        print("AntiMxQuant result:")
        print("x:\n", x.cpu())
        print("mxscale:\n", mxscale.cpu())
        print("y:\n", y.cpu())

        # method 2: generate inputs by npu_dynamic_mx_quant
        x_tmp = torch.randn((16, 128), dtype=x_dtype).npu()
        x, mxscale = torch_npu.npu_dynamic_mx_quant(x_tmp, axis=-1, round_mode="rint", dst_type=torch.float8_e5m2, block_size=32, scale_alg=0)
        y = torch_npu.npu_anti_mx_quant(x, mxscale, axis=-1, dst_type=dst_type, src_type=src_type)
        print("AntiMxQuant result:")
        print("x:\n", x.cpu())
        print("mxscale:\n", mxscale.cpu())
        print("y:\n", y.cpu())

    if __name__ == "__main__":
        anti_mx_quant_test(torch.bfloat16, 15, 291)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair

    class AntiMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, mxscale, axis=-1, dst_type=15, src_type=291):
            return torch_npu.npu_anti_mx_quant(x, mxscale, axis=axis, dst_type=dst_type, src_type=src_type)

    def anti_mx_quant_test(x_dtype, dst_type):
        x_tmp = torch.randn((16, 128), dtype=x_dtype).npu()
        x, mxscale = torch_npu.npu_dynamic_mx_quant(x_tmp, axis=-1, round_mode="rint", dst_type=torch.float8_e5m2, block_size=32, scale_alg=0)
        # In graph mode, mxscale must be converted from the uint8 storage type back to the float8_e8m0fnu type.
        mxscale = mxscale.view(torch.float8_e8m0fnu)
        model = AntiMxQuantModel()
        model.to('npu')
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        y = model(x, mxscale, axis=-1, dst_type=dst_type, src_type=291)
        y = y.cpu()
        print("AntiMxQuant result:")
        print("x:\n", x.cpu())
        print("y:\n", y)

    if __name__ == "__main__":
        anti_mx_quant_test(torch.bfloat16, 15)
    ```
