# torch\_npu.npu\_quant\_fusion\_attention

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950DT</term>                                |    √    |

## Function

Implements FP8 quantized fused computation of the Transformer attention score. Supports `float8_e4m3fn` and `hifloat8` quantized inputs with dequant scale computation. The computation formula is the same as that of `torch_npu.npu_fusion_attention`.

$$
attention\_out = Dropout(Softmax(Mask(scale*(pse+query*key^T), atten\_mask)), keep\_prob)*value
$$

## Prototype

```python
torch_npu.npu_quant_fusion_attention(query, key, value, head_num, input_layout, *, d_scale_q, d_scale_k, d_scale_v, p_scale=None, scale=1., query_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)
```

## Parameters

- **`query`** (`Tensor`): Required. Quantized input tensor. The data type can be `float8_e4m3fn`. The data layout can be ND. When `query_dtype` is set to `torch_npu.hifloat8`, the data type can be `uint8`. For details about the overall constraints, see [Constraints](#constraints).
- **`key`** (`Tensor`): Required. Quantized input tensor. The data type and constraints are the same as those of `query`.
- **`value`** (`Tensor`): Required. Quantized input tensor. The data type and constraints are the same as those of `query`.
- **`head_num`** (`int`): Required. Number of heads. The data type can be `int64`.
- **`input_layout`** (`str`): Required. Data layout of the input `query`, `key`, and `value`. Supported layouts: `BSH`, `SBH`, `BSND`, `BNSD`, and `TND`. For details, see the `input_layout` parameter description in [torch\_npu.npu\_fusion\_attention](torch_npu-npu_fusion_attention.md).
- **`d_scale_q`** (`Tensor`): Required. Dequantization scale for `query` at per-block granularity. The data type can be `float32`.
- **`d_scale_k`** (`Tensor`): Required. Dequantization scale for `key` at per-block granularity. The data type can be `float32`.
- **`d_scale_v`** (`Tensor`): Required. Dequantization scale for `value` at per-block granularity. The data type can be `float32`.
- **`p_scale`** (`Tensor`): Optional. Per-token dequantization scale. The data type can be `float32`.
- **`scale`** (`float`): Optional. Scaling factor used as a scalar value in the attention score computation. The data type must be `float`. The default value is `1.0`.
- **`query_dtype`** (`int`): Optional. Data type of the quantized input. When the input is `uint8` carrying `hifloat8` data, set this parameter to `torch_npu.hifloat8`. This parameter is introduced in TorchNPU 7.3.0.

## Outputs

Seven outputs are generated in the following types: `Tensor`, `Tensor`, `Tensor`, `Tensor`, `int`, `int`, and `int`, respectively.

- The first output is a `Tensor` representing the final output $attention\_out$ of the computation formula. The data type is `bfloat16`.
- The second output is a `Tensor` representing the intermediate maximum value of the softmax computation, used for backward computation. The data type is `float32`.
- The third output is a `Tensor` representing the intermediate sum value of the softmax calculation, used for backward computation. The data type is `float32`.
- The fourth output is a reserved `Tensor`, which is currently not used.
- The fifth output is an `int` reserved for future use.
- The sixth output is an `int` reserved for future use.
- The seventh output is an `int` reserved for future use.

## Constraints

- This API is used only in training scenarios.
- This API does not support graph mode or `aclgraph`.
- This API is only supported on <term>Ascend 950DT</term>.
- The data types of `query`, `key`, and `value` must be identical.
- The `input_layout` of `query`, `key`, and `value` must be identical.
- Shape descriptions for `query`, `key`, and `value`:
    - The shapes of `key` and `value` must be identical.
    - The input tensors must be 3D or 4D.
- When the input is `uint8` carrying `hifloat8` data, the optional parameter `query_dtype` must be set to `torch_npu.hifloat8`.

## Examples

Single-operator call (`float8_e4m3fn` inputs):

```python
import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices


@SupportedDevices(['Ascend950'])
def example_npu_quant_fusion_attention():
    # B: batch size, N: head num, Sq: query sequence length, Skv: key/value sequence length, D: head dim.
    B, N, Sq, Skv, D = 1, 8, 256, 256, 128
    scale = 0.08838

    query = torch.randn(B, N, Sq, D, dtype=torch.float16)
    key = torch.randn(B, N, Skv, D, dtype=torch.float16)
    value = torch.randn(B, N, Skv, D, dtype=torch.float16)

    # Calculate per-block quantize scale
    PER_BLOCK_Q_SIZE = 128
    PER_BLOCK_KV_SIZE = 256
    scale_q = torch.randn(B, N, (Sq + PER_BLOCK_Q_SIZE - 1) // PER_BLOCK_Q_SIZE, 1, dtype=torch.float32)
    scale_k = torch.randn(B, N, (Skv + PER_BLOCK_KV_SIZE - 1) // PER_BLOCK_KV_SIZE, 1, dtype=torch.float32)
    scale_v = torch.randn(B, N, (Skv + PER_BLOCK_KV_SIZE - 1) // PER_BLOCK_KV_SIZE, 1, dtype=torch.float32)

    d_scale_q = 1 / scale_q
    d_scale_k = 1 / scale_k
    d_scale_v = 1 / scale_v
    p_scale = torch.randn(1, dtype=torch.float32)

    # Quantize input to float8_e4m3fn
    query_fp8 = query.to(torch.float8_e4m3fn)
    key_fp8 = key.to(torch.float8_e4m3fn)
    value_fp8 = value.to(torch.float8_e4m3fn)

    attention_out, softmax_max, softmax_sum, _, _, _, _ = torch_npu.npu_quant_fusion_attention(
        query_fp8.npu(), key_fp8.npu(), value_fp8.npu(),
        head_num=N, input_layout="BNSD", scale=scale,
        d_scale_q=d_scale_q.npu(), d_scale_k=d_scale_k.npu(),
        d_scale_v=d_scale_v.npu(), p_scale=p_scale.npu())

    print(f"attention_out shape: {attention_out.shape}")


if __name__ == "__main__":
    if torch.npu.is_available():
        example_npu_quant_fusion_attention()
```