# (beta) torch_npu.contrib.npu_fused_attention

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Fuses BERT self-attention computations.

## Prototype

```python
torch_npu.contrib.npu_fused_attention(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, scale=1, keep_prob=0)
```

## Parameters

- **`hidden_states`** (`Tensor`): Required. Hidden states of the last layer. The data type supports `torch.float16`. The data format supports FRACTAL_NZ. The shape is 2D, where dimension 0 must be an integer multiple of 32 and dimension 1 can be 768 or 1024.
- **`attention_mask`** (`Tensor`): Required. Mask tensor used to mask out token positions that do not need to participate in attention computation (such as padding positions). The data type supports `torch.float16`. The data format supports FRACTAL_NZ. This parameter is 4D with shape `(B, 1, S, S)`, where B is the batch size and S is the sequence length.
- **`query_kernel`** (`Tensor`): Required. Query weight matrix. The data type supports `torch.float16`. The data format supports FRACTAL_NZ. The shape is 2D, and dimension 0 can be 768 or 1024.
- **`key_kernel`** (`Tensor`): Required. Key weight matrix. The data type supports `torch.float16`. The data format supports FRACTAL_NZ. The shape is 2D, and dimension 0 can be 768 or 1024.
- **`value_kernel`** (`Tensor`): Required. Value weight matrix. The data type supports `torch.float16`. The data format supports FRACTAL_NZ. The shape is 2D, and dimension 0 can be 768 or 1024.
- **`query_bias`** (`Tensor`): Required. Query bias. The data type supports `torch.float16`. The data format supports ND.
- **`key_bias`** (`Tensor`): Required. Key bias. The data type supports `torch.float16`. The data format supports ND.
- **`value_bias`** (`Tensor`): Required. Value bias. The data type supports `torch.float16`. The data format supports ND.
- **`scale`** (`double`): Optional. Scaling factor used to compute attention scores. The default value is `1`.
- **`keep_prob`** (`float`): Optional. Probability of retaining data during computation, equal to `1 - drop_rate`. The default value is `0`.

## Return Values

`Tensor`

Self-attention computation result.

## Constraints

- Data format requirements: `hidden_states`, `attention_mask`, `query_kernel`, `key_kernel`, and `value_kernel` must be FRACTAL_NZ (format ID 29), and `query_bias`, `key_bias`, and `value_bias` must be ND (format ID 2).
- On Atlas A2/A3 training products, set `torch_npu.npu.config.allow_internal_format = True` before the format conversion; otherwise `npu_format_cast` silently degrades FRACTAL_NZ to ND, causing the format check to fail.
- This API is a fused implementation for BERT self-attention, and the underlying operator performs tiling targeting the standard BERT configuration. The sequence length S (dimensions 2 and 3 of `attention_mask`) must satisfy the operator's constraints. The official examples and tests all use S=512, while a smaller S (for example, S=32) triggers a tiling failure.
- `scale` and `keep_prob` must be passed as positional arguments; keyword arguments (for example, `scale=0.125, keep_prob=0.5`) raise `TypeError: apply() takes no keyword arguments`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.config.allow_internal_format = True
>>> hidden_states = torch_npu.npu_format_cast(torch.rand(12288, 1024).npu().half(), 29)
>>> attention_mask = torch_npu.npu_format_cast(torch.rand(24, 1, 512, 512).npu().half(), 29)
>>> query_kernel = torch_npu.npu_format_cast(torch.rand(1024, 1024).npu().half(), 29)
>>> key_kernel = torch_npu.npu_format_cast(torch.rand(1024, 1024).npu().half(), 29)
>>> value_kernel = torch_npu.npu_format_cast(torch.rand(1024, 1024).npu().half(), 29)
>>> query_bias = torch_npu.npu_format_cast(torch.rand(1024).npu().half(), 2)
>>> key_bias = torch_npu.npu_format_cast(torch.rand(1024).npu().half(), 2)
>>> value_bias = torch_npu.npu_format_cast(torch.rand(1024).npu().half(), 2)
>>> context_layer = torch_npu.contrib.npu_fused_attention(
...     hidden_states, attention_mask, query_kernel, key_kernel, value_kernel,
...     query_bias, key_bias, value_bias, 0.125, 0.5)
>>> print(context_layer.shape)
```
