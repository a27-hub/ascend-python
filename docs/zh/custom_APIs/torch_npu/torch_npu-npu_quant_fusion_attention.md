# torch\_npu.npu\_quant\_fusion\_attention

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950DT</term> | √ |

## 功能说明

实现FP8量化的"Transformer Attention Score"融合计算，支持`float8_e4m3fn`和`hifloat8`量化输入，通过dequant scale进行反量化计算。计算公式与`torch_npu.npu_fusion_attention`一致。

$$
attention\_out = Dropout(Softmax(Mask(scale*(pse+query*key^T), atten\_mask)), keep\_prob)*value
$$

## 函数原型

```python
torch_npu.npu_quant_fusion_attention(query, key, value, head_num, input_layout, *, d_scale_q, d_scale_k, d_scale_v, p_scale=None, scale=1., query_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)
```

## 参数说明

- **query**（`Tensor`）：必选参数，量化后的输入张量，数据类型支持`float8_e4m3fn`，数据格式支持$ND$。当query_dtype指定为`torch_npu.hifloat8`时，数据类型支持`uint8`。综合约束请见[约束说明](#约束说明)。
- **key**（`Tensor`）：必选参数，量化后的输入张量，数据类型和约束同`query`。
- **value**（`Tensor`）：必选参数，量化后的输入张量，数据类型和约束同`query`。
- **head_num**（`int`）：必选参数，代表head个数，数据类型支持`int64`。
- **input_layout**（`string`）：必选参数，代表输入`query`、`key`、`value`的数据排布格式，支持BSH、SBH、BSND、BNSD、TND。具体含义请参考[torch\_npu.npu\_fusion\_attention](torch_npu-npu_fusion_attention.md)中的`input_layout`参数说明。
- **d_scale_q**（`Tensor`）：必选参数，`query`的反量化scale，per-block粒度，数据类型支持`float32`。
- **d_scale_k**（`Tensor`）：必选参数，`key`的反量化scale，per-block粒度，数据类型支持`float32`。
- **d_scale_v**（`Tensor`）：必选参数，`value`的反量化scale，per-block粒度，数据类型支持`float32`。
- **p_scale**（`Tensor`）：可选参数，per-token的量化scale，数据类型支持`float32`。
- **scale**（`float`）：可选参数，代表缩放系数，作为Attention Score计算的scalar值，数据类型支持`float`，默认值为1。
- **query_dtype**（`int`）：可选参数，表示量化输入的dtype。当输入为`uint8`承载的`hifloat8`时，需设置为`torch_npu.hifloat8`。该参数为TorchNPU 7.3.0版本新增参数。

## 输出说明

共7个输出，类型依次为**Tensor、Tensor、Tensor、Tensor、int、int、int。**

- 第1个输出为`Tensor`，代表最终输出$attention\_out$，数据类型为`bfloat16`。
- 第2个输出为`Tensor`，Softmax计算的Max中间结果，用于反向计算，数据类型为`float32`。
- 第3个输出为`Tensor`，Softmax计算的Sum中间结果，用于反向计算，数据类型为`float32`。
- 第4个输出为`Tensor`，预留参数，暂未使用。
- 第5个输出为`int`，预留参数，暂未使用。
- 第6个输出为`int`，预留参数，暂未使用。
- 第7个输出为`int`，预留参数，暂未使用。

## 约束说明

- 该接口仅在训练场景下使用。
- 该接口暂不支持图模式，不支持aclgraph。
- 该接口仅支持<term>Ascend 950DT</term>。
- 输入`query`、`key`、`value`的数据类型必须一致。
- 输入`query`、`key`、`value`的`input_layout`必须一致。
- 输入`query`、`key`、`value`的shape说明：
    - 输入`key`和`value`的shape必须一致。
    - 输入支持3维或4维张量。
- 当输入为`uint8`承载的`hifloat8`时，必须设置可选参数`query_dtype`为`torch_npu.hifloat8`。

## 调用示例

单算子模式调用（`float8_e4m3fn`类型输入）：

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